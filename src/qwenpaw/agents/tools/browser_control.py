# -*- coding: utf-8 -*-
# flake8: noqa: E501
"""Browser automation tool with the final layered browser_use API.

Public API:

browser_use(
    action_type="browser_management | basic_page_operation | advanced_page_operation",
    operation="operation_name",
    extra_params={...},
)
"""

import asyncio
import atexit
import json
import logging
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.parse import urlparse

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from ...config import (
    get_playwright_chromium_executable_path,
    get_system_default_browser,
    is_running_in_container,
)
from ...config.context import get_current_workspace_dir
from ...constant import WORKING_DIR
from .browser_snapshot import build_role_snapshot_from_aria

logger = logging.getLogger(__name__)


_TRUSTED_BROWSER_KEYWORDS = frozenset(
    {
        "chrome",
        "chromium",
        "edge",
        "firefox",
        "brave",
        "vivaldi",
        "opera",
        "360se",
        "yandex",
        "tor",
    },
)

_CDP_SCAN_MAX_PORT = 65535
_CDP_SCAN_CHUNK_SIZE = 512
_SNAPSHOT_MAX_CHARS = 24000
_RESULT_MAX_CHARS = 8000
_BROWSER_IDLE_TIMEOUT = 600.0


def _tool_response(text: str) -> ToolResponse:
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def _json_response(payload: dict[str, Any]) -> ToolResponse:
    return _tool_response(json.dumps(payload, ensure_ascii=False, indent=2))


def _ok(**payload: Any) -> ToolResponse:
    payload.setdefault("ok", True)
    return _json_response(payload)


def _error(message: str, **payload: Any) -> ToolResponse:
    return _json_response({"ok": False, "error": message, **payload})


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...<truncated>", True


def _coerce_extra_params(extra_params: Any) -> dict[str, Any]:
    if extra_params is None:
        return {}
    if isinstance(extra_params, dict):
        return extra_params
    if isinstance(extra_params, str):
        try:
            parsed = json.loads(extra_params)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_output_path(path: str) -> str:
    if Path(path).is_absolute():
        return path
    base_dir = (get_current_workspace_dir() or WORKING_DIR) / "browser"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / path)


def _validate_executable_path(executable_path: str) -> None:
    if not executable_path:
        return
    path = Path(executable_path)
    name = path.name.lower()
    if not any(keyword in name for keyword in _TRUSTED_BROWSER_KEYWORDS):
        raise ValueError(
            f"executable_path rejected: '{path.name}' is not a trusted browser binary",
        )
    if not path.is_file():
        raise ValueError(f"executable_path rejected: '{executable_path}' does not exist")


def _ensure_playwright_async():
    try:
        from playwright.async_api import async_playwright

        return async_playwright
    except ImportError as exc:
        raise ImportError(
            "Playwright not installed. Install with: "
            f"'{sys.executable}' -m pip install playwright && "
            f"'{sys.executable}' -m playwright install",
        ) from exc


def _chromium_launch_args() -> list[str]:
    args: list[str] = []
    if is_running_in_container() or sys.platform == "win32":
        args.append("--no-sandbox")
    if is_running_in_container():
        args.append("--disable-dev-shm-usage")
    if sys.platform == "win32":
        args.append("--disable-gpu")
    return args


def _resolve_browser_target(executable_path: str = "") -> tuple[Optional[str], Optional[str]]:
    if executable_path:
        _validate_executable_path(executable_path)
        return "chromium", executable_path
    default_kind, default_path = get_system_default_browser()
    if default_kind == "chromium" and default_path:
        return "chromium", default_path
    playwright_chromium = get_playwright_chromium_executable_path()
    if playwright_chromium:
        return "chromium", playwright_chromium
    if default_kind == "webkit" and sys.platform == "darwin":
        return "webkit", None
    return None, None


def _normalize_cdp_endpoint(cdp_endpoint: Any) -> tuple[str, int]:
    if cdp_endpoint is None or cdp_endpoint == "":
        return "", 0
    if isinstance(cdp_endpoint, int):
        return f"http://127.0.0.1:{cdp_endpoint}", cdp_endpoint
    raw = str(cdp_endpoint).strip()
    if not raw:
        return "", 0
    if raw.isdigit():
        port = int(raw)
        return f"http://127.0.0.1:{port}", port
    url = raw if "://" in raw else f"http://{raw}"
    parsed = urlparse(url)
    return url, parsed.port or 0


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _workspace_key() -> tuple[str, str]:
    workspace = get_current_workspace_dir()
    if workspace:
        return workspace.name, str(workspace)
    return "default", ""


def _new_state(workspace_id: str, workspace_dir: str) -> dict[str, Any]:
    user_data_dir = str(Path(workspace_dir) / "browser" / "user_data") if workspace_dir else ""
    return {
        "workspace_id": workspace_id,
        "workspace_dir": workspace_dir,
        "user_data_dir": user_data_dir,
        "playwright": None,
        "browser": None,
        "context": None,
        "pages": {},
        "refs": {},
        "current_page_id": None,
        "page_counter": 0,
        "headless": True,
        "launch_mode": None,
        "connected_via_cdp": False,
        "cdp_endpoint": None,
        "owned_browser_process": False,
        "browser_process": None,
        "browser_pid": None,
        "pending_downloads": {},
        "pending_file_choosers": {},
        "last_activity_time": 0.0,
        "idle_task": None,
        "last_browser_error": None,
    }


_workspace_states: dict[str, dict[str, Any]] = {}


def _get_state() -> dict[str, Any]:
    workspace_id, workspace_dir = _workspace_key()
    if workspace_id not in _workspace_states:
        _workspace_states[workspace_id] = _new_state(workspace_id, workspace_dir)
    return _workspace_states[workspace_id]


def _is_browser_running(state: dict[str, Any]) -> bool:
    return state.get("context") is not None or state.get("browser") is not None


def _touch(state: dict[str, Any]) -> None:
    state["last_activity_time"] = time.monotonic()


def _reset_runtime_state(state: dict[str, Any]) -> None:
    state["playwright"] = None
    state["browser"] = None
    state["context"] = None
    state["pages"].clear()
    state["refs"].clear()
    state["current_page_id"] = None
    state["page_counter"] = 0
    state["headless"] = True
    state["launch_mode"] = None
    state["connected_via_cdp"] = False
    state["cdp_endpoint"] = None
    state["owned_browser_process"] = False
    state["browser_process"] = None
    state["browser_pid"] = None
    state["pending_downloads"].clear()
    state["pending_file_choosers"].clear()
    state["last_browser_error"] = None
    state["last_activity_time"] = 0.0


def _next_page_id(state: dict[str, Any]) -> str:
    state["page_counter"] += 1
    return f"page_{state['page_counter']}"


def _resolve_page_id(state: dict[str, Any], page_id: Any = "default") -> str:
    normalized = str(page_id or "default").strip() or "default"
    current = state.get("current_page_id")
    if normalized == "default" and current and current in state["pages"]:
        return current
    return normalized


def _attach_page_listeners(state: dict[str, Any], page: Any, page_id: str) -> None:
    state["pending_downloads"].setdefault(page_id, [])
    state["pending_file_choosers"].setdefault(page_id, [])

    def on_download(download: Any) -> None:
        state["pending_downloads"].setdefault(page_id, []).append(download)

    def on_filechooser(chooser: Any) -> None:
        state["pending_file_choosers"].setdefault(page_id, []).append(chooser)

    page.on("download", on_download)
    page.on("filechooser", on_filechooser)


def _register_page(state: dict[str, Any], page: Any, page_id: str) -> None:
    if page_id in state["pages"] and state["pages"][page_id] is page:
        return
    state["pages"][page_id] = page
    state["refs"].setdefault(page_id, {})
    _attach_page_listeners(state, page, page_id)
    state["current_page_id"] = page_id


def _attach_context_listeners(state: dict[str, Any], context: Any) -> None:
    def on_page(page: Any) -> None:
        page_id = _next_page_id(state)
        _register_page(state, page, page_id)

    context.on("page", on_page)


async def _activate_page(state: dict[str, Any], page_id: str) -> Any:
    page = state["pages"].get(page_id)
    if page is None:
        raise ValueError(f"Page '{page_id}' not found")
    state["current_page_id"] = page_id
    try:
        await page.bring_to_front()
    except Exception:
        logger.debug("Failed to bring page %s to front", page_id, exc_info=True)
    return page


async def _get_or_create_page(state: dict[str, Any], page_id: str) -> Any:
    if page_id in state["pages"]:
        return await _activate_page(state, page_id)
    context = state.get("context")
    if context is None:
        raise RuntimeError("Browser not started")
    page = await context.new_page()
    _register_page(state, page, page_id)
    return page


async def _ensure_browser(state: dict[str, Any]) -> bool:
    if _is_browser_running(state):
        _touch(state)
        return True
    response = await _browser_management_start(
        state,
        headed=False,
        disable_cdp=False,
        executable_path="",
        browser_launch_args="",
        cdp_endpoint="",
        cdp_port=0,
    )
    payload = _tool_response_payload(response)
    if payload.get("ok"):
        return True
    state["last_browser_error"] = payload.get("error") or "Browser start failed"
    return False


def _tool_response_payload(response: ToolResponse) -> dict[str, Any]:
    try:
        block = response.content[0]
        text = block.get("text") if isinstance(block, dict) else block.text
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


async def _sleep_after(wait_time: Any) -> None:
    delay = _as_float(wait_time, 0.0)
    if delay > 0:
        await asyncio.sleep(delay)


async def _idle_watchdog(state: dict[str, Any]) -> None:
    try:
        while True:
            await asyncio.sleep(60)
            if not _is_browser_running(state):
                return
            idle = time.monotonic() - float(state.get("last_activity_time") or 0.0)
            if idle >= _BROWSER_IDLE_TIMEOUT:
                await _browser_management_stop(state)
                return
    except asyncio.CancelledError:
        return


def _start_idle_watchdog(state: dict[str, Any]) -> None:
    task = state.get("idle_task")
    if task and not task.done():
        task.cancel()
    state["idle_task"] = asyncio.ensure_future(_idle_watchdog(state))


def _cancel_idle_watchdog(state: dict[str, Any]) -> None:
    task = state.get("idle_task")
    current = asyncio.current_task()
    if task and not task.done() and task is not current:
        task.cancel()
    state["idle_task"] = None


async def _wait_for_cdp_ready(port: int, timeout: float = 15.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    version_url = f"http://127.0.0.1:{port}/json/version"
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib_request.urlopen(version_url, timeout=0.5) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for CDP endpoint on port {port}: {last_error}")


def _start_chromium_process(
    executable_path: str,
    user_data_dir: str,
    headed: bool,
    cdp_port: int,
    browser_launch_args: str,
) -> subprocess.Popen:
    Path(user_data_dir).mkdir(parents=True, exist_ok=True)
    args = [
        executable_path,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-sync",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-session-crashed-bubble",
        "--hide-crash-restore-bubble",
        "--password-store=basic",
    ]
    args.extend(_chromium_launch_args())
    if browser_launch_args:
        args.extend(shlex.split(browser_launch_args, posix=sys.platform != "win32"))
    if not headed:
        args.extend(["--headless=new", "--disable-gpu"])

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "cwd": str(Path(user_data_dir).parent),
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(args, **popen_kwargs)


async def _stop_owned_process(state: dict[str, Any]) -> bool:
    process = state.get("browser_process")
    if process is None:
        return False
    if process.poll() is not None:
        return True
    try:
        if sys.platform == "win32":
            process.terminate()
        else:
            process.send_signal(signal.SIGTERM)
        await asyncio.to_thread(process.wait, 5)
        return True
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            await asyncio.to_thread(process.wait, 5)
            return True
        except Exception:
            return False
    except Exception:
        return False


async def _browser_management_start(
    state: dict[str, Any],
    headed: bool,
    disable_cdp: bool,
    executable_path: str,
    browser_launch_args: str,
    cdp_endpoint: str,
    cdp_port: int,
) -> ToolResponse:
    if _is_browser_running(state):
        return _ok(
            message="Browser already running",
            launch_mode=state.get("launch_mode"),
            page_id=state.get("current_page_id"),
        )

    state["headless"] = not headed

    try:
        browser_kind, resolved_executable = _resolve_browser_target(executable_path)
        async_playwright = _ensure_playwright_async()
        playwright = await async_playwright().start()

        if not disable_cdp:
            if browser_kind != "chromium" or not resolved_executable:
                await playwright.stop()
                return _error("CDP start requires Chrome/Chromium/Edge executable")
            cdp_port = cdp_port or _find_free_local_port()
            cdp_endpoint = f"http://127.0.0.1:{cdp_port}"
            if _is_port_in_use(cdp_port):
                await playwright.stop()
                return _error(f"CDP port {cdp_port} is already in use")
            process = _start_chromium_process(
                executable_path=resolved_executable,
                user_data_dir=state["user_data_dir"],
                headed=headed,
                cdp_port=cdp_port,
                browser_launch_args=browser_launch_args,
            )
            try:
                await _wait_for_cdp_ready(cdp_port)
                browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
            except Exception:
                if process.poll() is None:
                    process.kill()
                    await asyncio.to_thread(process.wait, 5)
                await playwright.stop()
                raise
            contexts = browser.contexts
            context = contexts[0] if contexts else await browser.new_context(accept_downloads=True)
            state.update(
                {
                    "playwright": playwright,
                    "browser": browser,
                    "context": context,
                    "launch_mode": "managed_cdp",
                    "connected_via_cdp": True,
                    "cdp_endpoint": cdp_endpoint,
                    "owned_browser_process": True,
                    "browser_process": process,
                    "browser_pid": process.pid,
                },
            )
        else:
            if browser_kind == "chromium" and resolved_executable:
                launch_args = _chromium_launch_args()
                if browser_launch_args:
                    launch_args.extend(shlex.split(browser_launch_args, posix=sys.platform != "win32"))
                browser = await playwright.chromium.launch(
                    headless=not headed,
                    executable_path=resolved_executable,
                    args=launch_args,
                )
                context = await browser.new_context(accept_downloads=True)
            elif browser_kind == "webkit":
                browser = await playwright.webkit.launch(headless=not headed)
                context = await browser.new_context(accept_downloads=True)
            else:
                browser = await playwright.chromium.launch(headless=not headed)
                context = await browser.new_context(accept_downloads=True)
            state.update(
                {
                    "playwright": playwright,
                    "browser": browser,
                    "context": context,
                    "launch_mode": "playwright",
                    "connected_via_cdp": False,
                    "cdp_endpoint": None,
                    "owned_browser_process": False,
                    "browser_process": None,
                    "browser_pid": None,
                },
            )

        _attach_context_listeners(state, state["context"])
        for page in state["context"].pages:
            _register_page(state, page, "default" if not state["pages"] else _next_page_id(state))
        _touch(state)
        _start_idle_watchdog(state)
        payload = {
            "message": "Browser started",
            "launch_mode": state["launch_mode"],
            "headed": headed,
            "disable_cdp": disable_cdp,
            "page_id": state.get("current_page_id"),
        }
        if state.get("cdp_endpoint"):
            payload["cdp_endpoint"] = state["cdp_endpoint"]
        if state.get("browser_pid"):
            payload["browser_pid"] = state["browser_pid"]
        return _ok(**payload)
    except Exception as exc:
        await _browser_management_stop(state)
        state["last_browser_error"] = str(exc)
        return _error(f"Browser start failed: {exc}")


async def _browser_management_stop(state: dict[str, Any]) -> ToolResponse:
    _cancel_idle_watchdog(state)
    if not _is_browser_running(state) and not state.get("browser_process"):
        _reset_runtime_state(state)
        return _ok(message="Browser not running")

    owned = bool(state.get("owned_browser_process"))
    pid = state.get("browser_pid")
    stopped_process = False
    try:
        if state.get("context") is not None:
            try:
                await state["context"].close()
            except Exception:
                pass
        if state.get("browser") is not None:
            try:
                await state["browser"].close()
            except Exception:
                pass
        if state.get("playwright") is not None:
            try:
                await state["playwright"].stop()
            except Exception:
                pass
        if owned:
            stopped_process = await _stop_owned_process(state)
    finally:
        _reset_runtime_state(state)

    return _ok(
        message="Browser stopped",
        owned_browser_process=owned,
        browser_pid=pid,
        browser_process_stopped=stopped_process,
    )


async def _browser_management_connect_cdp(
    state: dict[str, Any],
    cdp_endpoint: str,
) -> ToolResponse:
    if not cdp_endpoint:
        return _error("cdp_endpoint required for connect_cdp")
    if _is_browser_running(state):
        return _error("Browser already running. Stop it before connect_cdp.")
    try:
        async_playwright = _ensure_playwright_async()
        playwright = await async_playwright().start()
        browser = await playwright.chromium.connect_over_cdp(cdp_endpoint)
        contexts = browser.contexts
        context = contexts[0] if contexts else await browser.new_context(accept_downloads=True)
        state.update(
            {
                "playwright": playwright,
                "browser": browser,
                "context": context,
                "launch_mode": "external_cdp",
                "connected_via_cdp": True,
                "cdp_endpoint": cdp_endpoint,
                "owned_browser_process": False,
                "browser_process": None,
                "browser_pid": None,
            },
        )
        _attach_context_listeners(state, context)
        for page in context.pages:
            _register_page(state, page, "default" if not state["pages"] else _next_page_id(state))
        if not state["pages"]:
            page = await context.new_page()
            _register_page(state, page, "default")
        _touch(state)
        _start_idle_watchdog(state)
        return _ok(
            message=f"Connected to Chrome via CDP at {cdp_endpoint}",
            cdp_endpoint=cdp_endpoint,
            pages=list(state["pages"].keys()),
        )
    except Exception as exc:
        await _browser_management_stop(state)
        return _error(f"CDP connect failed: {exc}")


def _fetch_cdp_targets_for_port(port: int) -> Optional[dict[str, Any]]:
    if port <= 0:
        return None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.03)
        if sock.connect_ex(("127.0.0.1", port)) != 0:
            return None
    try:
        with urllib_request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=0.2) as response:  # noqa: S310
            version = json.loads(response.read().decode("utf-8"))
        with urllib_request.urlopen(f"http://127.0.0.1:{port}/json", timeout=0.2) as response:  # noqa: S310
            targets = json.loads(response.read().decode("utf-8"))
        return {
            "endpoint": f"http://127.0.0.1:{port}",
            "browser": version.get("Browser"),
            "webSocketDebuggerUrl": version.get("webSocketDebuggerUrl"),
            "targets": targets if isinstance(targets, list) else [],
        }
    except Exception:
        return None


async def _browser_management_list_cdp_targets() -> ToolResponse:
    found: dict[str, Any] = {}
    ports = list(range(0, _CDP_SCAN_MAX_PORT + 1))
    for start in range(0, len(ports), _CDP_SCAN_CHUNK_SIZE):
        chunk = ports[start : start + _CDP_SCAN_CHUNK_SIZE]
        results = await asyncio.gather(
            *[asyncio.to_thread(_fetch_cdp_targets_for_port, port) for port in chunk],
        )
        for port, result in zip(chunk, results):
            if result is not None:
                found[str(port)] = result
    return _ok(
        message=f"Found {len(found)} CDP endpoint(s)",
        scanned_range=f"127.0.0.1:0-{_CDP_SCAN_MAX_PORT}",
        found=found,
    )


_BROWSER_DISK_CACHE_DIRS = [
    Path("Default") / "Cache",
    Path("Default") / "Code Cache",
    Path("Default") / "GPUCache",
    Path("Default") / "DawnWebGPUCache",
    Path("Default") / "DawnGraphiteCache",
    Path("GrShaderCache"),
    Path("ShaderCache"),
    Path("GraphiteDawnCache"),
]


async def _browser_management_clear_browser_cache(state: dict[str, Any]) -> ToolResponse:
    if _is_browser_running(state):
        context = state.get("context")
        page = next(iter(state["pages"].values()), None)
        if context is None or page is None:
            return _error("No open page available for cache clear")
        try:
            cdp = await context.new_cdp_session(page)
            await cdp.send("Network.clearBrowserCache")
            return _ok(message="HTTP cache cleared")
        except Exception as exc:
            return _error(f"CDP cache clear failed: {exc}")

    user_data_dir = state.get("user_data_dir") or ""
    if not user_data_dir:
        return _error("No browser user_data_dir configured")
    removed: list[str] = []
    errors: list[str] = []
    for rel_path in _BROWSER_DISK_CACHE_DIRS:
        path = Path(user_data_dir) / rel_path
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            removed.append(str(rel_path))
        except Exception as exc:
            errors.append(f"{rel_path}: {exc}")
    if errors:
        return _error("Browser cache partially cleared", removed=removed, errors=errors)
    return _ok(message="Browser cache cleared", removed=removed)


def _locator_from_ref(state: dict[str, Any], page: Any, page_id: str, ref: str) -> Any:
    ref = str(ref or "").strip()
    if not ref:
        raise ValueError("ref required")
    refs = state["refs"].get(page_id) or {}
    info = refs.get(ref)
    if info is None:
        raise ValueError(f"Unknown ref: {ref}")
    role = info.get("role", "generic")
    name = info.get("name")
    nth = info.get("nth")
    locator = page.get_by_role(role, name=name or None)
    if nth is not None:
        locator = locator.nth(nth)
    return locator


async def _basic_open(
    state: dict[str, Any],
    page_id: str,
    url: str,
    wait_time: float,
) -> ToolResponse:
    url = url.strip()
    if not url:
        return _error("url required for open")
    if not await _ensure_browser(state):
        return _error(state.get("last_browser_error") or "Browser not started")
    try:
        page = await _get_or_create_page(state, page_id)
        await page.goto(url)
        _touch(state)
        await _sleep_after(wait_time)
        return _ok(message="Page opened", page_id=page_id, url=page.url)
    except Exception as exc:
        return _error(f"Open failed: {exc}")


async def _basic_snapshot(state: dict[str, Any], page_id: str) -> ToolResponse:
    try:
        page = await _activate_page(state, page_id)
        raw = await page.locator(":root").aria_snapshot()
        snapshot, refs = build_role_snapshot_from_aria(
            str(raw or ""),
            interactive=False,
            compact=True,
        )
        snapshot, truncated = _truncate_text(snapshot, _SNAPSHOT_MAX_CHARS)
        state["refs"][page_id] = refs
        _touch(state)
        return _ok(
            page_id=page_id,
            url=page.url,
            snapshot=snapshot,
            refs=list(refs.keys()),
            ref_count=len(refs),
            truncated=truncated,
        )
    except Exception as exc:
        return _error(f"Snapshot failed: {exc}")


async def _basic_navigate_back(
    state: dict[str, Any],
    page_id: str,
    wait_time: float,
) -> ToolResponse:
    try:
        page = await _activate_page(state, page_id)
        await page.go_back()
        _touch(state)
        await _sleep_after(wait_time)
        return _ok(message="Navigated back", page_id=page_id, url=page.url)
    except Exception as exc:
        return _error(f"Navigate back failed: {exc}")


async def _basic_click_element(
    state: dict[str, Any],
    page_id: str,
    ref: str,
    wait_time: float,
) -> ToolResponse:
    try:
        page = await _activate_page(state, page_id)
        locator = _locator_from_ref(state, page, page_id, ref)
        await locator.click()
        _touch(state)
        await _sleep_after(wait_time)
        return _ok(message="Element clicked", page_id=page_id, ref=ref)
    except Exception as exc:
        return _error(f"Click element failed: {exc}")


async def _basic_type(
    state: dict[str, Any],
    page_id: str,
    ref: str,
    text: str,
) -> ToolResponse:
    try:
        page = await _activate_page(state, page_id)
        locator = _locator_from_ref(state, page, page_id, ref)
        await locator.fill(text)
        _touch(state)
        return _ok(message="Text entered", page_id=page_id, ref=ref)
    except Exception as exc:
        return _error(f"Type failed: {exc}")


async def _basic_press_key(
    state: dict[str, Any],
    page_id: str,
    key: str,
    wait_time: float,
) -> ToolResponse:
    key = key.strip()
    if not key:
        return _error("key required for press_key")
    try:
        page = await _activate_page(state, page_id)
        await page.keyboard.press(key)
        _touch(state)
        await _sleep_after(wait_time)
        return _ok(message="Key pressed", page_id=page_id, key=key)
    except Exception as exc:
        return _error(f"Press key failed: {exc}")


async def _basic_tabs(state: dict[str, Any]) -> ToolResponse:
    tabs: list[dict[str, Any]] = []
    for page_id, page in state["pages"].items():
        tabs.append(
            {
                "page_id": page_id,
                "url": getattr(page, "url", ""),
                "current": page_id == state.get("current_page_id"),
            },
        )
    return _ok(tabs=tabs, count=len(tabs), current_page_id=state.get("current_page_id"))


async def _basic_close(state: dict[str, Any], page_id: str) -> ToolResponse:
    page = state["pages"].get(page_id)
    if page is None:
        return _error(f"Page '{page_id}' not found")
    try:
        await page.close()
    except Exception:
        pass
    state["pages"].pop(page_id, None)
    state["refs"].pop(page_id, None)
    state["pending_downloads"].pop(page_id, None)
    state["pending_file_choosers"].pop(page_id, None)
    if state.get("current_page_id") == page_id:
        state["current_page_id"] = next(iter(state["pages"].keys()), None)
    _touch(state)
    return _ok(message="Page closed", page_id=page_id, current_page_id=state.get("current_page_id"))


async def _basic_wait_for(
    state: dict[str, Any],
    page_id: str,
    wait_time: float,
) -> ToolResponse:
    try:
        await _activate_page(state, page_id)
        await _sleep_after(wait_time)
        _touch(state)
        return _ok(message="Wait completed", page_id=page_id)
    except Exception as exc:
        return _error(f"Wait failed: {exc}")


async def _advanced_screenshot(
    state: dict[str, Any],
    page_id: str,
    file_path: str,
) -> ToolResponse:
    file_path = file_path.strip()
    if not file_path:
        file_path = f"page-{int(time.time())}.png"
    resolved = _resolve_output_path(file_path)
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    try:
        page = await _activate_page(state, page_id)
        await page.screenshot(path=resolved, full_page=False)
        _touch(state)
        return _ok(message="Screenshot saved", page_id=page_id, file_path=resolved, url=page.url)
    except Exception as exc:
        return _error(f"Screenshot failed: {exc}")


async def _advanced_click_xy(
    state: dict[str, Any],
    page_id: str,
    x: float,
    y: float,
) -> ToolResponse:
    try:
        page = await _activate_page(state, page_id)
        await page.mouse.click(x, y)
        _touch(state)
        return _ok(message="Coordinates clicked", page_id=page_id, x=x, y=y)
    except Exception as exc:
        return _error(f"Click xy failed: {exc}")


async def _advanced_run_code(
    state: dict[str, Any],
    page_id: str,
    code: str,
) -> ToolResponse:
    code = code.strip()
    if not code:
        return _error("code required for run_code")
    try:
        page = await _activate_page(state, page_id)
        if code.startswith(("(", "function", "async ")):
            result = await page.evaluate(code)
        else:
            try:
                result = await page.evaluate(f"() => {{ return ({code}); }}")
            except Exception:
                result = await page.evaluate(f"async () => {{ {code} }}")
        _touch(state)
        try:
            result_json = json.dumps(result, ensure_ascii=False)
            if len(result_json) > _RESULT_MAX_CHARS:
                return _ok(
                    message="Code executed",
                    page_id=page_id,
                    result=str(result)[:_RESULT_MAX_CHARS],
                    truncated=True,
                )
            return _ok(message="Code executed", page_id=page_id, result=result)
        except TypeError:
            return _ok(message="Code executed", page_id=page_id, result=str(result))
    except Exception as exc:
        return _error(f"Run code failed: {exc}")


async def _advanced_file_upload(
    state: dict[str, Any],
    page_id: str,
    file_path: str | list[str],
) -> ToolResponse:
    if not file_path:
        return _error("file_path required for file_upload")
    file_paths = file_path if isinstance(file_path, list) else [file_path]
    file_paths = [str(path) for path in file_paths]
    try:
        page = await _activate_page(state, page_id)
        chooser = None
        choosers = state["pending_file_choosers"].setdefault(page_id, [])
        if choosers:
            chooser = choosers.pop(0)
        if chooser is not None:
            await chooser.set_files(file_paths)
        else:
            await page.locator("input[type=file]").first.set_input_files(file_paths)
        _touch(state)
        return _ok(message="File uploaded", page_id=page_id, file_path=file_paths)
    except Exception as exc:
        return _error(f"File upload failed: {exc}")


async def _advanced_file_download(
    state: dict[str, Any],
    page_id: str,
    file_path: str,
) -> ToolResponse:
    file_path = file_path.strip()
    if not file_path:
        return _error("file_path required for file_download")
    resolved = _resolve_output_path(file_path)
    Path(resolved).parent.mkdir(parents=True, exist_ok=True)
    try:
        page = await _activate_page(state, page_id)
        downloads = state["pending_downloads"].setdefault(page_id, [])
        download = downloads.pop(0) if downloads else None
        if download is None:
            download = await page.wait_for_event("download", timeout=30000)
        await download.save_as(resolved)
        _touch(state)
        return _ok(message="Download saved", page_id=page_id, file_path=resolved)
    except Exception as exc:
        return _error(f"File download failed: {exc}")


async def _dispatch_browser_management(state: dict[str, Any], operation: str, params: dict[str, Any]) -> ToolResponse:
    if operation == "start":
        cdp_endpoint, cdp_port = _normalize_cdp_endpoint(params.get("cdp_endpoint"))
        return await _browser_management_start(
            state,
            headed=_as_bool(params.get("headed"), False),
            disable_cdp=_as_bool(params.get("disable_cdp"), False),
            executable_path=str(params.get("executable_path") or ""),
            browser_launch_args=str(params.get("browser_launch_args") or ""),
            cdp_endpoint=cdp_endpoint,
            cdp_port=cdp_port,
        )
    if operation == "stop":
        return await _browser_management_stop(state)
    if operation == "connect_cdp":
        cdp_endpoint, _ = _normalize_cdp_endpoint(params.get("cdp_endpoint"))
        return await _browser_management_connect_cdp(state, cdp_endpoint)
    if operation == "list_cdp_targets":
        return await _browser_management_list_cdp_targets()
    if operation == "clear_browser_cache":
        return await _browser_management_clear_browser_cache(state)
    return _error(
        f"Unknown browser_management operation: {operation}",
        allowed=["start", "stop", "connect_cdp", "list_cdp_targets", "clear_browser_cache"],
    )


async def _dispatch_basic_page_operation(state: dict[str, Any], operation: str, params: dict[str, Any]) -> ToolResponse:
    page_id = _resolve_page_id(state, params.get("page_id", "default"))
    wait_time = _as_float(params.get("wait_time"), 0.0)
    if operation == "open":
        return await _basic_open(
            state,
            page_id=page_id,
            url=str(params.get("url") or ""),
            wait_time=wait_time,
        )
    if operation == "tabs":
        return await _basic_tabs(state)
    if not await _ensure_browser(state):
        return _error(state.get("last_browser_error") or "Browser not started")
    if operation == "snapshot":
        return await _basic_snapshot(state, page_id)
    if operation == "navigate_back":
        return await _basic_navigate_back(state, page_id, wait_time)
    if operation == "click_element":
        return await _basic_click_element(
            state,
            page_id=page_id,
            ref=str(params.get("ref") or ""),
            wait_time=wait_time,
        )
    if operation == "type":
        return await _basic_type(
            state,
            page_id=page_id,
            ref=str(params.get("ref") or ""),
            text=str(params.get("text") or ""),
        )
    if operation == "press_key":
        return await _basic_press_key(
            state,
            page_id=page_id,
            key=str(params.get("key") or ""),
            wait_time=wait_time,
        )
    if operation == "close":
        return await _basic_close(state, page_id)
    if operation == "wait_for":
        return await _basic_wait_for(state, page_id, wait_time)
    return _error(
        f"Unknown basic_page_operation operation: {operation}",
        allowed=["open", "snapshot", "navigate_back", "click_element", "type", "press_key", "tabs", "close", "wait_for"],
    )


async def _dispatch_advanced_page_operation(state: dict[str, Any], operation: str, params: dict[str, Any]) -> ToolResponse:
    page_id = _resolve_page_id(state, params.get("page_id", "default"))
    if not await _ensure_browser(state):
        return _error(state.get("last_browser_error") or "Browser not started")
    if operation == "screenshot":
        return await _advanced_screenshot(
            state,
            page_id=page_id,
            file_path=str(params.get("file_path") or ""),
        )
    if operation == "click_xy":
        if "x" not in params or "y" not in params:
            return _error("x and y required for click_xy")
        return await _advanced_click_xy(
            state,
            page_id=page_id,
            x=_as_float(params.get("x")),
            y=_as_float(params.get("y")),
        )
    if operation == "run_code":
        return await _advanced_run_code(
            state,
            page_id=page_id,
            code=str(params.get("code") or ""),
        )
    if operation == "file_upload":
        return await _advanced_file_upload(
            state,
            page_id=page_id,
            file_path=params.get("file_path") or "",
        )
    if operation == "file_download":
        return await _advanced_file_download(
            state,
            page_id=page_id,
            file_path=str(params.get("file_path") or ""),
        )
    return _error(
        f"Unknown advanced_page_operation operation: {operation}",
        allowed=["screenshot", "click_xy", "run_code", "file_upload", "file_download"],
    )


async def browser_use(
    action_type: str,
    operation: str,
    extra_params: Optional[dict[str, Any]] = None,
) -> ToolResponse:
    """Control browser using the final browser_use interface."""
    state = _get_state()
    action_type = str(action_type or "").strip().lower()
    operation = str(operation or "").strip().lower()
    params = _coerce_extra_params(extra_params)

    if not action_type:
        return _error("action_type required")
    if not operation:
        return _error("operation required")

    try:
        if action_type == "browser_management":
            return await _dispatch_browser_management(state, operation, params)
        if action_type == "basic_page_operation":
            return await _dispatch_basic_page_operation(state, operation, params)
        if action_type == "advanced_page_operation":
            return await _dispatch_advanced_page_operation(state, operation, params)
        return _error(
            f"Unknown action_type: {action_type}",
            allowed=["browser_management", "basic_page_operation", "advanced_page_operation"],
        )
    except Exception as exc:
        logger.error("Browser tool error: %s", exc, exc_info=True)
        return _error(str(exc))


def _atexit_cleanup() -> None:
    if not _workspace_states:
        return
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running() or loop.is_closed():
            return
        for state in list(_workspace_states.values()):
            if _is_browser_running(state) or state.get("browser_process"):
                loop.run_until_complete(_browser_management_stop(state))
    except Exception:
        pass


atexit.register(_atexit_cleanup)
