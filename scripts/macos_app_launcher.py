from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import webview

APP_NAME = "Face Swap Studio"
HOST = "127.0.0.1"
PORT = 7860
STARTUP_TIMEOUT_SECONDS = 90

_BACKEND_PROCESS: subprocess.Popen[bytes] | None = None
_CLEANUP_DONE = False


def project_root() -> Path:
    explicit_root = os.environ.get(
        "FACE_SWAP_STUDIO_PROJECT_ROOT"
    )

    if explicit_root:
        return Path(
            explicit_root
        ).expanduser().resolve()

    return Path(
        __file__
    ).resolve().parents[1]


def log_path() -> Path:
    directory = (
        Path.home()
        / "Library"
        / "Logs"
        / APP_NAME
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / "app.log"
    )


def app_url() -> str:
    return (
        f"http://{HOST}:{PORT}"
    )


def health_url() -> str:
    return (
        f"{app_url()}/api/health"
    )


def process_ids_on_port() -> list[int]:
    command = [
        "lsof",
        "-ti",
        f"tcp:{PORT}",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    process_ids: list[int] = []

    for line in result.stdout.splitlines():
        value = line.strip()

        if not value:
            continue

        try:
            process_id = int(
                value
            )
        except ValueError:
            continue

        if process_id == os.getpid():
            continue

        process_ids.append(
            process_id
        )

    return process_ids


def kill_process_ids(
    process_ids: list[int],
    signal_number: signal.Signals,
) -> None:
    for process_id in process_ids:
        try:
            os.kill(
                process_id,
                signal_number,
            )
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def release_port() -> None:
    process_ids = process_ids_on_port()

    if not process_ids:
        return

    kill_process_ids(
        process_ids,
        signal.SIGTERM,
    )

    deadline = (
        time.monotonic()
        + 5.0
    )

    while time.monotonic() < deadline:
        if not process_ids_on_port():
            return

        time.sleep(
            0.2
        )

    remaining_process_ids = process_ids_on_port()

    if remaining_process_ids:
        kill_process_ids(
            remaining_process_ids,
            signal.SIGKILL,
        )

    time.sleep(
        0.5
    )


def terminate_backend_process(
    process: subprocess.Popen[bytes] | None,
) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGTERM,
        )
    except ProcessLookupError:
        return

    try:
        process.wait(
            timeout=8,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(
                process.pid,
                signal.SIGKILL,
            )
        except ProcessLookupError:
            pass

        try:
            process.wait(
                timeout=3,
            )
        except subprocess.TimeoutExpired:
            pass


def cleanup() -> None:
    global _CLEANUP_DONE

    if _CLEANUP_DONE:
        return

    _CLEANUP_DONE = True

    terminate_backend_process(
        _BACKEND_PROCESS
    )

    release_port()


def handle_shutdown_signal(
    signum: int,
    frame: object,
) -> None:
    del signum
    del frame

    cleanup()

    raise SystemExit(
        0
    )


def wait_for_server() -> None:
    deadline = (
        time.monotonic()
        + STARTUP_TIMEOUT_SECONDS
    )

    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                health_url(),
                timeout=1.5,
            ) as response:
                if response.status == 200:
                    return
        except Exception as error:
            last_error = error

        time.sleep(
            0.5
        )

    raise RuntimeError(
        "Face Swap Studio server did not start. "
        f"Last error: {last_error}"
    )


def safe_download_filename(
    filename: str | None,
    fallback: str,
) -> str:
    value = (
        filename
        or fallback
    ).strip()

    value = value.replace(
        "/",
        "_",
    ).replace(
        "\\",
        "_",
    ).replace(
        ":",
        "_",
    )

    safe_value = "".join(
        character
        if (
            character.isalnum()
            or character in {
                ".",
                "-",
                "_",
                " ",
            }
        )
        else "_"
        for character in value
    ).strip()

    if not safe_value:
        safe_value = fallback

    return safe_value


def unique_download_path(
    filename: str,
) -> Path:
    downloads_directory = (
        Path.home()
        / "Downloads"
    )

    downloads_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        downloads_directory
        / filename
    )

    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix

    for index in range(
        2,
        1000,
    ):
        candidate = (
            downloads_directory
            / f"{stem} {index}{suffix}"
        )

        if not candidate.exists():
            return candidate

    raise RuntimeError(
        "Could not create a unique download filename."
    )


class NativeApi:
    def download_file(
        self,
        url: str,
        suggested_filename: str | None = None,
    ) -> dict[str, str]:
        absolute_url = urllib.parse.urljoin(
            app_url(),
            url,
        )

        filename = safe_download_filename(
            suggested_filename,
            "face-swap-download",
        )

        destination = unique_download_path(
            filename
        )

        request = urllib.request.Request(
            absolute_url,
            headers={
                "User-Agent": APP_NAME,
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            data = response.read()

        if not data:
            raise RuntimeError(
                "Downloaded file is empty."
            )

        destination.write_bytes(
            data
        )

        return {
            "path": str(
                destination
            ),
            "filename": destination.name,
        }

    def shutdown(self) -> None:
        cleanup()


def start_backend(
    root: Path,
) -> subprocess.Popen[bytes]:
    python_path = (
        root
        / ".venv"
        / "bin"
        / "python"
    )

    app_path = (
        root
        / "app.py"
    )

    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python environment not found: {python_path}"
        )

    if not app_path.is_file():
        raise FileNotFoundError(
            f"Application entrypoint not found: {app_path}"
        )

    environment = os.environ.copy()

    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(root),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "FACE_SWAP_OPEN_BROWSER": "0",
            "FACE_SWAP_HOST": HOST,
            "FACE_SWAP_PORT": str(
                PORT
            ),
        }
    )

    output_log = log_path().open(
        "ab"
    )

    return subprocess.Popen(
        [
            str(python_path),
            str(app_path),
        ],
        cwd=str(root),
        env=environment,
        stdout=output_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def main() -> int:
    global _BACKEND_PROCESS

    root = project_root()

    atexit.register(
        cleanup
    )

    signal.signal(
        signal.SIGTERM,
        handle_shutdown_signal,
    )

    signal.signal(
        signal.SIGINT,
        handle_shutdown_signal,
    )

    try:
        release_port()

        _BACKEND_PROCESS = start_backend(
            root
        )

        wait_for_server()

        window = webview.create_window(
            APP_NAME,
            app_url(),
            width=1440,
            height=980,
            min_size=(
                1100,
                760,
            ),
            confirm_close=False,
            js_api=NativeApi(),
        )

        try:
            window.events.closed += cleanup
        except Exception:
            pass

        webview.start()

        cleanup()

        return 0

    except Exception as error:
        cleanup()

        message = (
            f"{APP_NAME} failed to start.\n\n"
            f"{error}\n\n"
            f"Log file:\n{log_path()}"
        )

        print(
            message,
            file=sys.stderr,
        )

        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    (
                        'display alert "Face Swap Studio" '
                        f'message "{str(error).replace(chr(34), chr(39))}"'
                    ),
                ],
                check=False,
            )
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )