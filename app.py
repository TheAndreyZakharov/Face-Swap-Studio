from __future__ import annotations

import os
import threading
import time
import webbrowser

import uvicorn

from src.face_swap_studio.ui.server import (
    create_application,
)
from src.face_swap_studio.utils.logging import (
    configure_logging,
)


def open_browser(
    url: str,
) -> None:
    time.sleep(
        1.0
    )

    webbrowser.open(
        url,
        new=1,
        autoraise=True,
    )


def main() -> None:
    configure_logging()

    host = os.getenv(
        "FACE_SWAP_HOST",
        "127.0.0.1",
    )

    port = int(
        os.getenv(
            "FACE_SWAP_PORT",
            "7860",
        )
    )

    browser_host = (
        "127.0.0.1"
        if host in {
            "0.0.0.0",
            "::",
        }
        else host
    )

    url = (
        f"http://{browser_host}:{port}"
    )

    application = create_application()

    if os.getenv(
        "FACE_SWAP_OPEN_BROWSER",
        "1",
    ).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        browser_thread = threading.Thread(
            target=open_browser,
            args=(
                url,
            ),
            daemon=True,
        )

        browser_thread.start()

    uvicorn.run(
        application,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()