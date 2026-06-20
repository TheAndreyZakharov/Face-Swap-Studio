from __future__ import annotations

from src.face_swap_studio.ui.app import create_app
from src.face_swap_studio.utils.logging import configure_logging
from src.face_swap_studio.utils.paths import load_settings


def main() -> None:
    configure_logging()
    settings = load_settings()

    application = settings["application"]
    demo = create_app()

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=str(application.get("host", "127.0.0.1")),
        server_port=int(application.get("port", 7860)),
        share=bool(application.get("share", False)),
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
