from __future__ import annotations

from pathlib import Path

from src.face_swap_studio.adapters.base import (
    AdapterRequest,
    BaseAdapter,
)
from src.face_swap_studio.adapters.process_runner import (
    require_output_file,
    run_external_process,
)
from src.face_swap_studio.utils.logging import get_logger
from src.face_swap_studio.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)


class GhostAdapter(BaseAdapter):
    def __init__(
        self,
        model_id: str,
        blocks: int,
    ) -> None:
        if blocks not in {1, 2, 3}:
            raise ValueError("GHOST поддерживает только 1, 2 или 3 блока.")

        self.model_id = model_id
        self.blocks = blocks

    def _checkpoint_path(self) -> Path:
        suffix = "block" if self.blocks == 1 else "blocks"

        return PROJECT_ROOT / "vendor" / "ghost" / "weights" / f"G_unet_{self.blocks}{suffix}.pth"

    def is_available(self) -> bool:
        required_files = (
            PROJECT_ROOT / ".environments" / "ghost" / "bin" / "python",
            PROJECT_ROOT / "scripts" / "external" / "ghost" / "infer.py",
            self._checkpoint_path(),
            PROJECT_ROOT / "vendor" / "ghost" / "arcface_model" / "backbone.pth",
        )

        return all(path.is_file() and path.stat().st_size > 0 for path in required_files)

    def process(
        self,
        request: AdapterRequest,
    ) -> Path:
        python_path = PROJECT_ROOT / ".environments" / "ghost" / "bin" / "python"
        wrapper_path = PROJECT_ROOT / "scripts" / "external" / "ghost" / "infer.py"

        request.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            python_path,
            wrapper_path,
            "--source",
            request.source_path,
            "--target",
            request.target_path,
            "--output",
            request.output_path,
            "--blocks",
            str(self.blocks),
        ]

        logger.info(
            "Запуск GHOST %s block(s): %s",
            self.blocks,
            [str(item) for item in command],
        )

        result = run_external_process(
            command,
            cwd=PROJECT_ROOT,
        )

        return require_output_file(
            request.output_path,
            model_name=f"GHOST U-Net {self.blocks} block(s)",
            process_result=result,
        )
