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


class Ghost2Adapter(BaseAdapter):
    model_id = "ghost2_head"

    def is_available(self) -> bool:
        required_files = (
            PROJECT_ROOT / ".environments" / "ghost2" / "bin" / "python",
            PROJECT_ROOT / "scripts" / "external" / "ghost2" / "infer.py",
            PROJECT_ROOT
            / "vendor"
            / "ghost2"
            / "aligner_checkpoints"
            / "aligner_1020_gaze_final.ckpt",
            PROJECT_ROOT / "vendor" / "ghost2" / "blender_checkpoints" / "blender_lama.ckpt",
        )

        return all(path.is_file() and path.stat().st_size > 0 for path in required_files)

    def process(
        self,
        request: AdapterRequest,
    ) -> Path:
        python_path = PROJECT_ROOT / ".environments" / "ghost2" / "bin" / "python"
        wrapper_path = PROJECT_ROOT / "scripts" / "external" / "ghost2" / "infer.py"

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
        ]

        logger.info(
            "Запуск GHOST 2.0: %s",
            [str(item) for item in command],
        )

        result = run_external_process(
            command,
            cwd=PROJECT_ROOT,
        )

        return require_output_file(
            request.output_path,
            model_name="GHOST 2.0 Head Swap",
            process_result=result,
        )
