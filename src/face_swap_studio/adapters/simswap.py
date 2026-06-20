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


class SimSwapAdapter(BaseAdapter):
    model_id = "simswap_512"

    def is_available(self) -> bool:
        required_paths = (
            PROJECT_ROOT / ".environments" / "simswap" / "bin" / "python",
            PROJECT_ROOT / "scripts" / "external" / "simswap" / "infer.py",
            PROJECT_ROOT / "vendor" / "simswap" / "checkpoints" / "512" / "550000_net_G.pth",
            PROJECT_ROOT / "vendor" / "simswap" / "arcface_model" / "arcface_checkpoint.tar",
        )

        return all(path.is_file() and path.stat().st_size > 0 for path in required_paths)

    def process(
        self,
        request: AdapterRequest,
    ) -> Path:
        python_path = PROJECT_ROOT / ".environments" / "simswap" / "bin" / "python"
        wrapper_path = PROJECT_ROOT / "scripts" / "external" / "simswap" / "infer.py"

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
            "--use-mask",
        ]

        logger.info(
            "Запуск SimSwap 512: %s",
            [str(item) for item in command],
        )

        result = run_external_process(
            command,
            cwd=PROJECT_ROOT,
        )

        return require_output_file(
            request.output_path,
            model_name="SimSwap 512",
            process_result=result,
        )
