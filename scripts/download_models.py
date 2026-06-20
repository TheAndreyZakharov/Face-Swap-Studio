from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]

CONNECT_TIMEOUT = 30
READ_TIMEOUT = 600
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Model:
    name: str
    category: str
    filename: str
    url: str
    sha256: str | None = None

    @property
    def destination(self) -> Path:
        return ROOT / "models" / self.category / self.filename


MODELS: dict[str, Model] = {
    "gfpgan-v1.4": Model(
        name="GFPGAN v1.4",
        category="enhancers",
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    ),
    "realesrgan-x4plus": Model(
        name="Real-ESRGAN x4plus",
        category="upscalers",
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
}


def create_session() -> requests.Session:
    retry_policy = Retry(
        total=5,
        connect=5,
        read=5,
        redirect=5,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Face-Swap-Studio/0.1",
            "Accept": "application/octet-stream,*/*",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_model(model: Model) -> None:
    if not model.destination.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model.destination}")

    if model.destination.stat().st_size == 0:
        model.destination.unlink(missing_ok=True)
        raise RuntimeError(f"Скачан пустой файл: {model.destination}")

    if model.sha256 is not None:
        actual_hash = calculate_sha256(model.destination)

        if actual_hash.lower() != model.sha256.lower():
            model.destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Контрольная сумма не совпала для {model.name}\n"
                f"Ожидалась: {model.sha256}\n"
                f"Получена: {actual_hash}"
            )


def download_once(
    session: requests.Session,
    model: Model,
    temporary: Path,
) -> None:
    downloaded_size = temporary.stat().st_size if temporary.exists() else 0

    headers: dict[str, str] = {}

    if downloaded_size > 0:
        headers["Range"] = f"bytes={downloaded_size}-"

    with session.get(
        model.url,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    ) as response:
        if downloaded_size > 0 and response.status_code == 206:
            write_mode = "ab"
            initial_size = downloaded_size
        else:
            write_mode = "wb"
            initial_size = 0
            downloaded_size = 0

        response.raise_for_status()

        remaining_size = int(response.headers.get("content-length", 0))
        total_size = initial_size + remaining_size if remaining_size > 0 else None

        with (
            temporary.open(write_mode) as file,
            tqdm(
                total=total_size,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=model.name,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                file.write(chunk)
                progress.update(len(chunk))


def download(model: Model, force: bool = False) -> None:
    destination = model.destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force:
        validate_model(model)
        print(f"Уже скачано: {destination}")
        return

    if force:
        destination.unlink(missing_ok=True)

    temporary = destination.with_suffix(destination.suffix + ".part")
    session = create_session()

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(f"Загрузка {model.name}, попытка {attempt}/{MAX_ATTEMPTS}")
                download_once(session, model, temporary)
                temporary.replace(destination)
                validate_model(model)
                print(f"Скачано: {destination}")
                return

            except (
                requests.RequestException,
                TimeoutError,
                ConnectionError,
            ) as error:
                print(
                    f"Ошибка загрузки: {error}",
                    file=sys.stderr,
                )

                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Не удалось скачать {model.name} после "
                        f"{MAX_ATTEMPTS} попыток. Частичный файл сохранён: "
                        f"{temporary}"
                    ) from error

                delay = min(5 * attempt, 30)
                print(f"Повтор через {delay} секунд. Загрузка продолжится с сохранённого места.")
                time.sleep(delay)
    finally:
        session.close()


def list_models() -> None:
    for key, model in MODELS.items():
        status = "скачано" if model.destination.exists() else "не скачано"
        print(f"{key}: {model.name}")
        print(f"  Статус: {status}")
        print(f"  Путь: {model.destination}")
        print(f"  URL: {model.url}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка моделей Face Swap Studio")

    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Названия моделей для загрузки",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Скачать все модели",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать уже скачанные модели",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список моделей",
    )

    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    selected_models = list(MODELS) if args.all else args.models

    if not selected_models:
        parser.print_help()
        return 1

    unknown_models = [model_name for model_name in selected_models if model_name not in MODELS]

    if unknown_models:
        print(
            "Неизвестные модели: " + ", ".join(unknown_models),
            file=sys.stderr,
        )
        print(
            "Доступные модели: " + ", ".join(sorted(MODELS)),
            file=sys.stderr,
        )
        return 2

    failed_models: list[str] = []

    for model_name in selected_models:
        try:
            download(MODELS[model_name], force=args.force)
        except Exception as error:
            failed_models.append(model_name)
            print(f"Ошибка {model_name}: {error}", file=sys.stderr)

    if failed_models:
        print(
            "Не удалось скачать: " + ", ".join(failed_models),
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nЗагрузка остановлена пользователем.", file=sys.stderr)
        raise SystemExit(130) from None
