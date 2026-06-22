<div align="center">

<img src="assets/forreadme/logo.png" alt="Логотип Face Swap Studio" width="300"/>

# Face Swap Studio

[![Русский](https://img.shields.io/badge/README_Language-Русский-brightgreen)](https://github.com/TheAndreyZakharov/Face-Swap-Studio/blob/main/README_RU.md)
[![English](https://img.shields.io/badge/README_Language-English-blue)](https://github.com/TheAndreyZakharov/Face-Swap-Studio/blob/main/README.md)

</div>

Face Swap Studio — это macOS-приложение для тестирования и сравнения нескольких моделей замены лиц в одном локальном интерфейсе.

Проект объединяет несколько внешних нейросетевых моделей, пайплайн обнаружения лиц, дополнительные инструменты постобработки, выбор нескольких target-изображений, настройку соответствий лиц для каждого изображения, предпросмотр результата и локальное сохранение сгенерированных файлов.

Нейросетевые модели, используемые в этом проекте, не были созданы или обучены в рамках этого проекта. Face Swap Studio предоставляет прикладной слой, интеграцию под macOS, адаптеры моделей, настройку выполнения на Apple Silicon, пользовательский интерфейс, управление сессиями и локальный workflow генерации вокруг этих внешних моделей.

Приложение работает локально на macOS. Загруженные изображения, найденные лица, промежуточные файлы, сгенерированные предпросмотры и временные архивы хранятся во временной папке сессии и удаляются при завершении работы приложения. При запуске приложение также выполняет дополнительную проверку и удаляет временные данные сессий, которые могли остаться от предыдущих запусков.

Скачанные пользователем результаты сохраняются отдельно в папку `Загрузки` и автоматически не удаляются.

## Назначение и ответственное использование

Face Swap Studio — это академический, учебный и технический экспериментальный проект.

Он предназначен для изучения локальных AI-пайплайнов обработки изображений, интеграции моделей, проектирования интерфейса, упаковки macOS-приложения и совместимости с Apple Silicon.

Используйте этот проект ответственно.

Не используйте приложение для создания обманного, вредоносного, незаконного, несогласованного, оскорбительного или неэтичного контента. Всегда уважайте приватность, согласие людей, авторские права, правила платформ, местные законы и достоинство других людей.

Все скриншоты в этом README используют демонстрационные изображения, на которых лица намеренно замазаны или скрыты, потому что люди на исходных изображениях не давали согласия на публичное отображение в этом репозитории.

## Интегрированные модели

Face Swap Studio может показывать несколько настроенных backend-моделей для замены лиц и улучшения изображения, если соответствующие файлы моделей присутствуют в проекте.

Модели замены лиц:

- InSwapper 128;
- HyperSwap 1A 256;
- HyperSwap 1B 256;
- UniFace 256;
- SimSwap 512 beta;
- Ghost U-Net 1 block;
- Ghost U-Net 2 blocks;
- Ghost U-Net 3 blocks;
- Ghost 2.0 head swap.

Инструменты постобработки:

- восстановление лиц GFPGAN;
- апскейлинг изображения Real-ESRGAN.

Проект не заявляет авторство этих нейросетевых моделей. Они интегрированы как внешние model backends. Их собственные лицензии, условия использования, model cards и ограничения должны соблюдаться отдельно.

## Как это работает

Face Swap Studio запускает локальный backend-сервер и открывает отдельное окно macOS-приложения.

Пользователь добавляет одно или несколько source identity изображений и одно или несколько target-изображений. Приложение обнаруживает лица, группирует найденные target-лица по изображениям, позволяет визуально задать замены, запускает выбранную модель face swap, при необходимости применяет постобработку и показывает сгенерированные результаты.

Схема обработки:

    Source identity изображения
                ↓
        Обнаружение source faces
                ↓
        Выбор target images
                ↓
        Обнаружение target faces
                ↓
    Визуальное соответствие source → target
                ↓
        Выбранная face-swap модель
                ↓
    Опциональное восстановление и апскейлинг
                ↓
    Предпросмотр результатов и экспорт в Загрузки

Приложение поддерживает генерацию как одного изображения, так и нескольких изображений за раз.

Можно выбрать сразу несколько target-изображений. У каждого выбранного target-изображения могут быть свои найденные лица и свои настройки replacement mappings.

## Установка из GitHub Release

Полный macOS application bundle может распространяться через GitHub Releases.

Так как упакованное приложение может быть очень большим, релизный архив может быть разбит на несколько частей.

Скачайте все части из последнего GitHub Release в одну папку.

Пример файлов релиза:

    Face Swap Studio.zip.part-aa
    Face Swap Studio.zip.part-ab
    Face Swap Studio.zip.part-ac
    ...
    Face Swap Studio.zip.sha256

После скачивания всех частей объедините их на Mac:

    cat "Face Swap Studio.zip.part-"* > "Face Swap Studio.zip"

Если в релизе есть checksum-файл, проверьте архив:

    shasum -a 256 -c "Face Swap Studio.zip.sha256"

Распакуйте архив:

    unzip "Face Swap Studio.zip"

При необходимости переместите приложение в `Applications`:

    mv "Face Swap Studio.app" /Applications/

При первом запуске macOS может показать предупреждение, потому что приложение не подписано и не notarized. В этом случае нажмите по приложению правой кнопкой мыши и выберите `Открыть`.

Если macOS quarantine мешает запуску, удалите quarantine-атрибут:

    xattr -cr "Face Swap Studio.app"

или, если приложение уже перемещено в Applications:

    xattr -cr "/Applications/Face Swap Studio.app"

## Требование Python для упакованного приложения

Упакованное приложение содержит проект и его виртуальное окружение, но виртуальное окружение может ссылаться на Python, установленный через Homebrew.

Среда разработки использует Python 3.11 через Homebrew:

    .venv/bin/python -> python3.11
    .venv/bin/python3.11 -> /opt/homebrew/opt/python@3.11/bin/python3.11

На другом Mac с Apple Silicon перед запуском приложения установите Python 3.11 через Homebrew:

    brew install python@3.11

Проверьте, что ожидаемый Python executable существует:

    ls -l /opt/homebrew/opt/python@3.11/bin/python3.11

Если второй Mac использует ту же Apple Silicon архитектуру и Python 3.11 установлен в этом расположении, bundled virtual environment должен работать так же, как на машине разработки.

## Первый запуск

После открытия Face Swap Studio появляется главное окно с пустым workspace.

<div align="center">

<img src="assets/forreadme/1.png" alt="Face Swap Studio в тёмной теме при первом запуске" width="600"/>

</div>

<div align="center">

<img src="assets/forreadme/2.png" alt="Face Swap Studio в светлой теме при первом запуске" width="600"/>

</div>

В приложении есть тёмная и светлая темы.

В верхней части находятся:

- статус подключения backend;
- переключатель темы;
- кнопка очистки сессии.

Если статус показывает `Connected`, значит backend работает корректно и приложение готово к использованию.

Переключатель темы меняет тёмный и светлый режим.

Кнопка `Clear session` удаляет текущие данные сессии из интерфейса и создаёт новую пустую сессию. Она очищает загруженные изображения, найденные лица, mappings, сгенерированные предпросмотры и временные result-файлы активной сессии.

## Шаг 1. Добавление source identities и target images

Первый блок содержит две зоны загрузки:

- `Source identities` — лица, которые будут использоваться как replacement identities;
- `Target images` — изображения, на которых лица будут заменяться.

Можно загрузить сразу несколько изображений. Дополнительные изображения также можно добавить позже.

<div align="center">

<img src="assets/forreadme/3.png" alt="Source identities и выбранные target images в Face Swap Studio" width="600"/>

</div>

Все загруженные source identity изображения используются для обнаружения лиц. Source images не нужно отдельно выбирать или подсвечивать.

Target images можно включать или исключать из генерации.

Target image, подсвеченное зелёным, выбрано для генерации. Target image без зелёной подсветки обрабатываться не будет.

Элементы управления target selection:

- `Select all` — включить все загруженные target images в генерацию;
- `Clear selection` — убрать все target images из выбора для генерации;
- `Add for generation` — включить отдельное target image;
- `Remove from generation` — исключить отдельное target image.

Если target image не выбран вручную, workflow в некоторых случаях может использовать active target image как fallback.

Каждую загруженную миниатюру можно открыть в крупном предпросмотре, нажав на неё.

Ненужные загруженные изображения можно удалить, наведя курсор на миниатюру и нажав кнопку `×` в правом верхнем углу. Это удобно, если изображение было добавлено случайно.

## Шаг 2. Обнаружение лиц

После добавления source и target изображений нужно запустить face detection.

В секции detection находится параметр `Detection confidence` и действие `Detect active target`.

Confidence value управляет строгостью обнаружения лиц. Более высокие значения могут уменьшить количество ложных срабатываний, а более низкие значения могут помочь найти больше лиц на сложных изображениях.

Во время обработки Face Swap Studio показывает загрузочный экран.

<div align="center">

<img src="assets/forreadme/4.png" alt="Экран загрузки face detection в Face Swap Studio" width="600"/>

</div>

Detection step анализирует source identities и выбранные target images.

## Шаг 3. Просмотр найденных лиц

Секция `Detected faces` в основном является информационной.

Она показывает все найденные source faces слева и найденные target faces справа.

<div align="center">

<img src="assets/forreadme/5.png" alt="Найденные source и target faces в Face Swap Studio" width="600"/>

</div>

Source faces отображаются единым списком.

Target faces сгруппированы по target image. Благодаря этому сразу понятно, какие лица были найдены в каждом target-файле.

На каждую миниатюру найденного лица можно нажать, чтобы открыть крупный предпросмотр.

Эта секция помогает проверить, что detector нашёл нужные лица до настройки replacement mappings.

## Шаг 4. Настройка замен

Секция `Define replacements` используется для назначения source faces на target faces.

<div align="center">

<img src="assets/forreadme/6.png" alt="Визуальная настройка replacement mappings в Face Swap Studio" width="600"/>

</div>

В верхней части секции находится быстрый список source faces.

Нажатие на одно source face в этом списке применяет его сразу ко всем найденным target faces в выбранных targets. Это удобно, если одно и то же лицо нужно применить ко всем target-лицам.

Ниже быстрого блока Face Swap Studio показывает replacement groups для каждого target image.

У каждого target image есть отдельный mapping-блок. Внутри блока каждое найденное target face можно назначить на:

- `Keep` — оставить это target face без изменений;
- одно из найденных source faces — заменить это target face выбранной identity.

Mapping interface визуальный. Он использует миниатюры лиц, а не только номера или текстовые labels.

Каждую миниатюру лица можно открыть крупнее.

Это позволяет удобно задавать разные замены для разных target images и разных лиц внутри одного target image.

## Шаг 5. Выбор модели и настроек постобработки

Секция generation settings содержит выбор модели и дополнительные настройки post-processing.

Face-swap модель выбирается из выпадающего списка.

Ниже selector приложение показывает короткие описания доступных моделей.

Зелёная метка `Available` означает, что соответствующая модель присутствует, корректно настроена и готова к запуску.

Post-processing panel можно раскрыть для настройки дополнительных улучшений.

Доступные post-processing options:

- face restoration с настраиваемой restoration strength;
- image upscaling с настраиваемым upscale factor;
- tile size для upscaling.

По умолчанию post-processing выключен.

После выбора модели и нужных настроек нажмите `Generate selected targets`.

Во время генерации приложение показывает загрузочный экран.

<div align="center">

<img src="assets/forreadme/7.png" alt="Экран генерации в Face Swap Studio" width="600"/>

</div>

## Шаг 6. Просмотр и скачивание сгенерированных изображений

После завершения генерации результаты появляются в секции `Generated images`.

<div align="center">

<img src="assets/forreadme/8.png" alt="Сгенерированные изображения и кнопки скачивания в Face Swap Studio" width="600"/>

</div>

Большая область предпросмотра показывает текущий активный результат.

Если было сгенерировано несколько target images, под большим изображением появляются кнопки переключения:

- `Previous`;
- `Next`.

Ниже большого предпросмотра все сгенерированные результаты отображаются как маленькие cards.

Нажатие на result card выбирает соответствующий результат.

Нажатие на большой предпросмотр или миниатюру результата открывает изображение в крупном modal preview.

У каждой result card есть собственная кнопка `Download`. Она скачивает только конкретное сгенерированное изображение в папку `Загрузки`.

Верхняя кнопка действия в result-блоке меняется в зависимости от количества сгенерированных файлов:

- `Download active` — отображается, когда есть один сгенерированный результат;
- `Download archive` — отображается, когда есть несколько сгенерированных результатов.

`Download archive` создаёт и скачивает zip-архив со всеми текущими доступными сгенерированными результатами.

Если какое-то сгенерированное изображение не должно попасть в архив, наведите курсор на его result card и нажмите кнопку `×`. Удалённые result cards удаляются из текущей сессии и не включаются в архив.

## Временные данные и поведение приватности

Face Swap Studio хранит временные рабочие данные в системной временной директории.

К временным данным сессии относятся:

- загруженные source images;
- загруженные target images;
- detected face crops;
- данные target analysis;
- промежуточные working images;
- generated preview images;
- временные zip-архивы, созданные для скачивания.

Эти данные удаляются при нормальном завершении backend.

Приложение также выполняет дополнительную очистку при запуске. Это удаляет временные session data, оставшиеся от предыдущих запусков, если приложение или backend были закрыты аварийно.

Файлы, которые пользователь скачал вручную, не являются временными session files. Скачанные изображения и архивы сохраняются в папке `Загрузки` и остаются там, пока пользователь не удалит их вручную.

## Локальная разработка

Проект можно запускать из исходного кода.

Создайте или активируйте virtual environment, затем выполните:

    ./scripts/run.sh

Скрипт активирует `.venv`, устанавливает нужные environment variables и запускает `app.py`.

Backend приложения работает по адресу:

    http://127.0.0.1:7860

Environment variables, используемые launcher:

    PYTHONUNBUFFERED=1
    PYTHONPATH=<project-root>
    PYTORCH_ENABLE_MPS_FALLBACK=1
    TOKENIZERS_PARALLELISM=false

macOS application wrapper использует тот же backend entry point, но открывает UI в отдельном окне приложения.

## Структура проекта

    Face-Swap-Studio/
    ├── app.py
    │   └── Основной backend entry point
    ├── assets/
    │   ├── icons/
    │   │   └── Иконка приложения
    │   └── forreadme/
    │       └── Логотип и скриншоты для README
    ├── config/
    │   └── Runtime configuration
    ├── data/
    │   ├── input/
    │   ├── output/
    │   └── temp/
    ├── models/
    │   ├── detectors/
    │   ├── enhancers/
    │   ├── swappers/
    │   └── upscalers/
    ├── scripts/
    │   ├── проверки окружения
    │   ├── скрипты скачивания и проверки моделей
    │   ├── external model runner scripts
    │   ├── скрипты запуска
    │   └── скрипты упаковки macOS-приложения
    ├── src/face_swap_studio/
    │   ├── adapters/
    │   │   └── Реализации model adapters
    │   ├── core/
    │   │   └── Detection, swapping, masking, enhancement и upscaling pipeline
    │   ├── domain/
    │   │   └── Core entities и processing options
    │   ├── models/
    │   │   └── Model manifest и model manager
    │   ├── services/
    │   │   └── Batch и image services
    │   ├── ui/
    │   │   ├── FastAPI server
    │   │   ├── session storage
    │   │   └── static HTML, CSS и JavaScript UI
    │   └── utils/
    │       └── Logging и path utilities
    └── tests/
        └── Тесты adapters, detector и pipeline

## Сборка macOS application bundle

Локальный macOS application bundle можно собрать командой:

    ./scripts/build_macos_app.sh

Готовое приложение появится здесь:

    dist/Face Swap Studio.app

Создать zip-архив для переноса:

    ditto -c -k --sequesterRsrc --keepParent "dist/Face Swap Studio.app" "dist/Face Swap Studio.zip"

Для публикации в GitHub Release большой архив можно разбить на части:

    cd dist
    split -b 1900m "Face Swap Studio.zip" "Face Swap Studio.zip.part-"

Создать checksum:

    shasum -a 256 "Face Swap Studio.zip" > "Face Swap Studio.zip.sha256"

## Среда разработки и тестирования

Основная среда разработки и тестирования:

    MacBook Air с Apple Silicon
    macOS
    Homebrew Python 3.11

Python path, используемый во время разработки:

    /opt/homebrew/opt/python@3.11/bin/python3.11

Приложение настроено для локального выполнения на Apple Silicon и использует совместимые с macOS runtime settings там, где это необходимо.

## Примечания

Face Swap Studio — это локальная application wrapper и workflow-среда для нескольких внешних face-swap моделей.

Проект сфокусирован на:

- интеграции приложения;
- удобстве локального использования на macOS;
- совместимости с Apple Silicon;
- экспериментах с несколькими моделями;
- визуальном batch workflow;
- изоляции временных сессий;
- удобном локальном экспорте сгенерированных результатов.

Проект не обучает и не заявляет права собственности на интегрированные нейросетевые модели.