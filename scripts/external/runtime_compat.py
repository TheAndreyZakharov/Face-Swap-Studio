from __future__ import annotations

import importlib
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Any

_COMPATIBILITY_MARKER = "_face_swap_studio_compatible"
_ORIGINAL_MARKER = "_face_swap_studio_original"


def configure_apple_silicon_environment(
    *,
    force_cpu: bool = False,
) -> None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "2",
    )
    os.environ.setdefault(
        "MKL_NUM_THREADS",
        "2",
    )
    os.environ.setdefault(
        "VECLIB_MAXIMUM_THREADS",
        "2",
    )
    os.environ.setdefault(
        "NUMEXPR_NUM_THREADS",
        "2",
    )

    if force_cpu:
        os.environ["FACE_SWAP_FORCE_CPU"] = "1"
    else:
        os.environ.pop(
            "FACE_SWAP_FORCE_CPU",
            None,
        )


def install_warning_filters() -> None:
    warning_patterns = (
        r".*Specified provider 'CUDAExecutionProvider'.*",
        r".*torch\.meshgrid.*indexing argument.*",
        r".*User provided device_type of 'cuda'.*",
        r".*is not currently supported on the MPS backend.*",
        r".*The parameter 'pretrained' is deprecated.*",
        r".*Arguments other than a weight enum.*",
        r".*_register_pytree_node is deprecated.*",
        r".*torch\.cuda\.amp\.autocast.*is deprecated.*",
        r".*HTTP_422_UNPROCESSABLE_ENTITY.*is deprecated.*",
        r".*estimate.*is deprecated since version 0\.26.*",
    )

    for pattern in warning_patterns:
        warnings.filterwarnings(
            "ignore",
            message=pattern,
        )


def install_numpy_compatibility(
    numpy_module: Any,
) -> None:
    aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
    }

    namespace = vars(
        numpy_module
    )

    for name, replacement in aliases.items():
        if name not in namespace:
            setattr(
                numpy_module,
                name,
                replacement,
            )


def should_force_cpu() -> bool:
    return os.environ.get(
        "FACE_SWAP_FORCE_CPU",
        "",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def select_torch_device(
    torch_module: Any,
) -> Any:
    if should_force_cpu():
        return torch_module.device(
            "cpu"
        )

    mps_backend = getattr(
        torch_module.backends,
        "mps",
        None,
    )

    if (
        mps_backend is not None
        and mps_backend.is_built()
        and mps_backend.is_available()
    ):
        return torch_module.device(
            "mps"
        )

    return torch_module.device(
        "cpu"
    )


def normalize_torch_device(
    torch_module: Any,
    value: Any,
    replacement: Any,
) -> Any:
    if should_force_cpu():
        replacement = torch_module.device(
            "cpu"
        )

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if (
            normalized == "cuda"
            or normalized.startswith(
                "cuda:"
            )
            or normalized == "mps"
            or normalized.startswith(
                "mps:"
            )
        ):
            return replacement

    if isinstance(
        value,
        torch_module.device,
    ):
        if value.type in {
            "cuda",
            "mps",
        }:
            return replacement

    return value


def normalize_map_location(
    torch_module: Any,
    value: Any,
) -> Any:
    if value is None:
        return torch_module.device(
            "cpu"
        )

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if (
            normalized == "cuda"
            or normalized.startswith(
                "cuda:"
            )
            or normalized == "mps"
            or normalized.startswith(
                "mps:"
            )
        ):
            return torch_module.device(
                "cpu"
            )

    if isinstance(
        value,
        torch_module.device,
    ):
        if value.type in {
            "cuda",
            "mps",
        }:
            return torch_module.device(
                "cpu"
            )

    if isinstance(
        value,
        dict,
    ):
        return {
            key: normalize_map_location(
                torch_module,
                destination,
            )
            for key, destination in value.items()
        }

    return value


def mark_compatible(
    replacement: Any,
    original: Any,
) -> Any:
    setattr(
        replacement,
        _COMPATIBILITY_MARKER,
        True,
    )
    setattr(
        replacement,
        _ORIGINAL_MARKER,
        original,
    )

    return replacement


def is_compatible(
    value: Any,
) -> bool:
    return bool(
        getattr(
            value,
            _COMPATIBILITY_MARKER,
            False,
        )
    )


def install_torch_load_compatibility(
    torch_module: Any,
) -> None:
    current_load = torch_module.load

    if is_compatible(
        current_load
    ):
        return

    original_load = current_load

    def compatible_load(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs["map_location"] = normalize_map_location(
            torch_module,
            kwargs.get(
                "map_location"
            ),
        )
        kwargs.setdefault(
            "weights_only",
            False,
        )

        return original_load(
            *args,
            **kwargs,
        )

    torch_module.load = mark_compatible(
        compatible_load,
        original_load,
    )


def create_compatible_factory(
    torch_module: Any,
    original_factory: Callable[..., Any],
    selected_device: Any,
) -> Callable[..., Any]:
    def compatible_factory(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "device" in kwargs:
            kwargs["device"] = normalize_torch_device(
                torch_module,
                kwargs["device"],
                selected_device,
            )

        return original_factory(
            *args,
            **kwargs,
        )

    return mark_compatible(
        compatible_factory,
        original_factory,
    )


def install_torch_factory_compatibility(
    torch_module: Any,
    selected_device: Any,
) -> None:
    factory_names = (
        "tensor",
        "as_tensor",
        "zeros",
        "zeros_like",
        "ones",
        "ones_like",
        "empty",
        "empty_like",
        "full",
        "full_like",
        "arange",
        "range",
        "linspace",
        "logspace",
        "eye",
        "rand",
        "rand_like",
        "randn",
        "randn_like",
        "randint",
        "randint_like",
        "normal",
        "scalar_tensor",
        "sparse_coo_tensor",
    )

    for factory_name in factory_names:
        original_factory = getattr(
            torch_module,
            factory_name,
            None,
        )

        if (
            original_factory is None
            or is_compatible(
                original_factory
            )
        ):
            continue

        setattr(
            torch_module,
            factory_name,
            create_compatible_factory(
                torch_module,
                original_factory,
                selected_device,
            ),
        )


def install_lightning_cuda_compatibility(
    selected_device: Any,
) -> None:
    module_names = (
        "lightning.fabric.utilities.device_dtype_mixin",
        "pytorch_lightning.core.mixins.device_dtype_mixin",
    )

    class_names = (
        "_DeviceDtypeModuleMixin",
        "DeviceDtypeModuleMixin",
    )

    for module_name in module_names:
        try:
            module = importlib.import_module(
                module_name
            )
        except ImportError:
            continue

        for class_name in class_names:
            mixin_class = getattr(
                module,
                class_name,
                None,
            )

            if mixin_class is None:
                continue

            current_cuda = getattr(
                mixin_class,
                "cuda",
                None,
            )

            if (
                current_cuda is None
                or is_compatible(
                    current_cuda
                )
            ):
                continue

            original_cuda = current_cuda

            def compatible_cuda(
                self: Any,
                device: Any = None,
                *,
                _selected_device: Any = selected_device,
            ) -> Any:
                del device

                return self.to(
                    _selected_device
                )

            mixin_class.cuda = mark_compatible(
                compatible_cuda,
                original_cuda,
            )


def install_autocast_compatibility(
    torch_module: Any,
    selected_device: Any,
) -> None:
    """
    Поддерживает оба старых варианта:

        @torch.cuda.amp.autocast(False)
        @torch.cuda.amp.autocast(enabled=False)

    Важно: False может быть первым позиционным аргументом.
    Нельзя одновременно передавать enabled=False ещё и через kwargs.
    """
    cuda_module = getattr(
        torch_module,
        "cuda",
        None,
    )
    cuda_amp_module = getattr(
        cuda_module,
        "amp",
        None,
    )

    if cuda_amp_module is not None:
        current_cuda_autocast = getattr(
            cuda_amp_module,
            "autocast",
            None,
        )

        if (
            current_cuda_autocast is not None
            and not is_compatible(
                current_cuda_autocast
            )
        ):
            original_cuda_autocast = current_cuda_autocast

            def compatible_cuda_autocast(
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                normalized_args = list(
                    args
                )

                if normalized_args:
                    normalized_args[0] = False
                    kwargs.pop(
                        "enabled",
                        None,
                    )
                else:
                    kwargs["enabled"] = False

                return original_cuda_autocast(
                    *normalized_args,
                    **kwargs,
                )

            cuda_amp_module.autocast = mark_compatible(
                compatible_cuda_autocast,
                original_cuda_autocast,
            )

    amp_module = getattr(
        torch_module,
        "amp",
        None,
    )

    if amp_module is None:
        return

    current_autocast = getattr(
        amp_module,
        "autocast",
        None,
    )

    if (
        current_autocast is None
        or is_compatible(
            current_autocast
        )
    ):
        return

    original_autocast = current_autocast

    def compatible_autocast(
        device_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        normalized_device_type = device_type

        if device_type == "cuda":
            normalized_device_type = selected_device.type

        normalized_args = list(
            args
        )

        if selected_device.type == "cpu":
            if normalized_args:
                normalized_args[0] = False
                kwargs.pop(
                    "enabled",
                    None,
                )
            else:
                kwargs["enabled"] = False

        return original_autocast(
            normalized_device_type,
            *normalized_args,
            **kwargs,
        )

    amp_module.autocast = mark_compatible(
        compatible_autocast,
        original_autocast,
    )


def install_torch_cuda_compatibility(
    torch_module: Any,
) -> str:
    selected_device = select_torch_device(
        torch_module
    )

    try:
        torch_module.set_num_threads(
            max(
                1,
                int(
                    os.environ.get(
                        "OMP_NUM_THREADS",
                        "2",
                    )
                ),
            )
        )
    except (
        AttributeError,
        RuntimeError,
        ValueError,
    ):
        pass

    current_tensor_cuda = torch_module.Tensor.cuda

    if not is_compatible(
        current_tensor_cuda
    ):
        original_tensor_cuda = torch_module.Tensor.cuda
        original_module_cuda = torch_module.nn.Module.cuda
        original_tensor_to = torch_module.Tensor.to
        original_module_to = torch_module.nn.Module.to

        def tensor_cuda(
            tensor: Any,
            device: Any = None,
            non_blocking: bool = False,
            memory_format: Any = torch_module.preserve_format,
        ) -> Any:
            del device

            return original_tensor_to(
                tensor,
                device=selected_device,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )

        def module_cuda(
            module: Any,
            device: Any = None,
        ) -> Any:
            del device

            return original_module_to(
                module,
                device=selected_device,
            )

        def tensor_to(
            tensor: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            normalized_args = list(
                args
            )

            if normalized_args:
                normalized_args[0] = normalize_torch_device(
                    torch_module,
                    normalized_args[0],
                    selected_device,
                )

            if "device" in kwargs:
                kwargs["device"] = normalize_torch_device(
                    torch_module,
                    kwargs["device"],
                    selected_device,
                )

            return original_tensor_to(
                tensor,
                *normalized_args,
                **kwargs,
            )

        def module_to(
            module: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            normalized_args = list(
                args
            )

            if normalized_args:
                normalized_args[0] = normalize_torch_device(
                    torch_module,
                    normalized_args[0],
                    selected_device,
                )

            if "device" in kwargs:
                kwargs["device"] = normalize_torch_device(
                    torch_module,
                    kwargs["device"],
                    selected_device,
                )

            return original_module_to(
                module,
                *normalized_args,
                **kwargs,
            )

        torch_module.Tensor.cuda = mark_compatible(
            tensor_cuda,
            original_tensor_cuda,
        )
        torch_module.nn.Module.cuda = mark_compatible(
            module_cuda,
            original_module_cuda,
        )
        torch_module.Tensor.to = mark_compatible(
            tensor_to,
            original_tensor_to,
        )
        torch_module.nn.Module.to = mark_compatible(
            module_to,
            original_module_to,
        )

    install_torch_factory_compatibility(
        torch_module,
        selected_device,
    )
    install_lightning_cuda_compatibility(
        selected_device
    )
    install_autocast_compatibility(
        torch_module,
        selected_device,
    )

    return str(
        selected_device
    )


def install_torch_compatibility(
    torch_module: Any,
) -> str:
    install_warning_filters()
    install_torch_load_compatibility(
        torch_module
    )

    return install_torch_cuda_compatibility(
        torch_module
    )


def preferred_onnx_providers(
    ort_module: Any,
) -> list[str]:
    available = list(
        ort_module.get_available_providers()
    )

    if "CPUExecutionProvider" in available:
        return [
            "CPUExecutionProvider",
        ]

    if (
        not should_force_cpu()
        and "CoreMLExecutionProvider" in available
    ):
        return [
            "CoreMLExecutionProvider",
        ]

    return available


def normalize_onnx_providers(
    ort_module: Any,
    providers: Sequence[Any] | None,
) -> list[Any]:
    available = set(
        ort_module.get_available_providers()
    )

    if "CPUExecutionProvider" in available:
        return [
            "CPUExecutionProvider",
        ]

    normalized: list[Any] = []

    for provider in providers or ():
        if isinstance(
            provider,
            tuple,
        ):
            provider_name = provider[0]
        else:
            provider_name = provider

        if not isinstance(
            provider_name,
            str,
        ):
            continue

        if provider_name == "CUDAExecutionProvider":
            continue

        if (
            should_force_cpu()
            and provider_name == "CoreMLExecutionProvider"
        ):
            continue

        if provider_name in available:
            normalized.append(
                provider
            )

    if normalized:
        return normalized

    return preferred_onnx_providers(
        ort_module
    )


def install_onnxruntime_compatibility(
    ort_module: Any,
) -> list[str]:
    install_warning_filters()

    current_session = ort_module.InferenceSession

    if is_compatible(
        current_session
    ):
        return preferred_onnx_providers(
            ort_module
        )

    original_session = current_session

    class CompatibleInferenceSession(original_session):
        _face_swap_studio_compatible = True
        _face_swap_studio_original = original_session

        def __init__(
            self,
            path_or_bytes: Any,
            sess_options: Any = None,
            providers: Sequence[Any] | None = None,
            provider_options: Any = None,
            **kwargs: Any,
        ) -> None:
            del provider_options

            selected_providers = normalize_onnx_providers(
                ort_module,
                providers,
            )

            super().__init__(
                path_or_bytes,
                sess_options=sess_options,
                providers=selected_providers,
                provider_options=None,
                **kwargs,
            )

        def set_providers(
            self,
            providers: Sequence[Any] | None = None,
            provider_options: Any = None,
        ) -> None:
            del provider_options

            selected_providers = normalize_onnx_providers(
                ort_module,
                providers,
            )

            super().set_providers(
                selected_providers,
                None,
            )

    ort_module.InferenceSession = CompatibleInferenceSession

    return preferred_onnx_providers(
        ort_module
    )