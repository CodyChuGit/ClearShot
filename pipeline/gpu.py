from __future__ import annotations

"""
GPU detection and ONNX Runtime session management.

Detects available GPU acceleration (CUDA, CoreML, CPU) and provides
a factory for creating ONNX Runtime inference sessions with the
best available provider.
"""

import platform
import functools
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class GpuInfo:
    backend: str       # "cuda", "coreml", "cpu"
    device: str        # Human-readable device name
    provider: str      # ONNX Runtime provider name

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


COREML_COMPUTE_UNITS = os.environ.get("CLEARSHOT_COREML_COMPUTE_UNITS", "CPUAndGPU")


@functools.lru_cache(maxsize=1)
def detect_gpu() -> dict[str, str]:
    """
    Auto-detect the best available GPU acceleration.

    Returns a dict with keys: backend, device, provider.
    Result is cached — only runs once per process.
    """
    providers = _get_available_providers()

    if os.environ.get("CLEARSHOT_DISABLE_COREML") == "1":
        providers = [p for p in providers if p != "CoreMLExecutionProvider"]

    # Try CUDA first (NVIDIA)
    if "CUDAExecutionProvider" in providers:
        device = _get_cuda_device_name()
        return GpuInfo(
            backend="cuda",
            device=device,
            provider="CUDAExecutionProvider",
        ).to_dict()

    # Try CoreML (Apple Silicon)
    if "CoreMLExecutionProvider" in providers:
        device = _get_apple_device_name()
        return GpuInfo(
            backend="coreml",
            device=device,
            provider="CoreMLExecutionProvider",
        ).to_dict()

    # CPU fallback
    return GpuInfo(
        backend="cpu",
        device="CPU",
        provider="CPUExecutionProvider",
    ).to_dict()


def get_providers() -> list[Any]:
    """
    Return ordered list of ONNX Runtime execution providers.
    Best available provider comes first, CPU last as fallback.
    """
    gpu = detect_gpu()
    provider = gpu["provider"]

    if provider == "CUDAExecutionProvider":
        return [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "DEFAULT",
            }),
            "CPUExecutionProvider",
        ]
    elif provider == "CoreMLExecutionProvider":
        _prepare_coreml_temp_dir()
        return [
            ("CoreMLExecutionProvider", {
                # CPUAndGPU maps CoreML work to the Metal GPU path instead of CPU-only.
                # CPU remains available for operators CoreML cannot partition.
                "MLComputeUnits": COREML_COMPUTE_UNITS,
            }),
            "CPUExecutionProvider",
        ]
    else:
        return ["CPUExecutionProvider"]


def create_session(model_path: str, **kwargs):
    """
    Create an ONNX Runtime InferenceSession with the best available provider.

    Args:
        model_path: Path to the .onnx model file.
        **kwargs: Additional session options.

    Returns:
        ort.InferenceSession
    """
    import onnxruntime as ort

    providers = get_providers()
    if _uses_coreml(providers):
        _prepare_coreml_temp_dir()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Limit threads for desktop use
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    return _create_session_with_fallback(ort, model_path, sess_options, providers)


def create_cpu_session(model_path: str, **kwargs):
    """Create an ONNX Runtime session pinned to CPU."""
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_available_providers() -> list[str]:
    """Get list of available ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except ImportError:
        return ["CPUExecutionProvider"]


def _create_session_with_fallback(ort, model_path: str, sess_options, providers: list[Any]):
    try:
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
    except Exception as exc:
        if providers == ["CPUExecutionProvider"]:
            raise

        print(f"[ClearShot] GPU provider failed for {model_path} ({exc}); retrying on CPU")
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )


def _uses_coreml(providers: list[Any]) -> bool:
    for provider in providers:
        if provider == "CoreMLExecutionProvider":
            return True
        if isinstance(provider, tuple) and provider[0] == "CoreMLExecutionProvider":
            return True
    return False


def _prepare_coreml_temp_dir() -> str:
    """
    CoreML model compilation fails if macOS hands it the bare temp root URL.
    Point TMPDIR at an app-owned subdirectory so CoreML can create its working
    package and compiled model artifacts reliably.
    """
    tmp_dir = Path(os.environ.get("CLEARSHOT_COREML_TMPDIR", tempfile.gettempdir()))
    if tmp_dir.name != "clearshot_coreml_tmp":
        tmp_dir = tmp_dir / "clearshot_coreml_tmp"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir) + os.sep
    return os.environ["TMPDIR"]


def _get_cuda_device_name() -> str:
    """Try to get the NVIDIA GPU name."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "NVIDIA GPU"


def _get_apple_device_name() -> str:
    """Get Apple Silicon chip name."""
    if platform.system() != "Darwin":
        return "Apple GPU"
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "Apple Silicon"
