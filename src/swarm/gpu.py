from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    total_vram_gb: float
    uuid: str | None = None


def _env_override() -> GpuSpec | None:
    v = os.getenv("SWARM_GPU_VRAM_GB")
    if not v:
        return None
    try:
        gb = float(v)
        return GpuSpec(name=os.getenv("SWARM_GPU_NAME", "env_override"), total_vram_gb=gb, uuid=None)
    except ValueError:
        return None


def detect_gpu() -> GpuSpec | None:
    """
    Best-effort single-GPU detection.

    Order:
    - SWARM_GPU_VRAM_GB env override
    - NVML via pynvml (if installed)
    - nvidia-smi (if present)
    - None if nothing found
    """
    env = _env_override()
    if env is not None:
        return env

    # NVML (optional)
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count <= 0:
                return None
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            total_gb = float(mem.total) / (1024.0**3)
            return GpuSpec(name=name.decode() if isinstance(name, (bytes, bytearray)) else str(name), total_vram_gb=total_gb, uuid=str(uuid))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass

    # nvidia-smi fallback
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        ).strip()
        if not out:
            return None
        # first GPU
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            name = parts[0]
            mem_mb = float(parts[1])
            uuid = parts[2] if len(parts) >= 3 else None
            return GpuSpec(name=name, total_vram_gb=mem_mb / 1024.0, uuid=uuid)
    except Exception:
        pass

    return None


def clamp_vram(requested_gb: float, detected: GpuSpec | None) -> float:
    """
    If user passed an explicit --gpu_vram_gb, keep it.
    Otherwise, use detected VRAM if available.
    """
    if requested_gb > 0:
        return float(requested_gb)
    if detected is None:
        return 0.0
    return float(detected.total_vram_gb)


def get_effective_vram_gb() -> float:
    """
    Returns a sensible VRAM value for experiments.

    - If GPU is detected, returns detected total VRAM.
    - If CPU/unknown GPU, returns a default (24GB) so routing is non-trivial.
    """
    info = detect_gpu()
    if info is None:
        return 24.0
    return float(info.total_vram_gb)

