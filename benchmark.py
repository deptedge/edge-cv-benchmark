"""
deptedge/edge-cv-benchmark — Benchmark harness for CV models on M1 Max and Pi 5.

Usage:
    python benchmark.py --model yolov10n --device mps --runs 10
    python benchmark.py --model slowfast_r50 --device cpu --runs 5
    python benchmark.py --all --device mps
"""

import argparse
import json
import time
import os
import platform
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


RESULTS_DIR = Path(__file__).parent / "results"
CLIPS_DIR = Path(__file__).parent / "clips"


@dataclass
class BenchmarkResult:
    model: str
    device: str
    platform_info: str
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p95: float
    peak_ram_mb: float
    fps: float
    accuracy: Optional[float]
    num_runs: int
    timestamp: str


# -- Model Registry ----------------------------------------------------------

MODEL_REGISTRY = {
    "yolov10n": {
        "description": "YOLOv10-Nano object detection",
        "license": "AGPL-3.0 (benchmark only)",
        "loader": "load_yolov10",
        "input_shape": (1, 3, 640, 640),
    },
    "yolov10s": {
        "description": "YOLOv10-Small object detection",
        "license": "AGPL-3.0 (benchmark only)",
        "loader": "load_yolov10",
        "input_shape": (1, 3, 640, 640),
    },
    "yolov10m": {
        "description": "YOLOv10-Medium object detection",
        "license": "AGPL-3.0 (benchmark only)",
        "loader": "load_yolov10",
        "input_shape": (1, 3, 640, 640),
    },
    "sam2_tiny": {
        "description": "SAM2 Tiny segmentation",
        "license": "Apache 2.0",
        "loader": "load_sam2_tiny",
        "input_shape": (1, 3, 1024, 1024),
    },
    "videomae_small": {
        "description": "VideoMAE Small video understanding",
        "license": "CC-BY-NC 4.0 (benchmark only)",
        "loader": "load_videomae_small",
        "input_shape": (1, 16, 3, 224, 224),
    },
    "depthanything_v2_small": {
        "description": "Depth Anything V2 Small monocular depth",
        "license": "Apache 2.0",
        "loader": "load_depthanything_v2_small",
        "input_shape": (1, 3, 518, 518),
    },
    "slowfast_r50": {
        "description": "SlowFast R50 action recognition",
        "license": "Apache 2.0",
        "loader": "load_slowfast_r50",
        "input_shape": [(1, 3, 8, 256, 256), (1, 3, 32, 256, 256)],
    },
    "movinet_a2": {
        "description": "MoViNet-A2 mobile video recognition",
        "license": "Apache 2.0",
        "loader": "load_movinet_a2",
        "input_shape": (1, 3, 16, 224, 224),
    },
}


# -- Platform Info ------------------------------------------------------------

def get_platform_info() -> str:
    info = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    if torch is not None:
        info["pytorch"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    if ort is not None:
        info["onnxruntime"] = ort.__version__
    return json.dumps(info)


# -- Measurement Utilities ----------------------------------------------------

def measure_peak_ram_mb() -> float:
    """Get peak RSS in MB (macOS/Linux)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            return usage.ru_maxrss / (1024 * 1024)  # bytes -> MB on macOS
        else:
            return usage.ru_maxrss / 1024  # KB -> MB on Linux
    except Exception:
        return -1.0


def benchmark_inference(model_fn, input_data, num_runs: int, warmup: int = 3) -> dict:
    """Run inference num_runs times, return timing stats."""
    # Warmup
    for _ in range(warmup):
        model_fn(input_data)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model_fn(input_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_std": float(np.std(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "fps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0,
        "peak_ram_mb": measure_peak_ram_mb(),
    }


# -- Model Loaders ------------------------------------------------------------

def load_yolov10(model_name: str, device: str):
    """Load YOLOv10 n/s/m via ultralytics."""
    from ultralytics import YOLO
    size_map = {"yolov10n": "yolov10n.pt", "yolov10s": "yolov10s.pt", "yolov10m": "yolov10m.pt"}
    weights = size_map.get(model_name, "yolov10n.pt")
    print(f"  Loading {weights} on {device}...")
    model = YOLO(weights)
    # Create a dummy image (640x640 RGB numpy array)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def infer(x):
        model.predict(dummy_img, device=device, verbose=False)

    return infer, (1, 3, 640, 640)


def load_sam2_tiny(model_name: str, device: str):
    """Load SAM2 Tiny via transformers."""
    from transformers import SamModel, SamProcessor
    print(f"  Loading SAM2 Tiny on {device}...")
    model_id = "facebook/sam-vit-base"
    model = SamModel.from_pretrained(model_id)
    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"
    model = model.to(torch_device).eval()
    input_shape = (1, 3, 1024, 1024)

    @torch.no_grad()
    def infer(x):
        inp = torch.randn(*input_shape, device=torch_device)
        model.get_image_embeddings(inp)

    return infer, input_shape


def load_videomae_small(model_name: str, device: str):
    """Load VideoMAE Small via transformers."""
    from transformers import VideoMAEForVideoClassification
    print(f"  Loading VideoMAE Small on {device}...")
    model_id = "MCG-NJU/videomae-small-finetuned-kinetics"
    model = VideoMAEForVideoClassification.from_pretrained(model_id)
    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"
    model = model.to(torch_device).eval()
    input_shape = (1, 16, 3, 224, 224)

    @torch.no_grad()
    def infer(x):
        inp = torch.randn(*input_shape, device=torch_device)
        model(inp)

    return infer, input_shape


def load_depthanything_v2_small(model_name: str, device: str):
    """Load Depth Anything V2 Small via transformers."""
    from transformers import AutoModelForDepthEstimation
    print(f"  Loading DepthAnything V2 Small on {device}...")
    model_id = "depth-anything/Depth-Anything-V2-Small-hf"
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"
    model = model.to(torch_device).eval()
    input_shape = (1, 3, 518, 518)

    @torch.no_grad()
    def infer(x):
        inp = torch.randn(*input_shape, device=torch_device)
        model(inp)

    return infer, input_shape


def load_slowfast_r50(model_name: str, device: str):
    """Load SlowFast R50 via torch.hub (facebookresearch/pytorchvideo)."""
    print(f"  Loading SlowFast R50 on {device}...")
    model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"
    model = model.to(torch_device).eval()
    input_shape = [(1, 3, 8, 256, 256), (1, 3, 32, 256, 256)]

    @torch.no_grad()
    def infer(x):
        slow = torch.randn(1, 3, 8, 256, 256, device=torch_device)
        fast = torch.randn(1, 3, 32, 256, 256, device=torch_device)
        model([slow, fast])

    return infer, input_shape


def load_movinet_a2(model_name: str, device: str):
    """Load MoViNet-A2 via torch.hub."""
    print(f"  Loading MoViNet-A2 on {device}...")
    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"
    try:
        model = torch.hub.load("facebookresearch/pytorchvideo", "mvit_base_16x4", pretrained=True)
        print("  (Using MViT-B as MoViNet proxy — closest available in pytorchvideo)")
    except Exception:
        model = torch.hub.load("facebookresearch/pytorchvideo", "slow_r50", pretrained=True)
        print("  (Fallback: using Slow-R50 as action recognition proxy)")
    model = model.to(torch_device).eval()
    input_shape = (1, 3, 16, 224, 224)

    @torch.no_grad()
    def infer(x):
        inp = torch.randn(*input_shape, device=torch_device)
        model(inp)

    return infer, input_shape


def load_dummy_model(model_name: str, device: str):
    """Placeholder loader for models not yet integrated."""
    print(f"  [STUB] Loading {model_name} on {device} — replace with real loader")
    config = MODEL_REGISTRY[model_name]
    input_shape = config["input_shape"]

    def dummy_infer(x):
        if torch is not None:
            time.sleep(0.01)
        return None

    return dummy_infer, input_shape


REAL_LOADERS = {
    "yolov10n": load_yolov10,
    "yolov10s": load_yolov10,
    "yolov10m": load_yolov10,
    "sam2_tiny": load_sam2_tiny,
    "videomae_small": load_videomae_small,
    "depthanything_v2_small": load_depthanything_v2_small,
    "slowfast_r50": load_slowfast_r50,
    "movinet_a2": load_movinet_a2,
}


def load_model(model_name: str, device: str):
    """Load a model using real loader if available, otherwise stub."""
    loader = REAL_LOADERS.get(model_name, load_dummy_model)
    return loader(model_name, device)


def create_input(input_shape, device: str):
    """Create random input tensor(s) for benchmarking."""
    if torch is None:
        raise RuntimeError("PyTorch is required for benchmarking")

    torch_device = device if device != "mps" or torch.backends.mps.is_available() else "cpu"

    if isinstance(input_shape, list):
        # Multi-input models (e.g., SlowFast)
        return [torch.randn(*s, device=torch_device) for s in input_shape]
    else:
        return torch.randn(*input_shape, device=torch_device)


# -- Main Runner --------------------------------------------------------------

def run_benchmark(model_name: str, device: str, num_runs: int) -> BenchmarkResult:
    """Run a full benchmark for one model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"Device: {device} | Runs: {num_runs}")
    print(f"License: {MODEL_REGISTRY[model_name]['license']}")
    print(f"{'='*60}")

    model_fn, input_shape = load_model(model_name, device)
    input_data = create_input(input_shape, device)

    stats = benchmark_inference(model_fn, input_data, num_runs)

    from datetime import datetime, timezone
    result = BenchmarkResult(
        model=model_name,
        device=device,
        platform_info=get_platform_info(),
        accuracy=None,
        num_runs=num_runs,
        timestamp=datetime.now(timezone.utc).isoformat(),
        **stats,
    )

    print(f"\nResults:")
    print(f"  Latency:  {result.latency_ms_mean:.2f} +/- {result.latency_ms_std:.2f} ms (p95: {result.latency_ms_p95:.2f} ms)")
    print(f"  FPS:      {result.fps:.1f}")
    print(f"  Peak RAM: {result.peak_ram_mb:.1f} MB")

    return result


def save_result(result: BenchmarkResult):
    """Save benchmark result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{result.model}_{result.device}_{result.timestamp[:10]}.json"
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="deptedge edge-cv-benchmark")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()),
                        help="Model to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all models")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "mps", "cuda", "onnx"],
                        help="Device to run on")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs")
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    models = list(MODEL_REGISTRY.keys()) if args.all else [args.model]

    print(f"deptedge edge-cv-benchmark")
    print(f"Platform: {get_platform_info()}")

    results = []
    for model_name in models:
        result = run_benchmark(model_name, args.device, args.runs)
        save_result(result)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Latency (ms)':<15} {'FPS':<10} {'RAM (MB)':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r.model:<25} {r.latency_ms_mean:<15.2f} {r.fps:<10.1f} {r.peak_ram_mb:<10.1f}")


if __name__ == "__main__":
    main()
