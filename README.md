# edge-cv-benchmark

Benchmark harness for 7+ CV models on M1 Max and Raspberry Pi 5.

Measures **latency (ms)**, **peak RAM (MB)**, **FPS**, and **accuracy** per model/device combo.

## Results — M1 Max (MPS)

| Model | Latency (ms) | FPS | RAM (MB) | License |
|-------|-------------|-----|----------|---------|
| YOLOv10n | 14.5 | 69.2 | 482 | AGPL-3.0 |
| YOLOv10s | 17.2 | 58.1 | 509 | AGPL-3.0 |
| YOLOv10m | 26.4 | 37.9 | 560 | AGPL-3.0 |
| DepthAnything V2 Small | 43.4 | 23.0 | 456 | Apache 2.0 |
| VideoMAE Small | 92.6 | 10.8 | 547 | CC-BY-NC 4.0 |
| SlowFast R50 | 149.4 | 6.7 | 821 | Apache 2.0 |
| MoViNet A2 (MViT proxy) | 147.7 | 6.8 | 864 | Apache 2.0 |

## Usage

```bash
# Single model
python benchmark.py --model yolov10n --device mps --runs 10

# All models
python benchmark.py --all --device mps --runs 10
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Structure

```
benchmark.py        # Main harness
results/            # JSON output per run
clips/              # Test video clips
requirements.txt    # Python deps
```

## Part of [deptedge](https://github.com/deptedge)

Domain-tuned CV models and MCP tools that let AI agents understand video — offline, on cheap hardware, without the cloud.
