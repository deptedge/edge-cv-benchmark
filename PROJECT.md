---
name: "edge-cv-benchmark"
description: "Benchmark harness for 7+ models on M1 Max and Pi 5"
owner: "forge"
contributors: ["neuron", "pixel"]
phase: 1
---

## Overview

Public benchmark repository measuring latency (ms), peak RAM (MB), FPS, and accuracy for 7+ CV models on M1 Max (MPS) and Raspberry Pi 5 (ARM64 + ONNX Runtime).

## Models to Benchmark

- YOLOv10 n/s/m
- SAM2 tiny
- VideoMAE small
- DepthAnything V2 small
- SlowFast R50
- MoViNet or MobileNetV4 action variant

## Metrics

| Metric | Unit |
|--------|------|
| Latency | ms |
| Peak RAM | MB |
| FPS | frames/sec |
| Accuracy | % on test clips |

## Milestones

| Milestone | Target | Week |
|-----------|--------|------|
| M1 | First model benchmarked on MPS | 2 |
| Baseline | All 7 models benchmarked on M1 Max | 4 |
| Pi 5 | Pi 5 + INT8 numbers added | 8 |
| M6 | Full report published, 100+ GitHub stars | 13 |

## Deliverables

- Rich GitHub README with tables and charts
- Reproducible harness scripts
- Raw JSON results per model per platform
