# Docker for go2-semantic-nav

## Dev container (`dev.Dockerfile`)
For x86_64 development on the dev workstation. Contains ROS 2 Humble + Nav2 + the full Python ML stack.

Build:
```bash
docker build -f docker/dev.Dockerfile -t go2-semantic-nav:dev .
```

Run with GPU passthrough:
```bash
docker run --rm -it --gpus all \
  -v "$HOME/Projects/personal/go2-semantic-nav:/workspace" \
  --network=host \
  go2-semantic-nav:dev
```

## Jetson container (TODO: Phase 4)
Planned base: `nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.3-py3` for JetPack 6.x.
Will add TensorRT engine-export helpers and a lightweight entry script.

Until then, run directly on the Jetson following `docs/deployment.md`.
