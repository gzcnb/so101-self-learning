# SO-101 Self-Learning Workbench

An experimental workbench for an SO-101 arm, an Orange Pi 5 Max, and a GPU workstation.
It combines typed Chinese instructions, two cameras, local vision-language supervision,
bounded manipulation, checkpoint recovery, and online tracking-error compensation.

One block-swap arrangement was verified on hardware after resumed execution.
Reliable stacking and a task policy learned from zero demonstrations remain open work.
The available RLS learner adapts joint compensation. The geometric manipulation
controller uses measured reference poses. The visual encoder and recurrent policy
definitions are included for research, with retraining and checkpoint publication
tracked as development tasks.

## Offline quick start

```sh
uv sync --python 3.12 --frozen
uv run python scripts/offline_demo.py
uv run python scripts/check_release.py
uv run python -m unittest discover -s workbench -p 'test_*.py' -v
uv run python workbench/swap_console.py
```

The demo uses a synthetic plant and never connects to hardware. The web console
binds to `127.0.0.1:8765`. Motion is disabled until a locally reviewed calibration
profile and explicit motion opt-in are present. See [setup](docs/setup.md),
[recorded results](docs/results.md), and [open tasks](docs/roadmap.md).

Apache-2.0. Geometry comes from TheRobotStudio/SO-ARM100 with attribution in NOTICE.
External LeRobot, llama.cpp, and model weights retain their own licenses.
