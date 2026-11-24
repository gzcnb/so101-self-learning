"""Local configuration shared by the desktop and board processes."""

import json
import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
CONFIG_PATH = Path(os.environ.get("SO101_CONFIG", APP_DIR / "settings.local.json")).expanduser()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}


def setting(name, default):
    return os.environ.get("SO101_" + name.upper(), CONFIG.get(name, default))


def local_path(name, default):
    path = Path(setting(name, default)).expanduser()
    return (path if path.is_absolute() else REPO_DIR / path).resolve()


BOARD_HOST = setting("host", "")
BOARD_USER = setting("user", "robot")
BOARD_DIR = setting("board_dir", "so101-workbench")
REMOTE_PYTHON = setting("remote_python", ".venv/bin/python")
KNOWN_HOSTS = Path(setting("known_hosts", str(Path.home() / ".ssh/known_hosts"))).expanduser()
SERIAL_PORT = setting("serial_port", "/dev/ttyACM0")
FIXED_CAMERA = setting("fixed_camera", "/dev/video0")
WRIST_CAMERA = setting("wrist_camera", "/dev/video2")
PROFILE_DIR = local_path("profile_dir", "examples/reference-profile")
STATE_DIR = local_path("state_dir", "runtime/console")
WEB_PORT = int(setting("web_port", 8765))
VLM_DIR = local_path("vlm_dir", "models/qwen-vl")
VLM_EXECUTABLE = Path(setting("vlm_executable", str(VLM_DIR / "llama-mtmd-cli"))).expanduser()
VLM_DEVICE = setting("vlm_device", "CPU")


def require_hardware_enabled():
    """Check explicit motion opt-in and the per-robot profile before touching torque."""
    enabled = str(setting("enable_motion", "0")).lower() in ("1", "true")
    if not enabled:
        raise RuntimeError("Hardware motion is disabled. Configure your robot and set SO101_ENABLE_MOTION=1.")
    reference = (REPO_DIR / "examples/reference-profile").resolve()
    if PROFILE_DIR == reference:
        raise RuntimeError("The reference profile is for offline inspection. Select a calibrated profile for your arm.")
    identity = PROFILE_DIR / "hardware.json"
    if not identity.exists() or json.loads(identity.read_text(encoding="utf-8")).get("reviewed_for_this_robot") is not True:
        raise RuntimeError("Your profile needs hardware.json with reviewed_for_this_robot set to true.")
