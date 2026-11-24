"""Install a locally reviewed calibration profile into a workbench directory."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--workbench", type=Path, default=Path(__file__).resolve().parents[1] / "workbench")
    args = parser.parse_args()
    profile = args.profile.resolve()
    identity = json.loads((profile / "hardware.json").read_text(encoding="utf-8"))
    if identity.get("reviewed_for_this_robot") is not True:
        parser.error("Review your own calibration and set reviewed_for_this_robot in your own profile.")
    required = ["paper_plane.json", "base_alignment_p4_checked.json", "base_alignment_checked.json",
                "base_alignment_p4_verified.json", "alignment_arm_calibration.json", "swap_stage1_verified.json"]
    references = ["swap_orange_temp_01", "swap_gray_orange_01", "swap_orange_gray_05"]
    for name in required + [f"{folder}/report.json" for folder in references]:
        if not (profile / name).is_file():
            parser.error(f"Missing profile file {name}")
    for name in required + [f"{folder}/report.json" for folder in references]:
        target = args.workbench / "checks" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile / name, target)
    calibration = args.workbench / "calibration"
    calibration.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(profile / "alignment_arm_calibration.json", calibration / "so101-workbench.json")
    print("Installed calibration data. No hardware connection was made.")


if __name__ == "__main__":
    main()
