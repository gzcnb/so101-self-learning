"""Prototype that validates task selection against a captured SO-101 scene."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

import cv2


PICK_COMMAND = re.compile(
    r"\s*(?:(?:请|麻烦|帮我)(?:你)?\s*)?(?:拿起|抓起|抓|夹起|捡起|拾起)"
    r"\s*(?:桌面?上(?:的)?)?(?:那个|这个|一个|那块|这块)?"
    r"(?:橙色|蓝色|红色|绿色|黄色|白色|黑色|灰色)(?:的)?(?:方块|积木|块)[。！？!?]?\s*"
)


def find_orange_target(image_path, view='fixed', *, minimum_aspect=0.4):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Cannot read image {image_path}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (4, 150, 140), (26, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    targets = []
    max_area = 4000 if view == 'fixed' else int(frame.shape[0] * frame.shape[1] * 0.12)
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        if 100 <= area <= max_area and minimum_aspect <= width / height <= 2.5:
            targets.append({
                "id": f"orange_candidate_{len(targets) + 1}",
                "description": "橙色候选物体",
                "box_px": [x, y, width, height],
                "center_px": [round(float(value), 1) for value in centers[index]],
            })
    return targets


def get_decision(image_path, instruction, targets):
    if not PICK_COMMAND.fullmatch(instruction):
        return {"action": "pause", "target_id": "none", "reason_code": "unsupported_task"}
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["select_target", "clarify", "pause"]},
            "target_id": {"type": "string", "enum": ["none", *(target["id"] for target in targets)]},
            "reason_code": {"type": "string", "enum": ["matched", "target_unavailable", "ambiguous", "unsupported_task"]},
        },
        "required": ["action", "target_id", "reason_code"],
        "additionalProperties": False,
    }
    prompt = (
        "你负责监督桌面机械臂。这个工位当前只支持选择图中的橙色方块作为抓取目标。"
        "结合用户指令、照片和候选框确认指向的物体。候选物体和指令必须一致。"
        "只有唯一且清楚匹配的橙色方块时才返回 select_target、对应编号和 matched。"
        "如果找不到、目标有歧义、要求放入容器或动作尚不支持，则返回 clarify 或 pause，target_id 为 none。"
        "reason_code 从 target_unavailable、ambiguous、unsupported_task 选择一项。返回 JSON。\n"
        f"用户指令：{instruction}\n候选物体（像素坐标）：{json.dumps(targets, ensure_ascii=False)}"
    )
    decision = run_vlm(image_path, prompt, schema)
    if (decision["action"] == "select_target" and decision["reason_code"] == "matched"
            and len(targets) == 1 and decision["target_id"] == targets[0]["id"]):
        return decision
    action = decision["action"] if decision["action"] in {"clarify", "pause"} else "pause"
    reason_code = decision["reason_code"]
    if reason_code not in {"target_unavailable", "ambiguous", "unsupported_task"}:
        reason_code = "target_unavailable" if len(targets) <= 1 else "ambiguous"
    return {"action": action, "target_id": "none", "reason_code": reason_code}


def run_vlm(image_path, prompt, schema):
    from settings import VLM_DIR, VLM_EXECUTABLE, VLM_DEVICE
    runtime = VLM_DIR
    with tempfile.TemporaryDirectory(dir=runtime) as directory:
        schema_file = Path(directory, "decision.json")
        schema_file.write_text(json.dumps(schema), encoding="utf-8")
        prompt_file = Path(directory, "prompt.txt")
        prompt_file.write_text(prompt, encoding="utf-8")
        result = subprocess.run(
            [
                str(VLM_EXECUTABLE),
                "--model", str(runtime / "model.gguf"),
                "--mmproj", str(runtime / "projector.gguf"),
                "--device", VLM_DEVICE, "--gpu-layers", "all",
                "--no-warmup", "--ctx-size", "2048", "--image-max-tokens", "256",
                "--temp", "0", "--predict", "96",
                "--image", str(image_path.resolve()),
                "--file", str(prompt_file), "--json-schema-file", str(schema_file),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=85,
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])
    return json.loads(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--view", choices=('fixed', 'wrist'), default='fixed')
    args = parser.parse_args()
    targets = find_orange_target(args.image, args.view)
    output = {"candidates": targets}
    if args.instruction:
        output["decision"] = get_decision(args.image, args.instruction, targets)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
