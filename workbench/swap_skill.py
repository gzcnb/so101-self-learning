"""Coordinate perception, bounded retries, and confirmed progress for one swap."""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import time
import uuid

import cv2
import paramiko

from settings import BOARD_HOST, BOARD_USER, BOARD_DIR, KNOWN_HOSTS, STATE_DIR, require_hardware_enabled

from supervise_snapshot import find_orange_target, run_vlm
from swap_skill_plan import WorkbenchPlanner, read_json
from swap_vision import identify, temporary_candidates


ROOT = Path(__file__).resolve().parent
APPROVAL = {'action': 'swap', 'object_a': 'orange', 'object_b': 'dark_gray_with_ring'}
LABEL = {'orange': '橙色方块', 'dark_gray': '带圆纹的深灰色方块'}
FRAME_ROOT = BOARD_DIR + '/checks/swap_console_frames'
STATE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = STATE_DIR / 'pending.json'
STACK_FIT = CHECKPOINT.with_name('stack_fit.json')
PLACEMENT_TOLERANCE = 25
CORRECTION_LIMIT = 34


def normalize_instruction(instruction):
    phrase = re.sub(r'[\s，。！、？,.!?]+', '', instruction)
    if not phrase or len(phrase) > 100 or any(word in phrase for word in ('不要', '不必', '不用', '不想', '别', '取消', '停止', '顺便', '然后')):
        raise ValueError('请输入一条明确的方块操作指令')
    return phrase


def authorize_text(instruction):
    phrase = normalize_instruction(instruction)
    has_swap = any(word in phrase for word in ('互换', '交换', '对调', '换位', '换个位置', '换一下'))
    has_pair = ('橙' in phrase and '灰' in phrase) or any(word in phrase for word in ('这两块', '这俩', '这两个方块', '两块方块'))
    if not has_swap or not has_pair or any(word in phrase for word in ('浅灰', '白色', '蓝色', '黄色', '红色', '绿色')):
        raise ValueError('当前技能支持互换橙色方块与带圆纹的深灰色方块')
    return phrase


def stack_requested(instruction):
    phrase = normalize_instruction(instruction)
    return ('橙' in phrase and '灰' in phrase and phrase.index('橙') < phrase.index('灰')
            and not any(word in phrase for word in ('浅灰', '白色', '蓝色', '黄色', '红色', '绿色', '互换', '交换'))
            and any(word in phrase for word in ('叠', '摞', '放在', '放到', '放上'))
            and phrase.endswith(('上', '上面', '上方', '顶面')))


def resume_requested(text):
    return text.strip().rstrip('。！') in ('继续上次换位', '继续换位')


def last_error(message):
    lines = message.strip().splitlines()
    return lines[-1] if lines else '控制器没有返回详细信息'


def save_checkpoint(progress):
    draft = CHECKPOINT.with_suffix('.part')
    draft.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')
    draft.replace(CHECKPOINT)


class Board:
    def __init__(self):
        self.client = paramiko.SSHClient()
        self.client.load_host_keys(str(KNOWN_HOSTS))

    def __enter__(self):
        self.client.connect(BOARD_HOST, username=BOARD_USER, password=os.environ['SO101_SSH_PASSWORD'],
                            timeout=8, auth_timeout=8, banner_timeout=8, allow_agent=False, look_for_keys=False)
        self.client.get_transport().set_keepalive(10)
        return self

    def __exit__(self, *exception):
        self.client.close()

    def command(self, command, timeout=220):
        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        outgoing = stdout.read().decode('utf-8', errors='replace')
        failure = stderr.read().decode('utf-8', errors='replace')
        return stdout.channel.recv_exit_status(), outgoing, failure

    def ensure_cameras(self):
        from settings import REMOTE_PYTHON
        run = "cd " + shlex.quote(BOARD_DIR)
        run += " && mkdir -p -m 700 checks/swap_console_frames"
        run += " && touch checks/swap_console_frames/heartbeat"
        run += " && { nohup " + shlex.quote(REMOTE_PYTHON) + " swap_camera_stream.py"
        run += " </dev/null >>checks/swap_console_frames/stream.log 2>&1 & }"
        code, _, message = self.command(run, timeout=12)
        if code:
            raise RuntimeError("无法启动双路相机。" + message.strip())

    def write_json(self, relative, content):
        with self.client.open_sftp() as sftp, sftp.open(relative, 'w') as output:
            output.write(json.dumps(content, ensure_ascii=False, indent=2) + '\n')

    def pull(self, run_id, names, out):
        out.mkdir(parents=True, exist_ok=True)
        with self.client.open_sftp() as sftp:
            for name in names:
                sftp.get(f'{BOARD_DIR}/checks/{run_id}/{name}', str(out / name))

    def robot(self, script, *args):
        from settings import REMOTE_PYTHON
        require_hardware_enabled()
        command = "cd " + shlex.quote(BOARD_DIR)
        command += " && export SO101_ENABLE_MOTION=1"
        command += ' && export SO101_FRAME_DIR="$PWD/checks/swap_console_frames"'
        command += " && flock -n checks/controller.lock " + shlex.quote(REMOTE_PYTHON)
        command += " " + shlex.join([script, *args])
        return self.command(command)


def verify_identity(image, scene, phrase, *, action='swap'):
    approval = APPROVAL | {'action': action}
    schema = {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': [action, 'pause']},
        'object_a': {'type': 'string', 'enum': ['orange', 'none']},
        'object_b': {'type': 'string', 'enum': ['dark_gray_with_ring', 'none']}},
        'required': list(APPROVAL), 'additionalProperties': False}
    intent = '互换两块' if action == 'swap' else '把橙块叠在深灰块顶面'
    prompt = (f'用户指令：{phrase}。固定相机纸面的稳定候选像素：{json.dumps(scene)}。'
              '核对纸上的橙色方块和深灰色圆纹方块。只选这两块，忽略纸外灰块和纸上的浅色块。'
              f'用户要{intent}且两个候选位置与图像相符，返回 action={action}，object_a=orange，'
              'object_b=dark_gray_with_ring。目标有歧义或指令不同则返回 action=pause，物体都为 none。')
    answer = run_vlm(image, prompt, schema)
    if answer != approval:
        raise RuntimeError('Qwen 无法确认这条指令指向纸面上的两块方块')
    return answer


def verify_stack_result(image, placement):
    frame = cv2.imread(str(image))
    x, y = map(round, placement['support_px'])
    section = frame[y - 53:y + 44, x - 50:x + 51]
    zoom = image.with_name(image.stem + '_stack_zoom.jpg')
    cv2.imwrite(str(zoom), cv2.resize(section, (505, 485), interpolation=cv2.INTER_CUBIC))
    schema = {'type': 'object', 'properties': {'stacked': {'type': 'boolean'}},
              'required': ['stacked'], 'additionalProperties': False}
    prompt = ('这是机械臂松手并移开后，固定相机拍下的纸面中央近景。'
              '若中央橙色方块的底面由紧贴在下方的深灰色方块顶面支撑，两块组成一座塔，返回 stacked=true。'
              '如果橙块在桌面上、两块并排、或者画面无法判断，返回 stacked=false。')
    return run_vlm(zoom, prompt, schema).get('stacked') is True


class SwapMission:
    def __init__(self, board, publish, cancelled):
        self.board = board
        self.event = publish
        self.cancelled = cancelled
        self.planner = WorkbenchPlanner()
        self.id = datetime.now().strftime('swap_%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:5]
        self.root = STATE_DIR / self.id
        self.inspections = 0

    def observe(self):
        self.inspections += 1
        run_id = f'{self.id}_scene_{self.inspections}'
        directory = self.root / run_id
        status, _, error = self.board.robot('swap_calibrated_scene.py', run_id)
        if status:
            raise RuntimeError('机械臂未完成安全观察。' + last_error(error))
        names = [f'scene_{i}_{repeat}_fixed.jpg' for i in range(3) for repeat in ('a', 'b')]
        self.board.pull(run_id, ['report.json', *names], directory)
        report = read_json(directory / 'report.json')
        if any(report['final_torque'].values()):
            raise RuntimeError('未确认六个舵机关闭扭矩')
        pairs = [(directory / names[i], directory / names[i + 1]) for i in (0, 2)]
        try:
            scene = identify(pairs, self.planner.h)
        except RuntimeError:
            pairs = [(directory / names[4], directory / names[5])]
            scene = identify(pairs, self.planner.h)
        return scene, pairs, report['released_positions']

    def find_temporary(self, scene, pairs, encoders):
        for temporary in temporary_candidates(pairs, scene, self.planner.h):
            try:
                self.planner.transfer('orange', scene['orange'], temporary, encoders)
                self.planner.transfer('dark_gray', scene['dark_gray'], scene['orange'], encoders)
                self.planner.transfer('orange', temporary, scene['dark_gray'], encoders)
            except ValueError:
                continue
            return temporary
        raise RuntimeError('纸面内找不到同时满足空白、间距和行程的暂放点')

    def confirm_park(self, run_id, report):
        reached = {pose['name'] for pose in report['poses']}
        if any(report['final_torque'].values()):
            raise RuntimeError('舵机仍有扭矩，已停止后续动作')
        if 'return_home' in reached:
            return
        aperture = report.get('released_positions', {}).get('gripper', 0)
        returned_block = 'return_to_source' in reached and 2452 <= aperture <= 2468
        empty_approach = 'clearance' in reached and 'source_hover' not in reached and 2076 <= aperture <= 2094
        if (returned_block or empty_approach) and not self.cancelled():
            status, _, _ = self.board.robot('swap_park_open.py', run_id)
            if not status:
                parked_id = run_id + '_park_open'
                location = self.root / parked_id
                self.board.pull(parked_id, ['report.json'], location)
                parked = read_json(location / 'report.json')
                if not any(parked['final_torque'].values()) and any(
                        pose['name'] == 'return_home' for pose in parked['poses']):
                    return
        raise RuntimeError('未确认机械臂安全退回，已停止后续动作')

    def execute_leg(self, index, color, target, scene, encoders, pending=None, *, correction=False):
        initial_source = scene[color]
        bias = (0, 0)
        extra_lift = 0
        for attempt in range(1, 5):
            if self.cancelled():
                raise RuntimeError('已在本次搬运开始前停止')
            run_id = f'{self.id}_leg_{index}_{attempt}'
            plan = self.planner.transfer(color, scene[color], target, encoders, bias=bias,
                                         extra_lift=extra_lift, correction=correction)
            plan_name = run_id + '.json'
            self.board.write_json(f'{BOARD_DIR}/checks/{plan_name}', plan)
            preflight, _, error = self.board.robot('swap_blocks.py', run_id, plan_name)
            if preflight:
                raise RuntimeError('位姿预检未通过。' + last_error(error))
            self.event('moving', f'第 {index}/3 步，搬运{LABEL[color]}（第 {attempt} 次）', step=index, attempt=attempt)
            status, _, error = self.board.robot('swap_blocks.py', run_id, plan_name, '--run')
            try:
                directory = self.root / run_id
                self.board.pull(run_id, ['report.json'], directory)
                report = read_json(directory / 'report.json')
            except OSError as exc:
                raise RuntimeError('搬运状态未知，已停止后续动作') from exc
            self.confirm_park(run_id, report)
            if pending and (report.get('placement') or {}).get('verified'):
                pending = pending | {'verified_release': True}
                save_checkpoint(pending)
            self.event('checking', f'核对第 {index}/3 步落点' if not status else f'第 {index}/3 步触发保护，重新确认现场', step=index)
            target_scene, _, encoders = self.observe()
            other = next(name for name in scene if name != color)
            if self.planner.table_distance(target_scene[other], scene[other]) > 24:
                raise RuntimeError(f'{LABEL[other]}的位置也发生变化，已停止以保留现场')
            target_error = self.planner.table_distance(target_scene[color], target)
            if report.get('released_at_destination') and target_error <= PLACEMENT_TOLERANCE:
                self.event('checked', f'第 {index}/3 步已确认', step=index, offset_mm=round(target_error, 1))
                return target_scene, encoders
            if report.get('released_at_destination') or self.planner.table_distance(target_scene[color], initial_source) > 32:
                raise RuntimeError(f'{LABEL[color]}离开了该步骤的可重试范围，已停止后续动作')
            scene = target_scene
            fault = report.get('phase_error') or report.get('error') or error
            if 'height guard' in fault.lower():
                extra_lift = min(12, extra_lift + 6)
            elif 'wrist' in fault.lower() or 'fit between' in fault.lower() or 'off center' in fault.lower():
                bias = self.planner.pickup_feedback(report, bias)
            self.event('retrying', f'第 {index}/3 步已重新定位，准备重试', step=index, attempt=attempt)
        raise RuntimeError(f'{LABEL[color]}达到重试上限，机械臂已经退回')

    def offsets(self, scene, initial):
        return {color: self.planner.table_distance(scene[color], initial[other]) for color, other in
                (('orange', 'dark_gray'), ('dark_gray', 'orange'))}

    def finish(self, scene, initial, temporary, *, already_swapped=False):
        errors = self.offsets(scene, initial)
        if max(errors.values()) > PLACEMENT_TOLERANCE:
            raise RuntimeError('最终画面未通过整体换位验收')
        result = {'initial': initial, 'final': scene, 'temporary': temporary,
                  'approximate_offsets_mm': {color: round(error, 1) for color, error in errors.items()},
                  'verified_existing_arrangement': already_swapped}
        (self.root / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        CHECKPOINT.unlink(missing_ok=True)
        message = '桌面已处于换位后的布局，未再搬运' if already_swapped else '换位完成，两块方块都已放稳'
        self.event('complete', message, step=3, result=result)

    def run(self, instruction):
        if stack_requested(instruction):
            if CHECKPOINT.exists():
                raise ValueError('请先完成进行中的换位任务')
            self.run_stack(normalize_instruction(instruction))
            return
        checkpoint = read_json(CHECKPOINT) if resume_requested(instruction) and CHECKPOINT.exists() else None
        if resume_requested(instruction) and checkpoint is None:
            raise ValueError('没有可继续的换位任务')
        if checkpoint is None and CHECKPOINT.exists():
            raise ValueError('有一项换位进行到一半，请输入「继续上次换位」')
        phrase = authorize_text(checkpoint['instruction'] if checkpoint else instruction)
        self.event('checking', '正在定位纸面上的两块方块', step=0, run_id=self.id)
        scene, pairs, encoders = self.observe()
        initial = checkpoint['initial'] if checkpoint else {color: tuple(point) for color, point in scene.items()}
        agreement = verify_identity(pairs[-1][-1], scene, phrase)
        self.board.write_json(BOARD_DIR + '/checks/swap_supervision.json', agreement)
        temporary = checkpoint['temporary'] if checkpoint else self.find_temporary(scene, pairs, encoders)
        next_leg = checkpoint['next_leg'] if checkpoint else 1
        if checkpoint and max(self.offsets(scene, initial).values()) <= 16:
            self.finish(scene, initial, temporary, already_swapped=True)
            return
        route = (('orange', temporary), ('dark_gray', initial['orange']), ('orange', initial['dark_gray']))
        correction = False
        if checkpoint:
            moving, target = route[next_leg - 1]
            for color, center in checkpoint['last_scene'].items():
                if color != moving and self.planner.table_distance(center, scene[color]) > 14:
                    raise RuntimeError('现场与上次停下时不同，无法安全继续原任务')
            if checkpoint.get('verified_release'):
                distance = self.planner.table_distance(scene[moving], target)
                if distance <= PLACEMENT_TOLERANCE:
                    next_leg += 1
                elif distance <= CORRECTION_LIMIT:
                    correction = True
                else:
                    raise RuntimeError('方块已离开此前核验的落点附近，无法安全继续原任务')
            elif self.planner.table_distance(checkpoint['last_scene'][moving], scene[moving]) > 34:
                raise RuntimeError('现场与上次停下时不同，无法安全继续原任务')
        self.event('planned', 'Qwen 已确认目标，继续上次换位' if checkpoint else 'Qwen 已确认目标，暂放点已选好',
                   step=next_leg - 1, points={**scene, 'temporary': temporary})
        for index, (color, target) in enumerate(route, 1):
            if index < next_leg:
                continue
            pending = {'instruction': phrase, 'initial': initial, 'temporary': temporary,
                       'next_leg': index, 'last_scene': scene}
            if correction:
                pending['verified_release'] = True
            save_checkpoint(pending)
            scene, encoders = self.execute_leg(index, color, target, scene, encoders, pending, correction=correction)
            correction = False
        self.finish(scene, initial, temporary)

    def run_stack(self, phrase):
        self.event('checking', '正在定位橙块和深灰底座', step=0, total_steps=1, run_id=self.id)
        scene, pairs, encoders = self.observe()
        agreement = verify_identity(pairs[-1][-1], scene, phrase, action='stack')
        self.board.write_json(BOARD_DIR + '/checks/swap_supervision.json', agreement)
        bias = (0, 0)
        trim = read_json(STACK_FIT) if STACK_FIT.exists() else None
        if trim and self.planner.table_distance(trim['target_px'], scene['dark_gray']) > 25:
            trim = None
        for attempt in range(1, 4):
            if self.cancelled():
                raise RuntimeError('已在叠放开始前停止')
            run_id = f'{self.id}_stack_{attempt}'
            plan = self.planner.transfer('orange', scene['orange'], scene['dark_gray'], encoders, bias=bias,
                                         stack=True, placement_trim=trim)
            plan_name = run_id + '.json'
            self.board.write_json(f'{BOARD_DIR}/checks/{plan_name}', plan)
            status, _, error = self.board.robot('swap_blocks.py', run_id, plan_name)
            if status and plan['edge_pickup'] and 'checked tabletop patch' in error:
                alternate = self.planner.transfer('orange', scene['orange'], scene['dark_gray'], encoders,
                                                  bias=(12, 5), stack=True, placement_trim=trim)
                alternate['edge_pickup'] = True
                self.board.write_json(f'{BOARD_DIR}/checks/{plan_name}', alternate)
                status, _, error = self.board.robot('swap_blocks.py', run_id, plan_name)
                plan = alternate
            if status:
                raise RuntimeError('叠放位姿预检未通过。' + last_error(error))
            self.event('moving', f'抓取橙块并叠放（第 {attempt} 次）', step=1, attempt=attempt)
            status, _, error = self.board.robot('swap_blocks.py', run_id, plan_name, '--run')
            directory = self.root / run_id
            self.board.pull(run_id, ['report.json'], directory)
            report = read_json(directory / 'report.json')
            self.confirm_park(run_id, report)
            placement = report.get('placement')
            if placement and placement['verified'] and not status:
                names = ['stack_b_fixed.jpg', 'stack_b_wrist.jpg', 'reset_fixed.jpg']
                self.board.pull(run_id, names, directory)
                self.event('checking', '机械臂已退回，Qwen 正在复核叠放结果', step=1)
                if not verify_stack_result(directory / 'reset_fixed.jpg', placement):
                    candidates = find_orange_target(directory / 'reset_fixed.jpg')
                    stayed = any(self.planner.table_distance(item['center_px'], placement['observed_px']) <= 7
                                 for item in candidates)
                    if not stayed or not verify_stack_result(directory / 'stack_b_fixed.jpg', placement):
                        raise RuntimeError('相机检测到两块靠近，Qwen 未能确认稳定叠放')
                result = {'placement': placement, 'report': run_id, 'supervisor_verified': True}
                (self.root / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
                self.event('complete', '叠放完成，橙块已停稳在灰块上', step=1, result=result)
                return
            if report.get('released_at_destination'):
                updated, _, encoders = self.observe()
                if (self.planner.table_distance(scene['dark_gray'], updated['dark_gray']) > 14
                        or self.planner.table_distance(updated['orange'], updated['dark_gray']) < 34):
                    raise RuntimeError('松爪后未确认叠放，当前布局不满足安全重试条件')
                trim = self.planner.stack_feedback(report, scene['dark_gray'], trim)
                STACK_FIT.write_text(json.dumps(trim, indent=2) + '\n', encoding='utf-8')
                self.event('retrying', '橙块已滑落，已重新定位并修正叠放落点', step=1, attempt=attempt)
                bias = (0, 0)
                scene = updated
                continue
            updated, _, encoders = self.observe()
            if any(self.planner.table_distance(scene[color], updated[color]) > 14 for color in scene):
                raise RuntimeError('叠放过程中方块位置已改变，机械臂已退回')
            fault = report.get('phase_error') or report.get('error') or error
            if 'did not align with the support' in fault.lower():
                trim = self.planner.stack_feedback(report, scene['dark_gray'], trim, lower=False)
                STACK_FIT.write_text(json.dumps(trim, indent=2) + '\n', encoding='utf-8')
                scene = updated
                self.event('retrying', '已退回，按松爪前的画面修正落点', step=1, attempt=attempt)
                continue
            if 'wrist' not in fault.lower() and 'off center' not in fault.lower() and 'fit between' not in fault.lower():
                raise RuntimeError('叠放保护已触发。' + last_error(fault))
            bias = self.planner.pickup_feedback(report, plan['pickup_bias_mm'])
            scene = updated
            self.event('retrying', '已退回并重新定位，准备重试叠放', step=1, attempt=attempt)
        raise RuntimeError('已达到叠放重试上限，机械臂已经退回')
