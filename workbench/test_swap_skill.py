"""Check that failed motions only retry after a verified return and fresh observation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from swap_skill import APPROVAL, SwapMission, authorize_text, stack_requested


class TestSwapSkill(unittest.TestCase):
    def test_rejects_negation_and_wrong_objects(self):
        for instruction in ('不要互换这两块', '交换橙色和浅灰色方块', '夹起橙色方块'):
            with self.assertRaises(ValueError):
                authorize_text(instruction)
        self.assertTrue(authorize_text('把橙色和带圆纹的深灰色方块互换'))

    def test_recognizes_only_orange_on_dark_gray(self):
        self.assertTrue(stack_requested('将橙色方块叠放在灰色方块上面'))
        self.assertFalse(stack_requested('把灰色方块叠放在橙色方块上面'))
        self.assertFalse(stack_requested('把橙色方块叠在浅灰色方块上面'))
        self.assertFalse(stack_requested('把橙色和灰色方块互换'))

    def exercise(self, fail_safely):
        events = []
        failures = []
        observations = []
        still = {'orange': (10, 10), 'dark_gray': (80, 80)}
        done = still | {'orange': (40, 10)}

        class Planner:
            def table_distance(self, a, b):
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

            def transfer(self, color, source, target, encoders, **extra):
                return {'bias': extra['bias']}

            def pickup_feedback(self, report, bias):
                return (3, 0)

        class Board:
            def write_json(self, path, plan):
                pass

            def robot(self, script, *args):
                if '--run' in args:
                    failures.append(args)
                return (1, '', 'wrist error') if len(failures) == 1 and '--run' in args else (0, '', '')

            def pull(self, run_id, files, target):
                target.mkdir()
                report = {'poses': [{'name': 'return_home'}] if fail_safely else [], 'final_torque': {'gripper': 0},
                          'released_at_destination': len(failures) > 1, 'error': 'wrist error' if len(failures) == 1 else None}
                (target / 'report.json').write_text(json.dumps(report))

        def observe():
            observations.append(True)
            return (done if len(failures) > 1 else still), [], {'gripper': 2085}

        mission = SwapMission.__new__(SwapMission)
        mission.board = Board()
        mission.planner = Planner()
        mission.id = 'test'
        mission.cancelled = lambda: False
        mission.event = lambda state, message, **_: events.append(state)
        mission.observe = observe
        with TemporaryDirectory() as directory:
            mission.root = Path(directory)
            if fail_safely:
                final, _ = mission.execute_leg(1, 'orange', (40, 10), still, {'gripper': 2085})
                self.assertEqual(final, done)
                self.assertIn('retrying', events)
                self.assertEqual(len(failures), 2)
                self.assertEqual(len(observations), 2)
            else:
                with self.assertRaises(RuntimeError):
                    mission.execute_leg(1, 'orange', (40, 10), still, {'gripper': 2085})
                self.assertEqual(len(failures), 1)
                self.assertFalse(observations)

    def test_retry_after_verified_return(self):
        self.exercise(fail_safely=True)

    def test_does_not_retry_if_return_is_unconfirmed(self):
        self.exercise(fail_safely=False)

    def test_resume_corrects_only_near_previously_verified_placement(self):
        initial = {'orange': (45, 10), 'dark_gray': (10, 10)}
        temporary = (80, 10)
        scene = {'orange': temporary, 'dark_gray': (45, 38)}
        route_end = {'orange': initial['dark_gray'], 'dark_gray': initial['orange']}
        checkpoint = {'instruction': '互换这两块', 'initial': initial, 'temporary': temporary,
                      'next_leg': 2, 'last_scene': {'orange': temporary, 'dark_gray': initial['dark_gray']}}

        mission = SwapMission.__new__(SwapMission)
        mission.id = 'correction'
        mission.board = Mock()
        mission.event = Mock()
        mission.observe = Mock(return_value=(scene, [('first.jpg', 'last.jpg')], {}))
        mission.planner = Mock()
        mission.planner.table_distance.side_effect = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
        mission.execute_leg = Mock(return_value=(route_end, {}))
        mission.finish = Mock()

        with TemporaryDirectory() as directory:
            pending = Path(directory) / 'pending.json'
            with patch('swap_skill.CHECKPOINT', pending), patch('swap_skill.verify_identity', return_value=APPROVAL):
                pending.write_text(json.dumps(checkpoint))
                with self.assertRaises(RuntimeError):
                    mission.run('继续上次换位')
                mission.execute_leg.assert_not_called()
                pending.write_text(json.dumps(checkpoint | {'verified_release': True}))
                mission.run('继续上次换位')

        correction, final = mission.execute_leg.call_args_list
        self.assertEqual(correction.args[:3], (2, 'dark_gray', [45, 10]))
        self.assertTrue(correction.kwargs['correction'])
        self.assertEqual(final.args[:3], (3, 'orange', [10, 10]))
        self.assertFalse(final.kwargs['correction'])
        mission.finish.assert_called_once()


if __name__ == '__main__':
    unittest.main()
