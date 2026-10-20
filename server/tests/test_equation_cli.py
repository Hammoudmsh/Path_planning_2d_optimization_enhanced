import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from _load_search import SOURCE


class EquationCLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('equation_config', SOURCE.with_name('benchmark_config.py'))
        cls.config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.config)

    def test_cli_test_profile_and_baseline_isolation(self):
        run = subprocess.run([
            sys.executable, str(SOURCE), 'street', 'test',
            '--lambda', '5.35', '--beta', '0', '--w', '1.15', '--dry-run',
        ], cwd=SOURCE.parents[1], capture_output=True, text=True, check=True)
        config = json.loads(run.stdout)
        self.assertEqual((config['map_limit'], config['scenarios_per_map']), (2, 1))
        self.assertEqual(config['scenario_selection'], 'longest')
        standard, original, optimized = config['benchmark_approaches']
        self.assertEqual(standard['parameters']['heuristic_weight'], 1.0)
        self.assertEqual(original['parameters'], {'turn_weight': 0.35, 'heuristic_weight': 1.15})
        self.assertEqual(optimized['parameters'], {
            'turn_weight': 5.35, 'turn_count_weight': 0.0, 'heuristic_weight': 1.15})
        self.assertTrue(config['resume_benchmark'])
        self.assertFalse(config['force_recompute_benchmark'])

    def test_overrides_do_not_mutate_profiles_and_have_distinct_outputs(self):
        base = self.config.get_config('street', 'test', project_root=SOURCE.parents[1])
        first = self.config.apply_equation_overrides(base, lambda_weight=0.3500001, beta_weight=0)
        second = self.config.apply_equation_overrides(base, lambda_weight=0.3500002, beta_weight=0)
        self.assertNotEqual(first['results'], second['results'])
        self.assertEqual(base['benchmark_approaches'][-1]['parameters']['turn_weight'], 0.35)
        self.assertIs(self.config.apply_equation_overrides(base), base)
        self.assertEqual(first['benchmark_cache_dir'], Path(first['results']) / 'cache')

    def test_invalid_equation_coefficients(self):
        base = self.config.get_config('street', 'test', project_root=SOURCE.parents[1])
        for kwargs in ({'lambda_weight': -1}, {'lambda_weight': float('nan')},
                       {'beta_weight': -1}, {'beta_weight': float('inf')},
                       {'heuristic_weight': 0}, {'heuristic_weight': float('inf')}):
            with self.assertRaises(ValueError):
                self.config.apply_equation_overrides(base, **kwargs)

    def test_launch_directory_does_not_change_paths_or_checkpoint_identity(self):
        for use_json in (False, True):
            configurations = []
            for cwd in (SOURCE.parents[1], SOURCE.parent):
                command = [sys.executable, str(SOURCE), 'street', 'test',
                           '--lambda', '0.35', '--beta', '5', '--w', '1.15', '--dry-run']
                if use_json:
                    command.extend(['--config-json', str(SOURCE.with_name('exp_config.json'))])
                output = subprocess.run(command, cwd=cwd, capture_output=True,
                                        text=True, check=True)
                configurations.append(json.loads(output.stdout))
            self.assertEqual(configurations[0], configurations[1])
            self.assertEqual(Path(configurations[0]['maps_dir']),
                             SOURCE.parents[1] / 'datasets' / 'street' / 'maps')


if __name__ == '__main__':
    unittest.main()
