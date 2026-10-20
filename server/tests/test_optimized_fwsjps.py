import ast
import json
import importlib.util
import math
import unittest

import numpy as np

from _load_search import SOURCE, load_search


class OptimizedFWSJPSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_search()

    def compare_search(self, grid, start, goal, **parameters):
        original = self.ns['run_sa_jps'](grid, start, goal, **parameters)
        optimized = self.ns['run_optimized_fwsjps'](grid, start, goal, **parameters)
        for field in ('success', 'path', 'expanded_nodes', 'generated_successors',
                      'queue_pushes', 'jump_calls'):
            self.assertEqual(getattr(original, field), getattr(optimized, field), field)
        if original.success:
            self.assertEqual(original.search_objective, optimized.search_objective)
            self.assertEqual(original.turn_severity, optimized.turn_severity)
            self.assertTrue(self.ns['validate_path'](grid, optimized.path, start, goal)[0])
        else:
            self.assertTrue(math.isnan(optimized.search_objective))

    def test_all_small_occupancy_patterns(self):
        # Exhaust all 3x3 maps, goals and free ray origins. This covers forced
        # neighbours, boundaries, blocked corners, cache hits and dead ends.
        for mask in range(512):
            grid = np.array([(mask >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
            points = [(x, y) for y in range(3) for x in range(3) if grid[y, x]]
            for goal in points:
                original, old_counter = self.ns['build_jump_function'](grid, goal)
                optimized, new_counter = self.ns['build_optimized_jump_function'](grid, goal)
                for x, y in points:
                    for dx, dy in self.ns['DIRECTIONS_8']:
                        self.assertEqual(original(x, y, dx, dy), optimized(x, y, dx, dy))
                self.assertEqual(old_counter, new_counter)

    def test_random_queries_and_parameter_settings(self):
        rng = np.random.default_rng(20261008)
        for i in range(300):
            grid = rng.random((18, 23)) > (0.1 + (i % 5) * 0.12)
            start, goal = (0, 0), (22, 17)
            grid[0, 0] = grid[17, 22] = True
            self.compare_search(grid, start, goal,
                                turn_weight=(0.0, 0.35, 1.0)[i % 3],
                                heuristic_weight=(1.0, 1.15, 1.5)[(i // 3) % 3])

    def test_start_equals_goal_and_disconnected(self):
        grid = np.ones((5, 5), dtype=bool)
        self.compare_search(grid, (2, 2), (2, 2), heuristic_weight=1.15)
        grid[2, :] = False
        self.compare_search(grid, (0, 0), (4, 4), heuristic_weight=1.15)

    def test_long_corridor_without_recursion_or_shared_goal_cache(self):
        grid = np.ones((1, 5000), dtype=bool)
        for goal in ((4999, 0), (2500, 0)):
            result = self.ns['run_optimized_fwsjps'](grid, (0, 0), goal)
            self.assertTrue(result.success)
            self.assertEqual(len(result.path), goal[0] + 1)
        grid[0, 1250] = False
        self.assertFalse(self.ns['run_optimized_fwsjps'](grid, (0, 0), (2500, 0)).success)

    def test_dispatcher_and_parameters(self):
        grid = np.ones((5, 5), dtype=bool)
        names = ['Standard JPS', 'Fast Weighted SA-JPS', 'Optimized Fast Weighted SA-JPS']
        results = self.ns['run_all_algorithms'](grid, (0, 0), (4, 4), names)
        self.assertEqual([result.algorithm for result in results], names)
        results = self.ns['run_all_algorithms'](grid, (0, 0), (4, 4), [{
            'label': 'custom', 'algorithm': names[2],
            'parameters': {'turn_weight': 0.5, 'heuristic_weight': 1.3},
        }])
        self.assertEqual((results[0].algorithm, results[0].turn_weight,
                          results[0].heuristic_weight), ('custom', 0.5, 1.3))
        with self.assertRaises(ValueError):
            self.ns['run_all_algorithms'](grid, (0, 0), (4, 4), [{
                'algorithm': names[2], 'parameters': {'unsupported': True},
            }])
        for kwargs in ({'turn_weight': -1}, {'heuristic_weight': 0}):
            with self.assertRaises(ValueError):
                self.ns['run_optimized_fwsjps'](grid, (0, 0), (4, 4), **kwargs)

    def test_notebook_search_definitions_match_script(self):
        notebook = json.loads(SOURCE.with_name(
            'Astar_JPS_SA_JPS_dataset_suffix_plot_data_mean_median_std.ipynb'
        ).read_text(encoding='utf-8'))
        names = {'run_jps', 'run_sa_jps', 'build_jump_function',
                 'build_optimized_jump_function', 'run_optimized_fwsjps',
                 'run_all_algorithms', 'BenchmarkCheckpoint', 'benchmark_dataset', 'package_results',
                 '_benchmark_cache_signature', '_save_benchmark_cache',
                 '_run_benchmark_approach', 'obstacle_percentage'}
        script = {n.name: ast.dump(n) for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}
        found = {}
        for cell in notebook['cells']:
            code = ''.join(cell['source'])
            if cell['cell_type'] == 'code' and any(
                'def ' + name + '(' in code or 'class ' + name + ':' in code for name in names
            ):
                for node in ast.parse(code).body:
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                        found[node.name] = ast.dump(node)
        self.assertEqual(script, found)

    def test_comparison_configuration_and_single_execution(self):
        spec = importlib.util.spec_from_file_location(
            'comparison_config', SOURCE.with_name('benchmark_config.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        quick = module.get_config('street', 'optimized_vs_jps')
        full = json.loads(SOURCE.with_name('exp_config.json').read_text())['overrides']
        for config in (quick, full):
            approaches = config['benchmark_approaches']
            self.assertEqual([item['algorithm'] for item in approaches],
                             ['Standard JPS', 'Fast Weighted SA-JPS', 'Optimized Fast Weighted SA-JPS'])
            self.assertEqual(approaches[0]['parameters']['heuristic_weight'], 1.0)
            self.assertEqual(approaches[1]['parameters'], approaches[2]['parameters'])
            self.assertFalse(config['run_ablation_study'])
            self.assertFalse(config['run_inference_example'])
            self.assertFalse(config['generate_all_dataset_descriptions'])
            self.assertTrue(config['resume_benchmark'])
            self.assertTrue(config['use_benchmark_cache'])
            self.assertFalse(config['force_recompute_benchmark'])
        tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
        calls = [node for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == 'run_experiment']
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
