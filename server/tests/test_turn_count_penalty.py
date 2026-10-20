import gc
import math
import unittest

import numpy as np

from _load_search import load_search


class TurnCountPenaltyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_search()

    def assert_objective_matches_path(self, grid, start, goal, result, beta):
        self.assertTrue(self.ns['validate_path'](grid, result.path, start, goal)[0])
        turns, angle, _ = self.ns['path_turn_metrics'](result.path)
        self.assertEqual(result.search_turn_count, turns)
        expected = self.ns['path_length'](result.path) + 0.35 * angle / 45 + beta * turns
        self.assertAlmostEqual(result.search_objective, expected, places=7)

    def test_positive_penalty_objective_and_path_validity(self):
        for seed in range(30):
            rng = np.random.default_rng(seed)
            grid = rng.random((20, 24)) > 0.2
            grid[0, 0] = grid[-1, -1] = True
            for beta in (1.0, 2.0, 5.0):
                result = self.ns['run_optimized_fwsjps'](
                    grid, (0, 0), (23, 19), turn_count_weight=beta)
                if result.success:
                    self.assert_objective_matches_path(grid, (0, 0), (23, 19), result, beta)
                else:
                    self.assertTrue(math.isnan(result.search_turn_count))
            gc.collect()

    def test_initial_heading_and_straight_path_have_no_turn_penalty(self):
        grid = np.ones((1, 10), dtype=bool)
        for goal in ((0, 0), (9, 0)):
            result = self.ns['run_optimized_fwsjps'](
                grid, (0, 0), goal, turn_count_weight=100)
            self.assertEqual(result.search_turn_count, 0)
            self.assertEqual(result.search_objective, goal[0])

    def test_penalty_can_select_a_path_with_fewer_turns(self):
        grid = np.random.default_rng(1).random((20, 24)) > 0.2
        grid[0, 0] = grid[-1, -1] = True
        baseline = self.ns['run_optimized_fwsjps'](grid, (0, 0), (23, 19))
        smoother = self.ns['run_optimized_fwsjps'](
            grid, (0, 0), (23, 19), turn_count_weight=5.0)
        self.assertTrue(baseline.success and smoother.success)
        self.assertLess(smoother.search_turn_count, baseline.search_turn_count)
        self.assert_objective_matches_path(grid, (0, 0), (23, 19), smoother, 5.0)

    def test_dispatcher_records_parameter_and_rejects_invalid_values(self):
        grid = np.ones((5, 5), dtype=bool)
        result = self.ns['run_all_algorithms'](grid, (0, 0), (4, 4), [{
            'label': 'smoother', 'algorithm': 'Optimized Fast Weighted SA-JPS',
            'parameters': {'turn_count_weight': 5.0},
        }])[0]
        self.assertEqual(result.turn_count_weight, 5.0)
        self.assertEqual(result.parameters['turn_count_weight'], 5.0)
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                self.ns['run_optimized_fwsjps'](grid, (0, 0), (4, 4), turn_count_weight=value)


if __name__ == '__main__':
    unittest.main()
