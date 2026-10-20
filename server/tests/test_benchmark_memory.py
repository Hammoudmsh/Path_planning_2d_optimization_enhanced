import gc
import unittest
import weakref

import numpy as np

from _load_search import load_search
import test_benchmark_resume as resume_fixture
from test_benchmark_resume import environment, run


class SearchMemoryTest(unittest.TestCase):
    def test_obstacle_count_does_not_invert_grid(self):
        ns = load_search(extra_functions={'obstacle_percentage'})

        class NoInversion(np.ndarray):
            def __invert__(self):
                raise AssertionError('Must not allocate an inverted grid')

        for free_cells in (0, 1, 27, 64):
            grid = np.zeros(64, dtype=bool)
            grid[:free_cells] = True
            grid = grid.reshape(8, 8).view(NoInversion)
            self.assertEqual(ns['obstacle_percentage'](grid),
                             100.0 * (64 - free_cells) / 64)

    def test_jump_caches_are_released_after_each_algorithm(self):
        ns = load_search(extra_functions={'_run_benchmark_approach'})
        references = []
        for name in ('build_jump_function', 'build_optimized_jump_function'):
            builder = ns[name]

            def track(grid, goal, builder=builder):
                jump, counter = builder(grid, goal)
                references.append(weakref.ref(jump))
                return jump, counter

            ns[name] = track

        was_enabled = gc.isenabled()
        gc.disable()
        try:
            grid = np.ones((24, 24), dtype=bool)
            # Demonstrate that returning from the original search is not enough
            # to release its recursive closure and memoized scan states.
            ns['run_sa_jps'](grid, (0, 0), (23, 23))
            self.assertIsNotNone(references[-1]())
            gc.collect()
            self.assertIsNone(references[-1]())
            for name in ('Standard JPS', 'Fast Weighted SA-JPS',
                         'Optimized Fast Weighted SA-JPS'):
                results = ns['_run_benchmark_approach'](grid, (0, 0), (23, 23), name)
                self.assertTrue(results[0].success)
                self.assertTrue(all(reference() is None for reference in references))
        finally:
            if was_enabled:
                gc.enable()
            gc.collect()


class MapMemoryTest(unittest.TestCase):
    def test_only_current_and_example_maps_are_retained(self):
        # Reuse the small on-disk dataset fixture, not the other test methods.
        fixture = resume_fixture.BenchmarkResumeTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        for index in range(1, 4):
            name = f'tiny{index}.map'
            (root / name).write_text((root / 'tiny.map').read_text())
            (root / (name + '.scen')).write_text(
                (root / 'tiny.map.scen').read_text().replace('tiny.map', name))
        ns, _, _ = environment(root)
        ns['selected_maps'] = sorted(root.glob('*.map'))
        read = ns['read_map']
        references = []

        def tracked_read(path):
            grid = read(path)
            references.append(weakref.ref(grid))
            self.assertLessEqual(sum(ref() is not None for ref in references), 2)
            return grid

        ns['read_map'] = tracked_read
        original_progress = ns['tqdm']

        def checked_progress(**kwargs):
            self.assertEqual(references, [], 'Map arrays loaded before benchmark starts')
            return original_progress(**kwargs)

        ns['tqdm'] = checked_progress
        original_obstacles = ns['obstacle_percentage']
        obstacle_calls = []

        def tracked_obstacles(grid):
            obstacle_calls.append(grid.shape)
            return original_obstacles(grid)

        ns['obstacle_percentage'] = tracked_obstacles
        frame, _ = run(ns, use_cache=False)
        self.assertEqual(len(frame), 36)
        self.assertEqual(len(obstacle_calls), 4)
        self.assertEqual(len(references), 4)


if __name__ == '__main__':
    unittest.main()
