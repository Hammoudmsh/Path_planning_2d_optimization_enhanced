"""Exercise real searches and durable restart, including abrupt process exit."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import pandas as pd

from _load_search import load_search


APPROACHES = [
    {'label': 'Standard JPS', 'algorithm': 'Standard JPS',
     'parameters': {'heuristic_weight': 1.0}},
    {'label': 'Original FWSJPS', 'algorithm': 'Fast Weighted SA-JPS',
     'parameters': {'turn_weight': 0.35, 'heuristic_weight': 1.15}},
    {'label': 'Optimized FWSJPS', 'algorithm': 'Optimized Fast Weighted SA-JPS',
     'parameters': {'turn_weight': 0.35, 'heuristic_weight': 1.15}},
]


class SilentProgress:
    def __init__(self, **kwargs):
        self.initial = kwargs.get('initial', 0)
        self.completed = self.initial

    def update(self, count):
        self.completed += count

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def set_postfix(self, *args, **kwargs):
        pass


def environment(root, crash_after=None):
    ns = load_search(extra_functions={
        'BenchmarkCheckpoint', 'benchmark_dataset', 'active_dataset_name',
        'dataset_output_path', 'locate_file', 'select_scenarios', 'obstacle_percentage',
        '_benchmark_cache_signature', '_benchmark_cache_paths', '_load_cached_benchmark',
        '_save_benchmark_cache', '_atomic_pickle_save', '_atomic_json_save',
        '_run_benchmark_approach',
    }, extra_constants={'BENCHMARK_CACHE_VERSION'})
    ns.update(pd=pd, json=json, os=os, re=re, hashlib=hashlib, defaultdict=defaultdict)
    ns['CONFIG'].update(dataset_name='tiny', scenarios_per_map=None,
                        scenario_selection='uniform', random_seed=42,
                        resume_benchmark=True, use_benchmark_cache=True,
                        force_recompute_benchmark=False, benchmark_cache_dir=root / 'cache')
    ns.update(RESULTS_DIR=root, SCENARIOS_DIR=root, MAPS_DIR=root,
              selected_maps=[root / 'tiny.map'])
    feature_names = ('Obstacle density', 'Free-cell ratio', 'Dead-end ratio',
                     'Corridor ratio', 'Junction ratio', 'Obstacle-edge density',
                     'Mean axis visibility', 'Connected components', 'Largest component ratio')
    ns['get_map_type'] = lambda grid: ('test', dict.fromkeys(feature_names, 0.0))
    progress = []

    def make_progress(**kwargs):
        bar = SilentProgress(**kwargs)
        progress.append(bar)
        return bar

    ns['tqdm'] = make_progress
    real_run = ns['run_all_algorithms']
    calls = []

    def tracked_run(grid, start, goal, approaches):
        if crash_after is not None and len(calls) == crash_after:
            os._exit(17)  # No exception handlers or connection.close() run.
        results = real_run(grid, start, goal, approaches)
        calls.append((start, goal, results[0].algorithm))
        return results

    ns['run_all_algorithms'] = tracked_run
    return ns, calls, progress


def run(ns, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return ns['benchmark_dataset'](approaches=APPROACHES, **kwargs)


class BenchmarkResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'tiny.map').write_text(
            'type octile\nheight 6\nwidth 8\nmap\n' + '........\n' * 6)
        (self.root / 'tiny.map.scen').write_text(
            'version 1\n'
            '0 tiny.map 8 6 0 0 2 0 2\n'
            '1 tiny.map 8 6 0 0 5 0 5\n'
            '2 tiny.map 8 6 0 0 7 5 9.071067811865476\n')

    def test_abrupt_exit_resume_equivalence_and_cache(self):
        worker = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 '--crash-worker', str(self.root), '2'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(worker.returncode, 17, worker.stderr)
        database = next((self.root / 'cache').glob('*_checkpoint.sqlite3'))
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM searches').fetchone()[0], 2)

        ns, calls, progress = environment(self.root)
        resumed, examples = run(ns)
        self.assertEqual(len(calls), 7)  # Only unsaved searches run.
        self.assertEqual(len(resumed), 9)
        self.assertEqual(len(resumed.drop_duplicates(
            ['Map name', 'Source scenario index', 'Algorithm'])), 9)
        self.assertEqual(progress[0].completed, 3)
        self.assertEqual([r.algorithm for r in next(iter(examples.values()))[3]],
                         [item['label'] for item in APPROACHES])
        ns_clean, clean_calls, _ = environment(self.root)
        ns_clean['CONFIG']['resume_benchmark'] = False
        clean, _ = run(ns_clean, use_cache=False)
        self.assertEqual(len(clean_calls), 9)
        pd.testing.assert_frame_equal(resumed.drop(columns=['Time ms']),
                                      clean.drop(columns=['Time ms']))
        ns_cache, cached_calls, _ = environment(self.root)
        cached, cached_examples = run(ns_cache)
        self.assertEqual(cached_calls, [])
        pd.testing.assert_frame_equal(resumed, cached)
        self.assertEqual([r.path for r in next(iter(cached_examples.values()))[3]],
                         [r.path for r in next(iter(examples.values()))[3]])

    def test_completed_scenarios_progress_and_checkpoint_without_final_cache(self):
        ns, _, _ = environment(self.root)
        real_run = ns['run_all_algorithms']
        count = 0

        def interrupt(grid, start, goal, approaches):
            nonlocal count
            if count == 4:
                raise KeyboardInterrupt()
            count += 1
            return real_run(grid, start, goal, approaches)

        ns['run_all_algorithms'] = interrupt
        with self.assertRaises(KeyboardInterrupt):
            run(ns, use_cache=False)
        ns, calls, progress = environment(self.root)
        frame, _ = run(ns, use_cache=False)
        self.assertEqual(len(calls), 5)
        self.assertEqual(progress[0].initial, 1)
        self.assertEqual(progress[0].completed, 3)
        ns, calls, progress = environment(self.root)
        restored, examples = run(ns, use_cache=False)
        self.assertEqual(calls, [])
        self.assertEqual(progress[0].initial, 3)
        self.assertEqual(len(next(iter(examples.values()))[3]), 3)
        pd.testing.assert_frame_equal(frame, restored)

    def test_new_settings_and_forced_restart(self):
        ns, _, _ = environment(self.root)
        run(ns)
        ns, calls, _ = environment(self.root)
        run(ns, force_recompute=True)
        self.assertEqual(len(calls), 9)
        ns, calls, _ = environment(self.root)
        ns['CONFIG']['scenarios_per_map'] = 2
        frame, _ = run(ns)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(frame), 6)
        self.assertEqual(len(list((self.root / 'cache').glob('*_checkpoint.sqlite3'))), 2)

    def test_transaction_rolls_back_uncommitted_result(self):
        ns, _, _ = environment(self.root)
        database = self.root / 'transaction.sqlite3'
        checkpoint = ns['BenchmarkCheckpoint'](database, {'test': True})
        checkpoint.connection.execute('INSERT INTO searches VALUES (?, ?, ?, ?, ?)',
                                      ('tiny.map', 0, 'Standard JPS', '{}', None))
        checkpoint.close()  # Uncommitted transaction is rolled back.
        checkpoint = ns['BenchmarkCheckpoint'](database, {'test': True})
        self.assertEqual(checkpoint.rows(), [])
        checkpoint.close()
        with self.assertRaises(ValueError):
            ns['BenchmarkCheckpoint'](database, {'test': False})

    def test_turn_penalty_has_separate_checkpoint_and_saved_metadata(self):
        ns, _, _ = environment(self.root)
        run(ns)
        smooth_approaches = deepcopy(APPROACHES)
        smooth_approaches[-1]['parameters']['turn_count_weight'] = 5.0
        ns, calls, _ = environment(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            frame, _ = ns['benchmark_dataset'](approaches=smooth_approaches)
        self.assertEqual(len(calls), 9)
        selected = frame[frame['Algorithm'] == 'Optimized FWSJPS']
        self.assertTrue((selected['Turn count weight'] == 5.0).all())
        self.assertTrue((selected['Search turn count'] == selected['Turns']).all())
        self.assertEqual(len(list((self.root / 'cache').glob('*_checkpoint.sqlite3'))), 2)
        ns, calls, _ = environment(self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            cached, _ = ns['benchmark_dataset'](approaches=smooth_approaches)
        self.assertEqual(calls, [])
        pd.testing.assert_frame_equal(frame, cached)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--crash-worker':
        ns, _, _ = environment(Path(sys.argv[2]), crash_after=int(sys.argv[3]))
        run(ns)
    else:
        unittest.main()
