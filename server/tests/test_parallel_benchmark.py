"""Concurrent real-search execution, saved-row preservation and hard-kill locks."""
from contextlib import closing
import json
import contextlib
import io
import os
from pathlib import Path
import sqlite3
import zipfile
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from test_benchmark_resume import environment, run
from parallel_benchmark import FileLock, backup_results, ProgressReporter


class ParallelBenchmarkTest(unittest.TestCase):
    def test_eight_workers_preserve_partial_checkpoint_and_do_not_repeat_searches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tiny.map').write_text('type octile\nheight 6\nwidth 8\nmap\n' + '........\n' * 6)
            (root / 'tiny.map.scen').write_text('version 1\n' + ''.join(
                f'{i} tiny.map 8 6 0 0 {i % 7 + 1} 5 9\n' for i in range(15)))
            crash = subprocess.run([sys.executable, str(Path(__file__).with_name('test_benchmark_resume.py')),
                                    '--crash-worker', str(root), '2'], capture_output=True, timeout=30)
            self.assertEqual(crash.returncode, 17, crash.stderr)
            database = next((root / 'cache').glob('*.sqlite3'))
            with closing(sqlite3.connect(database)) as db:
                before = db.execute('SELECT * FROM searches ORDER BY rowid').fetchall()
            # Keep backup within the test temporary directory.
            result_dir = root / 'results'
            result_dir.mkdir()
            backup = backup_results(dict(results=result_dir, benchmark_cache_dir=root / 'cache'))
            with closing(sqlite3.connect(backup / 'cache' / database.name)) as db:
                self.assertEqual(before, db.execute('SELECT * FROM searches ORDER BY rowid').fetchall())
            children = [subprocess.Popen([sys.executable, __file__, '--worker', str(root), str(i)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for i in range(8)]
            try:
                for child in children:
                    _, error = child.communicate(timeout=90)
                    self.assertEqual(child.returncode, 0, error.decode())
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill(); child.wait()
            ns, final_calls, _ = environment(root)
            frame, _ = run(ns)
            calls = sum(json.loads((root / f'calls_{i}.json').read_text()) for i in range(8))
            self.assertEqual(calls + len(final_calls), 43)
            self.assertEqual(len(frame), 45)
            with closing(sqlite3.connect(database)) as db:
                after = db.execute('SELECT * FROM searches ORDER BY rowid LIMIT 2').fetchall()
                self.assertEqual(before, after)
                self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_cli_launcher_and_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            server = root / 'server'
            server.mkdir()
            source = Path(__file__).resolve().parents[1]
            for name in ('Astar_JPS_SA_JPS_config_profiles_ready.py', 'benchmark_config.py', 'parallel_benchmark.py'):
                shutil.copy2(source / name, server / name)
            maps = root / 'datasets/street/maps'
            scenarios = root / 'datasets/street/scenarios'
            maps.mkdir(parents=True); scenarios.mkdir(parents=True)
            (maps / 'tiny.map').write_text('type octile\nheight 6\nwidth 8\nmap\n' + '........\n' * 6)
            (scenarios / 'tiny.map.scen').write_text('version 1\n0 tiny.map 8 6 0 0 7 5 9\n')
            config = root / 'experiment.json'
            config.write_text(json.dumps(dict(base_profile='street_all', output_root='parallel_test', overrides=dict(
                download_data=False, map_sizes=None, map_limit=1, create_zip_archive=True))))
            command = [sys.executable, str(server / 'Astar_JPS_SA_JPS_config_profiles_ready.py'),
                       'street', 'street_all', '--config-json', str(config), '--workers', '8']
            result = subprocess.run(command, capture_output=True, text=True, timeout=120, cwd=root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('Maps 1/1 | Scenarios 1/1 | Algorithm runs 2/2 (100.0%)', result.stdout)
            output = root / 'parallel_test/street/street_all'
            self.assertTrue(list(output.glob('per_run_results*.csv')))
            archive = next(root.glob('*.zip'))
            with zipfile.ZipFile(archive) as zipped:
                self.assertTrue(any('per_run_results' in name for name in zipped.namelist()))
                self.assertFalse(any(name.endswith('.lock') for name in zipped.namelist()))
            self.assertEqual(len(list((output / 'worker_logs').rglob('worker_*.log'))), 8)
            with closing(sqlite3.connect(next((output / 'cache').glob('*.sqlite3')))) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM searches').fetchone()[0], 2)

    def test_progress_counts_partial_resumed_and_incremental_results(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.sqlite3'
            progress = ProgressReporter(path, {'map': {1, 2}}, ['JPS', 'optimized'])
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE searches (map_name TEXT, scenario INTEGER, algorithm TEXT)')
                db.executemany('INSERT INTO searches VALUES (?, ?, ?)',
                               [('map', 1, 'JPS'), ('map', 1, 'optimized'), ('map', 2, 'JPS'),
                                ('other', 1, 'JPS'), ('map', 2, 'unselected')])
                db.commit()
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    progress.report(8, 8)
                    progress.report(8, 8)
                self.assertEqual(out.getvalue().count('Maps 0/1 | Scenarios 1/2 | Algorithm runs 3/4 (75.0%)'), 2)
                db.execute("INSERT INTO searches VALUES ('map', 2, 'optimized')")
                db.commit()
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    progress.report(0, 8)
                self.assertIn('Maps 1/1 | Scenarios 2/2 | Algorithm runs 4/4 (100.0%)', out.getvalue())

    def test_lock_released_after_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child = subprocess.Popen([sys.executable, __file__, '--hold-lock', str(root)])
            try:
                deadline = time.monotonic() + 20
                while not (root / 'ready').exists():
                    if time.monotonic() > deadline:
                        self.fail('Lock holder did not start')
                    time.sleep(.05)
                lock = FileLock(root / 'claim.lock')
                self.assertFalse(lock.acquire())
                child.kill(); child.wait()
                self.assertTrue(lock.acquire())
                lock.release()
            finally:
                if child.poll() is None:
                    child.kill(); child.wait()


if __name__ == '__main__':
    if '--worker' in sys.argv:
        root = Path(sys.argv[2])
        ns, calls, _ = environment(root)
        ns['CONFIG']['_parallel_worker'] = True
        real = ns['run_all_algorithms']
        def slow(*args, **kwargs):
            time.sleep(.05)
            return real(*args, **kwargs)
        ns['run_all_algorithms'] = slow
        run(ns)
        (root / f'calls_{sys.argv[3]}.json').write_text(json.dumps(len(calls)))
    elif '--hold-lock' in sys.argv:
        root = Path(sys.argv[2])
        lock = FileLock(root / 'claim.lock')
        assert lock.acquire()
        (root / 'ready').touch()
        time.sleep(120)
    else:
        unittest.main()
