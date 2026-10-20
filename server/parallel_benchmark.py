"""Same-host process coordination; OS locks are released even after a hard kill."""
from contextlib import closing
import errno
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime


class FileLock:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        self.handle = handle
        return True

    def release(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class ScenarioLocks:
    """Bound disk usage with 256 lock stripes; collisions defer work safely."""
    def __init__(self, checkpoint):
        self.directory = Path(str(checkpoint) + '.locks')
        self.current = None

    def release(self):
        if self.current is not None:
            self.current.release()
            self.current = None

    def acquire(self, key):
        self.release()
        stripe = hashlib.sha256(repr(key).encode()).digest()[0]
        lock = FileLock(self.directory / f'{stripe:02x}.lock')
        if not lock.acquire():
            return False
        self.current = lock
        return True


def backup_results(config):
    """Online SQLite backup preserves committed rows, including any journal data."""
    results = Path(config['results'])
    destination = results.parent / (results.name + '_backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True)
    for source in Path(config['benchmark_cache_dir']).glob('*_checkpoint.sqlite3'):
        target = destination / 'cache' / source.name
        target.parent.mkdir(exist_ok=True)
        with closing(sqlite3.connect(source, timeout=120)) as original:
            with closing(sqlite3.connect(target)) as saved:
                original.backup(saved)
                count = saved.execute('SELECT count(*) FROM searches').fetchone()[0]
        print(f'Backed up {count:,} saved searches: {target}', flush=True)
    for pattern in ('*.csv', '*.json'):
        for source in results.rglob(pattern):
            target = destination / source.relative_to(results)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f'Backup: {destination}', flush=True)
    return destination


class ProgressReporter:
    """Count committed searches incrementally without reading metrics or taking write locks."""
    def __init__(self, checkpoint, workload, labels):
        self.checkpoint = Path(checkpoint)
        self.workload = workload
        self.labels = set(labels)
        self.done = {}
        self.last_rowid = 0

    def report(self, active, workers):
        try:
            if self.checkpoint.exists():
                with closing(sqlite3.connect(self.checkpoint.resolve().as_uri() + '?mode=ro',
                                            uri=True, timeout=0.2)) as db:
                    records = db.execute('SELECT rowid, map_name, scenario, algorithm FROM searches '
                                         'WHERE rowid > ? ORDER BY rowid', (self.last_rowid,)).fetchall()
                for rowid, name, scenario, algorithm in records:
                    if scenario in self.workload.get(name, ()) and algorithm in self.labels:
                        self.done.setdefault((name, scenario), set()).add(algorithm)
                    self.last_rowid = rowid
        except sqlite3.OperationalError:
            print('Progress: checkpoint busy or initializing; retrying next update.', flush=True)
            return
        total_scenarios = sum(len(ids) for ids in self.workload.values())
        complete = {key for key, labels in self.done.items() if labels == self.labels}
        maps_done = sum(all((name, scenario) in complete for scenario in ids)
                        for name, ids in self.workload.items())
        searches = sum(len(labels) for labels in self.done.values())
        total_searches = total_scenarios * len(self.labels)
        percent = 100 * searches / total_searches if total_searches else 100
        print(f'Progress | Maps {maps_done:,}/{len(self.workload):,} | '
              f'Scenarios {len(complete):,}/{total_scenarios:,} | '
              f'Algorithm runs {searches:,}/{total_searches:,} ({percent:.1f}%) | '
              f'Workers {active}/{workers}', flush=True)


def launch_workers(config, count, prepare, finalize, progress_factory=None):
    if not config.get('resume_benchmark') or config.get('force_recompute_benchmark'):
        raise ValueError('Parallel execution requires resume_benchmark=True and force_recompute_benchmark=False.')
    backup_results(config)
    prepare(dict(config, run_main_benchmark=False, run_smoke_test=False))
    progress = progress_factory() if progress_factory else None
    if progress:
        progress.report(0, count)
    log_dir = Path(config['results']) / 'worker_logs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_dir.mkdir(parents=True)
    children = []
    logs = []
    try:
        for index in range(1, count + 1):
            env = dict(os.environ, JPS_PARALLEL_WORKER=str(index), JPS_PARALLEL_COUNT=str(count))
            # Avoid nested BLAS thread pools in each search process.
            env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            log = open(log_dir / f'worker_{index}.log', 'w', encoding='utf-8')
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, '-u', *sys.argv], env=env,
                                             stdout=log, stderr=subprocess.STDOUT))
        print(f'Started {count} workers. Logs: {log_dir}', flush=True)
        last_status = 0.0
        while any(child.poll() is None for child in children):
            failed = [child for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'A worker failed. Check {log_dir}; saved results are intact. Rerun the same command.')
            if time.monotonic() - last_status > 10:
                active = sum(child.poll() is None for child in children)
                if progress:
                    progress.report(active, count)
                else:
                    print(f'Workers active: {active}/{count}', flush=True)
                last_status = time.monotonic()
            time.sleep(1)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError(f'A worker failed. Check {log_dir}; rerun the same command to resume.')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            child.wait()
        for log in logs:
            log.close()
    print('Workers finished. Checking remaining scenarios and writing combined reports.', flush=True)
    result = finalize(config)
    if progress:
        progress.report(0, count)
    return result
