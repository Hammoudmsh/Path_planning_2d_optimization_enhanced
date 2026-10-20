"""Small reproducible validation/timing sample; does not run the notebook pipeline."""
import json
import platform
import statistics
import sys
from pathlib import Path

from _load_search import load_search


def main():
    # Match the original runner's recursion setting for the unmodified baseline.
    sys.setrecursionlimit(30000)
    ns = load_search()
    root = Path(__file__).resolve().parents[2]
    algorithms = {
        'Standard JPS': lambda g, s, t: ns['run_jps'](g, s, t),
        'Original FWSJPS': lambda g, s, t: ns['run_sa_jps'](
            g, s, t, heuristic_weight=1.15, turn_weight=0.35),
        'Optimized FWSJPS': ns['run_optimized_fwsjps'],
    }
    rows = []
    for dataset in ('street', 'random', 'maze', 'dao', 'wc3maps512', 'bg512'):
        scenario = sorted((root / 'datasets' / dataset / 'scenarios').glob('*.scen'))[0]
        records = scenario.read_text().splitlines()[1:]
        grid = ns['read_map'](root / 'datasets' / dataset / 'maps' / scenario.name[:-5])
        # Three nontrivial scenario ranks spanning the selected file.
        for index in sorted({len(records) // 4, len(records) // 2, 3 * len(records) // 4}):
            fields = records[index].split()
            sx, sy, gx, gy = map(int, fields[4:8])
            start, goal = (sx, sy), (gx, gy)
            results = {name: run(grid, start, goal) for name, run in algorithms.items()}
            old, new = results['Original FWSJPS'], results['Optimized FWSJPS']
            assert old.success == new.success and old.path == new.path
            assert old.search_objective == new.search_objective
            for result in results.values():
                assert result.success
                assert ns['validate_path'](grid, result.path, start, goal)[0]
            # Warmup above; rotate timing order. Every call gets fresh query caches.
            timings = {name: [] for name in algorithms}
            end_to_end = {name: [] for name in algorithms}
            names = list(algorithms)
            for repeat in range(3):
                for name in names[repeat:] + names[:repeat]:
                    started = ns['time'].perf_counter()
                    result = algorithms[name](grid, start, goal)
                    end_to_end[name].append((ns['time'].perf_counter() - started) * 1000)
                    timings[name].append(result.runtime_ms)
            for name, result in results.items():
                rows.append({
                    'dataset': dataset, 'map': scenario.name[:-5], 'scenario': index,
                    'algorithm': name,
                    'median_search_ms': statistics.median(timings[name]),
                    'median_end_to_end_ms': statistics.median(end_to_end[name]),
                    'path_length': ns['path_length'](result.path),
                    'turn_severity': ns['path_turn_metrics'](result.path)[1] / 45.0,
                    'expanded_nodes': result.expanded_nodes,
                })
        print(f'{dataset}: paths match; three scenarios timed', flush=True)
    totals = {name: sum(r['median_search_ms'] for r in rows if r['algorithm'] == name)
              for name in algorithms}
    report = {
        'python': platform.python_version(), 'platform': platform.platform(),
        'repetitions': 3, 'maps_per_dataset': 1, 'scenarios_per_map': 3,
        'turn_weight': 0.35, 'heuristic_weight': 1.15,
        'notes': 'Small convenience sample, not a full dataset benchmark. Search time '
                 'includes query cache setup but follows existing exclusion of path reconstruction. '
                 'End-to-end time also includes reconstruction and return cleanup.',
        'sum_of_median_search_ms': totals,
        'speedup_over_original': totals['Original FWSJPS'] / totals['Optimized FWSJPS'],
        'speedup_over_jps': totals['Standard JPS'] / totals['Optimized FWSJPS'],
        'rows': rows,
    }
    output = root / 'server' / 'tests' / 'optimized_fwsjps_sample_results.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
