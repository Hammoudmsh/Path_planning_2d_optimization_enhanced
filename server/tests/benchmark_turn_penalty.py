"""Exploratory street sample for turn-count penalties; no map preprocessing."""
import gc
import json
import sys
from pathlib import Path

from _load_search import load_search


def main():
    sys.setrecursionlimit(30000)
    ns = load_search()
    root = Path(__file__).resolve().parents[2]
    rows = []
    for map_name in ('Berlin_0_256.map', 'Boston_0_256.map', 'Denver_0_256.map'):
        grid = ns['read_map'](root / 'datasets/street/maps' / map_name)
        records = (root / 'datasets/street/scenarios' / (map_name + '.scen')).read_text().splitlines()[1:]
        # Exclude trivial near-zero distances; span the remaining scenario ranks.
        indices = ns['np'].linspace(len(records) // 10, len(records) - 1, 10, dtype=int)
        for index in indices:
            fields = records[index].split()
            sx, sy, gx, gy = map(int, fields[4:8])
            start, goal = (sx, sy), (gx, gy)
            for beta in (0.0, 1.0, 2.0, 5.0):
                gc.collect()
                result = ns['run_optimized_fwsjps'](
                    grid, start, goal, turn_count_weight=beta)
                gc.collect()
                assert result.success
                assert ns['validate_path'](grid, result.path, start, goal)[0]
                turns, angle, maximum = ns['path_turn_metrics'](result.path)
                length = ns['path_length'](result.path)
                assert abs(result.search_objective - (length + 0.35 * angle / 45 + beta * turns)) < 1e-7
                rows.append(dict(map=map_name, scenario=int(index), turn_count_weight=beta,
                                 turns=turns, total_angle=angle, maximum_angle=maximum,
                                 path_length=length, runtime_ms=result.runtime_ms,
                                 expanded_nodes=result.expanded_nodes))
        print(f'{map_name}: 10 scenarios checked at four penalties', flush=True)
    baseline = [row for row in rows if row['turn_count_weight'] == 0]
    summary = []
    for beta in (0.0, 1.0, 2.0, 5.0):
        group = [row for row in rows if row['turn_count_weight'] == beta]
        aggregate = {key: sum(row[key] for row in group) / len(group)
                     for key in ('turns', 'total_angle', 'path_length', 'runtime_ms', 'expanded_nodes')}
        aggregate.update(turn_count_weight=beta, queries=len(group),
                         fewer_turns=sum(a['turns'] < b['turns'] for a, b in zip(group, baseline)),
                         same_turns=sum(a['turns'] == b['turns'] for a, b in zip(group, baseline)),
                         more_turns=sum(a['turns'] > b['turns'] for a, b in zip(group, baseline)))
        summary.append(aggregate)
    output = root / 'server/tests/turn_penalty_sample_results.json'
    output.write_text(json.dumps({
        'note': 'Exploratory 30-query sample, not a full benchmark or independent validation. '
                'Single timings per query; no speed claim is based on this sweep.',
        'turn_weight': 0.35, 'heuristic_weight': 1.15, 'summary': summary, 'rows': rows,
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
