import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _load_search import SOURCE, load_search


class MapSizeTest(unittest.TestCase):
    def test_header_filter_is_applied_before_map_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Headers suffice: discovery must not load map arrays.
            for name, width, height in [('a_1024.map', 256, 256),
                                        ('b.map', 1024, 512),
                                        ('c.map', 1024, 1024),
                                        ('d.map', 1024, 1024),
                                        ('e.map', 1024, 1024)]:
                (root / name).write_text(f'type octile\nheight {height}\nwidth {width}\nmap\n')
            ns = load_search(extra_functions={'discover_selected_maps'})
            ns.update(MAPS_DIR=root, CONFIG={'map_sizes': [1024], 'map_limit': 2})
            self.assertEqual([p.name for p in ns['discover_selected_maps']()], ['c.map', 'd.map'])
            ns['CONFIG']['map_sizes'] = None
            self.assertEqual([p.name for p in ns['discover_selected_maps']()], ['a_1024.map', 'b.map'])

    def test_cli_preserves_test_workload_and_isolates_size_outputs(self):
        result = subprocess.run([
            sys.executable, str(SOURCE), 'street', 'test', '--map-size', '1024', '--map-limit', '100',
            '--lambda', '0.35', '--beta', '5', '--w', '1.15', '--dry-run',
        ], cwd=SOURCE.parent, capture_output=True, text=True, check=True)
        config = json.loads(result.stdout)
        self.assertEqual(config['map_sizes'], [1024])
        self.assertEqual((config['map_limit'], config['scenarios_per_map']), (100, 1))
        self.assertEqual(Path(config['results']).parent.name, 'maps_1024x1024')
        self.assertEqual(Path(config['results']).name, 'limit_100')


if __name__ == '__main__':
    unittest.main()
