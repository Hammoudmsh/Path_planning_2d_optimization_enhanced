"""Sensitivity settings remain isolated and resume correctly with concurrent workers."""
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark_config as config


class SensitivityTest(unittest.TestCase):
    def test_grid_and_existing_profiles(self):
        baseline=config.get_config('street','street_all')
        study=config.get_config('street','sensitivity')
        approaches=study['benchmark_approaches']
        self.assertEqual(len(approaches),13)
        self.assertEqual(len({a['label'] for a in approaches}),13)
        self.assertEqual({a['algorithm'] for a in approaches},{'Optimized Fast Weighted SA-JPS'})
        for a in approaches:
            self.assertLessEqual(sum(a['parameters'][key]!=value for key,value in config.SENSITIVITY_BASELINE.items()),1)
        self.assertEqual(study['scenarios_per_map'],50)
        self.assertIsNone(study['map_limit'])
        self.assertTrue(study['resume_benchmark'])
        self.assertFalse(study['force_recompute_benchmark'])
        self.assertNotEqual(study['results'],baseline['results'])
        self.assertEqual(len(baseline['benchmark_approaches']),2)
        self.assertEqual(baseline['benchmark_approaches'][1]['parameters'],config.SENSITIVITY_BASELINE)
        with self.assertRaisesRegex(ValueError,'already sweep'):
            config.apply_equation_overrides(study,beta_weight=1)
        self.assertIsNone(config.get_config('street','sensitivity_all')['scenarios_per_map'])

    def test_parallel_cli_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);server=root/'server';server.mkdir()
            for name in ['benchmark_config.py','parallel_benchmark.py','Astar_JPS_SA_JPS_config_profiles_ready.py']:
                shutil.copy2(Path(__file__).resolve().parents[1]/name,server/name)
            maps=root/'datasets/street/maps';scenarios=root/'datasets/street/scenarios'
            maps.mkdir(parents=True);scenarios.mkdir(parents=True)
            for i in range(2):
                name=f'tiny{i}.map'
                (maps/name).write_text('type octile\nheight 6\nwidth 8\nmap\n'+'........\n'*6)
                (scenarios/(name+'.scen')).write_text('version 1\n'+f'0 {name} 8 6 0 0 7 5 9.071067811865476\n')
            command=[sys.executable,str(server/'Astar_JPS_SA_JPS_config_profiles_ready.py'),'street','sensitivity_test','--workers','2']
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=120)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            result=root/'results_new_ablation/street/sensitivity_test'
            database=next((result/'cache').glob('*checkpoint.sqlite3'))
            with closing(sqlite3.connect(database)) as db:
                before=db.execute('SELECT * FROM searches ORDER BY map_name,scenario,algorithm').fetchall()
            self.assertEqual(len(before),26)
            expected={a['label']:a['parameters'] for a in config.sensitivity_approaches()}
            self.assertEqual({r[2] for r in before},set(expected))
            for row in before:
                metrics=json.loads(row[3]);parameters=expected[row[2]]
                self.assertEqual(metrics['Turn penalty weight'],parameters['turn_weight'])
                self.assertEqual(metrics['Turn count weight'],parameters['turn_count_weight'])
                self.assertEqual(metrics['Heuristic weight'],parameters['heuristic_weight'])
            command[-1]='3'
            resumed=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=120)
            self.assertEqual(resumed.returncode,0,resumed.stdout+resumed.stderr)
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(before,db.execute('SELECT * FROM searches ORDER BY map_name,scenario,algorithm').fetchall())
            self.assertTrue(list(result.glob('summary*.csv')))
            self.assertFalse((root/'results_new_ablation/street/street_all').exists())


if __name__=='__main__': unittest.main()
