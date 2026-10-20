"""Report calculations must preserve pairing, range boundaries, and zero denominators."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import create_report as report
import numpy as np
import pandas as pd


class ReportTest(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i, reference in enumerate([0, 99, 100, 249, 250, 499, 500, 999, 1000]):
            for algorithm in report.ALGORITHMS:
                row={m:float(i+1) for m in report.METRICS}
                row.update({'map':'tiny.map','scenario':i,'algorithm':algorithm,
                            'Map width':256,'Map height':256,'Reference path length':reference,
                            'Execution workers':'8','Success':True})
                rows.append(row)
        rows[0]['Path length']=0;rows[1]['Path length']=0
        rows[2]['Execution workers']='unknown'
        rows[3]['Success']=False
        return pd.DataFrame(rows)

    def test_boundaries_failures_and_pooled_results(self):
        frame=self.fixture()
        pairs,details=report.match_pairs(frame)
        self.assertEqual(len(pairs),9)
        self.assertEqual(list(pairs.groupby('Range',observed=False).size()),[2,2,2,2,1])
        self.assertEqual(details['both_successful_pairs'],8)
        tables=report.build_tables(frame,pairs)
        rows=tables['summary_numeric']
        lengths=rows[(rows.Grouping=='Overall') & (rows.Metric=='Path length')]
        self.assertTrue((lengths.N==8).all())
        changes=tables['relative_changes']
        length=changes[(changes.Grouping=='Overall') & (changes.Metric=='Path length')].iloc[0]
        self.assertEqual(length['Positive baseline pairs'],7)
        self.assertEqual(length['Zero or negative baseline pairs excluded'],1)

    def test_unmatched_mismatch_and_relative_definitions(self):
        frame=self.fixture().iloc[:6].copy()
        frame['Success']=True
        frame.loc[frame.algorithm==report.ALGORITHMS[0],'Time ms']=[10,100,200]
        frame.loc[frame.algorithm==report.ALGORITHMS[1],'Time ms']=[20,50,400]
        pairs,_=report.match_pairs(frame)
        changes=report.build_tables(frame,pairs)['relative_changes']
        runtime=changes[(changes.Grouping=='Overall') & (changes.Metric=='Runtime (ms)')].iloc[0]
        self.assertEqual(runtime['Difference between medians (%)'],-50)
        self.assertEqual(runtime['Per-scenario change median (%)'],100)
        frame.loc[1,'Reference path length']=999
        pairs,details=report.match_pairs(frame.iloc[:-1])
        self.assertEqual(details['matched_before_filters'],2)
        self.assertEqual(details['invalid_pair_metadata_excluded'],1)
        self.assertEqual(len(pairs),1)

    def test_statistics_known_values_and_edge_cases(self):
        result = report.difference_statistics([-1, -2, -3, -4, -5, -6], primary=True)
        self.assertEqual(result['p-value'], 0.03125)
        self.assertEqual(result['Paired sign effect'], -1)
        self.assertEqual(result['Rank-biserial effect'], -1)
        flipped = report.difference_statistics([1, 2, 3, 4, 5, 6], primary=True)
        self.assertEqual(flipped['p-value'], result['p-value'])
        self.assertEqual(flipped['Rank-biserial effect'], 1)
        tied = report.difference_statistics([0, 0, 0], primary=True)
        self.assertEqual(tied['p-value'], 1)
        self.assertEqual(tied['Rank-biserial effect'], 0)
        self.assertTrue(np.isnan(report.difference_statistics([1], primary=True)['p-value']))
        mixed = report.difference_statistics([-1, 1, 0], primary=True)
        self.assertEqual(mixed['p-value'], 1)
        self.assertEqual(mixed['Rank-biserial effect'], 0)
        np.testing.assert_allclose(report.holm_adjust([.01, .04, .03, np.nan]), [.03, .06, .06, np.nan])
        exploratory = report.difference_statistics(np.arange(1, 30))
        self.assertLess(exploratory['p-value'], .001)

    def test_statistics_use_maps_as_primary_units(self):
        pairs, _ = report.match_pairs(self.fixture())
        stats = report.statistical_comparisons(pairs)
        primary = stats[(stats.Grouping == 'Overall') & (stats.Unit == 'Map median difference (primary)')]
        self.assertTrue((primary['N units'] == 1).all())
        self.assertEqual(len(primary), len(report.METRICS))
        exploratory = stats[(stats.Grouping == 'Overall') & (stats.Unit == 'Scenario difference (exploratory)')]
        self.assertTrue((exploratory['N units'] == 8).all())

    def test_presentation_helpers_and_path_reconstruction(self):
        import presentation_plots as plots
        np.testing.assert_array_equal(plots.classify_pairs([100,100,100,np.nan], [98,100.5,102,1], rtol=.01), [1,1,1])
        functions=plots.load_search_functions(Path(report.__file__).with_name('Astar_JPS_SA_JPS_config_profiles_ready.py'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'tiny.map').write_text('type octile\nheight 6\nwidth 8\nmap\n'+'........\n'*6)
            grid=functions['read_map'](root/'tiny.map')
            start=(0,0);goal=(7,5)
            first=functions['run_jps'](grid,start,goal)
            second=functions['run_optimized_fwsjps'](grid,start,goal,turn_weight=.35,turn_count_weight=5,heuristic_weight=1.15)
            row={'map':'tiny.map','scenario':0,'Reference path length_jps':10}
            for suffix,result in [('jps',first),('opt',second)]:
                row.update({key+'_'+suffix:value for key,value in result.metrics().items()})
                row.update({key+'_'+suffix:value for key,value in {'Start x':0,'Start y':0,'Goal x':7,'Goal y':5,'Map width':8,'Map height':6,'Execution workers':'8'}.items()})
            row.update({'Turn penalty weight_opt':.35,'Turn count weight_opt':5,'Heuristic weight_opt':1.15})
            data=pd.DataFrame([row])
            figures,records=plots.path_examples(data,root,root,Path(report.__file__).with_name('Astar_JPS_SA_JPS_config_profiles_ready.py'),{},report)
            self.assertEqual(records[0]['status'],'Verified reconstruction')
            self.assertEqual(len(figures),1)
            self.assertTrue((root/'figure_4_path_example_1.png').exists())
            self.assertEqual(plots.select_examples(data)[0][1]['scenario'],0)
            # Large maps need the benchmark's recursion setting for original JPS.
            (root/'long.map').write_text('type octile\nheight 1\nwidth 1024\nmap\n' + '.'*1024 + '\n')
            row['map']='long.map'
            for suffix in ['jps','opt']:
                row.update({'Map width_'+suffix:1024,'Map height_'+suffix:1,'Goal x_'+suffix:1023,'Goal y_'+suffix:0,'Path length_'+suffix:1023,'Turns_'+suffix:0})
            figures,records=plots.path_examples(pd.DataFrame([row]),root,root,Path(report.__file__).with_name('Astar_JPS_SA_JPS_config_profiles_ready.py'),{},report)
            self.assertEqual(records[0]['status'],'Verified reconstruction')


    def test_read_only_snapshot_and_full_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);dbpath=root/'checkpoint.sqlite3'
            db=sqlite3.connect(dbpath)
            db.execute('CREATE TABLE metadata(signature TEXT)')
            db.execute('INSERT INTO metadata VALUES (?)',(json.dumps({'version':6}),))
            db.execute('CREATE TABLE searches(map_name TEXT, scenario INTEGER, algorithm TEXT, metrics TEXT)')
            for row in self.fixture().to_dict('records'):
                # Legacy checkpoints do not record worker count.
                row.pop('Execution workers')
                db.execute('INSERT INTO searches VALUES (?,?,?,?)',(row['map'],row['scenario'],row['algorithm'],json.dumps(row)))
            db.commit();db.close()
            original=dbpath.read_bytes()
            output=report.main(['--checkpoint',str(dbpath),'--output',str(root/'report')])
            self.assertEqual(dbpath.read_bytes(),original)
            self.assertTrue((output/'statistical_comparisons.csv').exists())
            self.assertIn('Statistical comparison', (output/'report.html').read_text())
            self.assertGreaterEqual(len(list(output.glob('*.png'))),7)
            self.assertEqual(len(list(output.glob('*.pdf'))),len(list(output.glob('*.png'))))
            self.assertFalse((output/'table_4_worker_counts.csv').exists())
            self.assertNotIn('worker', (output/'report.html').read_text().lower())
            self.assertFalse(any('worker' in p.name for p in output.glob('*.png')))
            self.assertIn('Relative changes:',(output/'report.html').read_text())
            self.assertEqual(json.loads((output/'manifest.json').read_text())['included_pairs'],9)


if __name__=='__main__':unittest.main()
