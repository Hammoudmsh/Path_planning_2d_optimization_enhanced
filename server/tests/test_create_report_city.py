"""City grouping must avoid treating map variants as independent replicates."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import create_report_city as report


class CityReportTest(unittest.TestCase):
    def pairs(self):
        rows=[]
        for city in ['CityA','CityB','CityC']:
            for variant in range(3):
                for scenario in range(2):
                    row={'map':f'{city}_{variant}_256.map','scenario':scenario,'Both successful':True,
                         'Map size':'256 x 256','Range':report.RANGES[0]}
                    for metric in report.METRICS:
                        row[metric+'_jps']=10.;row[metric+'_opt']=9.
                    rows.append(row)
        return pd.DataFrame(rows)

    def test_groups_and_statistical_units(self):
        pairs,mapping=report.assign_city_groups(self.pairs())
        self.assertEqual(mapping.City.nunique(),3)
        stats,maps,cities=report.city_statistical_comparisons(pairs)
        overall=stats[stats.Grouping=='Overall']
        self.assertTrue((overall['N units']==3).all())
        self.assertTrue((overall.Maps==9).all())
        self.assertTrue((overall['p-value']==.25).all())
        self.assertTrue((overall['Paired sign effect']==-1).all())
        self.assertEqual(set(stats.Unit),{'City median of map differences (primary)'})
        # Repeating every scenario within a map cannot increase inferential N.
        more=pd.concat([pairs,pairs[pairs['map']=='CityA_0_256.map']]*3,ignore_index=True)
        again,_,_=report.city_statistical_comparisons(more)
        np.testing.assert_allclose(stats['p-value'],again['p-value'],equal_nan=True)

    def test_explicit_mapping_and_strict_names(self):
        pairs=self.pairs();pairs.loc[0,'map']='unknown.map'
        with self.assertRaisesRegex(ValueError,'Cannot infer'):
            report.assign_city_groups(pairs)
        names=pairs['map'].unique()
        assigned,table=report.assign_city_groups(pairs,{name:'Group' for name in names})
        self.assertEqual(assigned.City.nunique(),1)
        with self.assertRaisesRegex(ValueError,'Missing'):
            report.assign_city_groups(pairs,{'unknown.map':'Group'})

    def test_median_of_maps_not_pooled_scenarios(self):
        pairs=self.pairs()
        pairs=pairs[pairs['map'].str.startswith('CityA')].copy()
        for i,variant in enumerate([0,1,2]):
            pairs.loc[pairs['map']==f'CityA_{variant}_256.map','Time ms_opt']=[0,12,13][i]
        pairs=pd.concat([pairs,pairs[pairs['map']=='CityA_0_256.map'].loc[lambda d:d.index.repeat(100)]])
        pairs,_=report.assign_city_groups(pairs)
        stats,_,cities=report.city_statistical_comparisons(pairs)
        row=cities[(cities.Grouping=='Overall')&(cities.Metric=='Runtime (ms)')].iloc[0]
        self.assertEqual(row['Median of map differences'],2.)


if __name__=='__main__':unittest.main()
