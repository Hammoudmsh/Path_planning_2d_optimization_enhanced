"""Presentation figures for create_report.py; no checkpoint writes."""
import ast
import __future__
import heapq
import json
import math
from pathlib import Path
import time
import sys
import gc
from typing import Tuple, Dict, List, Optional, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest


def paired_changes(part, metric):
    x, y = part[metric+'_jps'], part[metric+'_opt']
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
    return 100*(y[valid]-x[valid])/x[valid]


def classify_pairs(x, y, rtol=0, atol=0):
    x, y = np.asarray(x), np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    tie = np.abs(y-x) <= atol + rtol*np.abs(x)
    return np.array([np.sum((y<x) & ~tie), np.sum(tie), np.sum((y>x) & ~tie)])


def select_examples(data):
    data = data.sort_values(['map', 'scenario']).copy()
    x, y = data['Path length_jps'], data['Path length_opt']
    valid = (x > 0) & np.isfinite(x) & np.isfinite(y)
    data = data[valid].copy()
    if data.empty: return []
    data['extra'] = 100*(data['Path length_opt']-data['Path length_jps'])/data['Path length_jps']
    data['saved'] = data['Turns_jps']-data['Turns_opt']
    cols = ['Reference path length_jps', 'extra', 'saved']
    def central(pool):
        values = pool[cols]
        scale = (values.quantile(.75)-values.quantile(.25)).replace(0, 1)
        distance = ((values-values.median()).abs()/scale).sum(axis=1)
        return pool.loc[distance.idxmin()]
    representative = central(data)
    selected = [('Representative', representative)]
    pool = data[(data.extra > 1e-7) & (data.saved > 0)]
    pool = pool.drop(index=representative.name, errors='ignore')
    if not pool.empty: selected.append(('Longer but smoother', central(pool)))
    return selected


def load_search_functions(source):
    """Load only whitelisted definitions, never the benchmark's executable top level."""
    names = {'sign','movement_direction','octile_distance','segment_distance','in_bounds','walkable',
             'can_step','expand_jump_path','path_length','path_turn_metrics','count_waypoints',
             'SearchResult','has_forced_neighbour','pruned_directions','build_jump_function',
             'reconstruct_jps_state_path','run_jps','direction_angle_degrees','turn_severity_45',
             'reconstruct_sa_jps_path','build_optimized_jump_function','run_optimized_fwsjps',
             'validate_path','read_map'}
    constants = {'DIRECTIONS_8','DIRECTION_TO_ID','START_DIRECTION_ID','OPTIMIZED_TURN_SEVERITY'}
    nodes=[]
    for node in ast.parse(source.read_text(encoding='utf-8-sig')).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names: nodes.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets = node.targets if isinstance(node,ast.Assign) else [node.target]
            if any(isinstance(t,ast.Name) and t.id in constants for t in targets): nodes.append(node)
    namespace=dict(Tuple=Tuple, Dict=Dict, List=List, Optional=Optional, Sequence=Sequence, Point=Tuple[int,int], Direction=Tuple[int,int], np=np,math=math,heapq=heapq,time=time,Path=Path,dataclass=dataclass,lru_cache=lru_cache,SQRT2=math.sqrt(2))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
    return namespace


def path_examples(data, output, maps_dir, source, signature, api):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 30000))
    selected = select_examples(data)
    records=[]; figures=[]
    functions=None
    for index,(kind,row) in enumerate(selected,1):
        record={'selection':kind,'map':row['map'],'scenario':int(row.scenario)}
        try:
            coords=[row.get(c+'_jps',np.nan) for c in ['Start x','Start y','Goal x','Goal y']]
            if not np.isfinite(coords).all(): raise ValueError('Start/goal coordinates unavailable')
            if any(row.get(c+'_opt') != row.get(c+'_jps') for c in ['Start x','Start y','Goal x','Goal y']): raise ValueError('Paired coordinates disagree')
            matches=list(maps_dir.rglob(row['map']))
            if len(matches)!=1: raise ValueError(f'Expected one map under {maps_dir}; found {len(matches)}')
            if functions is None: functions=load_search_functions(source)
            grid=functions['read_map'](matches[0]); start=tuple(map(int,coords[:2]));goal=tuple(map(int,coords[2:]))
            if grid.shape != (int(row['Map height_jps']),int(row['Map width_jps'])): raise ValueError('Map dimensions differ from checkpoint')
            # Saved weights identify this experiment; never assume the current profile's weights.
            params={name:float(row[column+'_opt']) for name,column in [
                ('turn_weight','Turn penalty weight'),('turn_count_weight','Turn count weight'),('heuristic_weight','Heuristic weight')]}
            if not all(np.isfinite(v) for v in params.values()): raise ValueError('Saved optimized weights unavailable')
            jps_params={'heuristic_weight':1.0,'smooth_tie_break':False}
            for approach in signature.get('approaches',[]):
                if isinstance(approach,dict) and approach.get('algorithm')=='Standard JPS': jps_params.update(approach.get('parameters',{}))
            results=[functions['run_jps'](grid,start,goal,**jps_params),functions['run_optimized_fwsjps'](grid,start,goal,**params)]
            for suffix,result in zip(['jps','opt'],results):
                valid,message=functions['validate_path'](grid,result.path,start,goal)
                metrics=result.metrics()
                if not result.success or not valid: raise ValueError('Reconstructed path invalid: '+message)
                if not np.isclose(metrics['Path length'],row['Path length_'+suffix],rtol=1e-9,atol=1e-7) or metrics['Turns']!=row['Turns_'+suffix]:
                    raise ValueError('Reconstruction differs from saved length/turn count; no misleading overlay produced')
            fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
            for ax,result,suffix,color,label in zip(axes,results,['jps','opt'],api.COLORS,api.SHORT):
                ax.imshow(grid,cmap='gray',origin='upper',vmin=0,vmax=1,interpolation='nearest')
                path=np.asarray(result.path);ax.plot(path[:,0],path[:,1],color=color,lw=2)
                ax.scatter(*start,c='#20b85a',edgecolors='black',s=65,zorder=5,label='Start')
                ax.scatter(*goal,c='#e33b44',marker='*',edgecolors='black',s=100,zorder=5,label='Goal')
                ax.set_title(f"{label}\nLength {row['Path length_'+suffix]:.2f} | Turns {row['Turns_'+suffix]:.0f}\nSaved runtime {row['Time ms_'+suffix]:.2f} ms")
                ax.set_xlabel('x');ax.set_ylabel('y');ax.legend(loc='best')
            fig.suptitle(f"{kind}: {row['map']} | scenario {int(row.scenario)}")
            name=f'figure_4_path_example_{index}';api.save_figure(fig,output,name)
            figures.append((name,'Paths reconstructed using saved weights and checked against saved length/turn count. Runtime annotations are original saved measurements, not rerun timings.'))
            record.update(status='Verified reconstruction',parameters=params,path_file=name+'.json')
            (output/(name+'.json')).write_text(json.dumps({'map':str(matches[0]),'start':start,'goal':goal,'paths':[r.path for r in results]},default=lambda v:v.item()),encoding='utf-8')
        except (OSError,ValueError,KeyError) as exc:
            record['status']='Unavailable: '+str(exc)
        records.append(record)
        gc.collect()
    pd.DataFrame(records,columns=['selection','map','scenario','status','parameters','path_file']).to_csv(output/'path_examples.csv',index=False)
    return figures,records


def presentation_figures(pairs, tables, output, maps_dir, source, manifest, runtime_tolerance, api):
    data=pairs[pairs['Both successful']].copy()
    figures,examples=path_examples(data,output,maps_dir,source,manifest['checkpoint_signature'],api)
    strata=[('All matched scenarios',data)]
    wins=[];heat=[]
    for k,(stratum,part) in enumerate(strata):
        suffix=''
        label=stratum
        fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
        x,y=part['Time ms_jps'],part['Time ms_opt'];valid=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        if valid.any():
            chart=ax.hexbin(x[valid],y[valid],xscale='log',yscale='log',bins='log',mincnt=1,gridsize=55,cmap='viridis')
            fig.colorbar(chart,ax=ax,label='Scenario count (log scale)')
            low=min(x[valid].min(),y[valid].min());high=max(x[valid].max(),y[valid].max())
            ax.plot([low,high],[low,high],'--',color='#c23b43',label='Equal runtime');ax.legend()
        else: ax.text(.5,.5,'No positive finite runtime pairs',ha='center',transform=ax.transAxes)
        ax.set_xlabel('Standard JPS runtime (ms)');ax.set_ylabel('Optimized runtime (ms)')
        ax.set_title(f'{label}\nBelow diagonal = optimized faster | n={int(valid.sum()):,}')
        name='figure_5_runtime_scatter'+suffix;api.save_figure(fig,output,name)
        figures.append((name,'Paired runtimes on logarithmic axes; all matched scenarios are pooled.'))
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
        for i,metric in enumerate(['Time ms','Path length','Turns']):
            rtol=runtime_tolerance if metric=='Time ms' else 0
            atol=1e-7 if metric=='Path length' else 0
            count=classify_pairs(part[metric+'_jps'],part[metric+'_opt'],rtol,atol)
            pct=100*count/count.sum() if count.sum() else np.zeros(3)
            left=0
            for j,(value,color,text) in enumerate(zip(pct,['#288a70','#c6ced5','#c86151'],['Win: lower','Tie','Loss: higher'])):
                ax.barh(i,value,left=left,color=color,label=text if i==0 else None)
                if value>=5: ax.text(left+value/2,i,f'{value:.1f}%',ha='center',va='center',color='black')
                left+=value
            wins.append(dict(Stratum=stratum,Metric=metric,N=int(count.sum()),Wins=int(count[0]),Ties=int(count[1]),Losses=int(count[2]),Relative_tolerance=rtol,Absolute_tolerance=atol))
        ax.set_yticks(range(3),['Runtime','Path length','Turns']);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('Matched successful scenarios (%)');ax.set_title(label);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
        name='figure_6_win_tie_loss'+suffix;api.save_figure(fig,output,name)
        figures.append((name,f'Optimized relative to JPS. Runtime ties: within {runtime_tolerance*100:g}% of JPS (user-selected descriptive tolerance, not a measured noise threshold). Path ties: absolute difference <= 1e-7; turn ties: exact equality.'))
        sizes=sorted(part['Map size'].unique(),key=lambda v:tuple(map(int,v.split(' x '))))
        if not sizes: continue
        fig,axes=plt.subplots(1,2,figsize=(13,max(3.5,len(sizes)*1.1)),constrained_layout=True)
        for ax,metric in zip(axes,['Time ms','Turns']):
            values=np.full((len(sizes),5),np.nan);counts=np.zeros((len(sizes),5),int)
            for i,size in enumerate(sizes):
                for j,ran in enumerate(api.RANGES):
                    cell=part[(part['Map size']==size)&(part.Range==ran)]
                    if metric=='Time ms': v=-paired_changes(cell,metric)
                    else: v=api.finite(cell['Turns_jps']-cell['Turns_opt'])
                    if len(v): values[i,j]=np.median(v)
                    counts[i,j]=len(v)
                    heat.append(dict(Stratum=stratum,Map_size=size,Range=ran,Metric=metric,N=len(v),Median=values[i,j]))
            limit=max(1,np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1
            cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#eeeeee')
            image=ax.imshow(values,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
            for i in range(len(sizes)):
                for j in range(5):ax.text(j,i,f'{values[i,j]:.1f}\nn={counts[i,j]:,}' if counts[i,j] else 'No data',ha='center',va='center',fontsize=9,color='white' if abs(values[i,j])>.65*limit else 'black')
            ax.set_xticks(range(5),[f'R{i+1}' for i in range(5)]);ax.set_yticks(range(len(sizes)),sizes)
            ax.set_title('Median paired runtime reduction (%)' if metric=='Time ms' else 'Median paired turns saved')
            fig.colorbar(image,ax=ax,shrink=.8)
        fig.suptitle(label+' | positive = optimized improvement')
        name='figure_8_improvement_heatmap'+suffix;api.save_figure(fig,output,name)
        figures.append((name,'Cells use per-scenario changes, then take their median. Runtime requires a positive JPS baseline; turns saved is an absolute difference, so zero-turn baselines remain included. Grey cells have no eligible data.'))
    pd.DataFrame(wins).to_csv(output/'win_tie_loss.csv',index=False)
    pd.DataFrame(heat).to_csv(output/'improvement_heatmap.csv',index=False)
    unit = manifest.get('statistics', {}).get('primary_unit', 'Map median difference (primary)')
    level = manifest.get('statistics', {}).get('primary_level', 'Map')
    stats=tables['statistical_comparisons'];stats=stats[(stats.Grouping=='Overall')&(stats.Unit==unit)].copy()
    intervals=[]
    fig,ax=plt.subplots(figsize=(12,7),constrained_layout=True)
    for i,(_,row) in enumerate(stats.iterrows()):
        n=int(row['Nonzero units']);positive=int(row['Optimized higher units'])
        if n:
            ci=binomtest(positive,n).proportion_ci(confidence_level=.95,method='exact');low,high=2*ci.low-1,2*ci.high-1
            effect=row['Paired sign effect'];ax.errorbar(effect,i,xerr=[[effect-low],[high-effect]],fmt='o',color=api.COLORS[1],capsize=4)
        else:
            low=high=np.nan;ax.text(0,i,'All ties / no data',ha='center',fontsize=9)
        p=row['Holm-adjusted p-value'];annotation=('p(adj) < machine precision' if p == 0 else f'p(adj)={p:.2e}') if np.isfinite(p) else 'Not tested'
        ax.text(1.06,i,annotation,va='center',fontsize=9)
        intervals.append(dict(Metric=row.Metric,Effect=row['Paired sign effect'],CI_low=low,CI_high=high,**{('Nonzero_cities' if level == 'City' else 'Nonzero_maps'): n},Holm_p=p))
    ax.set_yticks(range(len(stats)),stats.Metric);ax.invert_yaxis();ax.axvline(0,color='grey',ls='--');ax.set_xlim(-1.08,1.65)
    ax.set_xticks([-1,-.5,0,.5,1]);ax.set_xlabel('Paired sign effect: negative = lower optimized values')
    ax.set_title(f'{level}-level effects with pointwise 95% exact binomial confidence intervals')
    name='figure_7_effect_sizes';api.save_figure(fig,output,name)
    figures.append((name,f'Intervals transform the exact binomial interval for the proportion of positive nonzero {level.lower()} summaries. They condition on nonzero units, assume independent {level.lower()} groups, and are pointwise (not multiplicity-adjusted). All-tie metrics have no interval. Lower reference gap does not imply better optimality.'))
    pd.DataFrame(intervals).to_csv(output/'effect_size_intervals.csv',index=False)
    return {'presentation_figures':[{'name':name,'caption':caption} for name,caption in figures],
            'path_examples':examples,'runtime_tie_relative_tolerance':runtime_tolerance,
            'path_selection_rule':'Closest to cohort medians of reference length, percentage length change and turns saved, scaled by IQR; tie-break map/scenario. Trade-off example uses the same rule within positive-extra-length and positive-turn-savings cases, excluding the representative.'}
