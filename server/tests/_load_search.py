"""Load search definitions without executing the exported notebook's experiments."""
import ast
import sys
import __future__
import heapq
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = Path(__file__).resolve().parents[1] / 'Astar_JPS_SA_JPS_config_profiles_ready.py'


def load_search(extra_functions=(), extra_constants=()):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    namespace = dict(globals(), Point=Tuple[int, int], Direction=Tuple[int, int],
                     SQRT2=math.sqrt(2), CONFIG={'sa_turn_weight': 0.35,
                                               'fast_sa_heuristic_weight': 1.15})
    constants = {'DIRECTIONS_8', 'DIRECTION_TO_ID', 'START_DIRECTION_ID',
                 'OPTIMIZED_TURN_SEVERITY', 'ALGORITHM_ORDER'}
    functions = {
        'sign', 'movement_direction', 'octile_distance', 'segment_distance',
        'in_bounds', 'walkable', 'can_step', 'expand_jump_path', 'path_length',
        'path_turn_metrics', 'count_waypoints', 'SearchResult',
        'reconstruct_point_path', 'run_astar', 'has_forced_neighbour',
        'pruned_directions', 'build_jump_function', 'reconstruct_jps_state_path',
        'run_jps', 'direction_angle_degrees', 'turn_severity_45',
        'reconstruct_sa_jps_path', 'run_sa_jps', 'build_optimized_jump_function',
        'run_optimized_fwsjps', 'validate_path', 'normalise_approaches',
        'run_all_algorithms', 'read_map', 'read_scenario',
    }
    functions.update(extra_functions)
    constants.update(extra_constants)
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in constants for target in targets):
                nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace
