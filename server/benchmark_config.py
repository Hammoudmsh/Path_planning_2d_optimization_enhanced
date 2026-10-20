from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, Optional


DATASETS: Dict[str, Dict[str, str]] = {
    "street": {
        "maps_url": "https://www.movingai.com/benchmarks/street/street-map.zip",
        "scenarios_url": "https://www.movingai.com/benchmarks/street/street-scen.zip",
    },
    "dao": {
        "maps_url": "https://movingai.com/benchmarks/dao/dao-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/dao/dao-scen.zip",
    },
    "wc3maps512": {
        "maps_url": "https://movingai.com/benchmarks/wc3maps512/wc3maps512-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/wc3maps512/wc3maps512-scen.zip",
    },
    "bg512": {
        "maps_url": "https://movingai.com/benchmarks/bg512/bg512-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/bg512/bg512-scen.zip",
    },
    "bgmaps": {
        "maps_url": "https://movingai.com/benchmarks/bgmaps/bgmaps-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/bgmaps/bgmaps-scen.zip",
    },
    "sc1": {
        "maps_url": "https://movingai.com/benchmarks/sc1/sc1-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/sc1/sc1-scen.zip",
    },
    "maze": {
        "maps_url": "https://movingai.com/benchmarks/maze/maze-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/maze/maze-scen.zip",
    },
    "random": {
        "maps_url": "https://movingai.com/benchmarks/random/random-map.zip",
        "scenarios_url": "https://movingai.com/benchmarks/random/random-scen.zip",
    },
}


COMMON_CONFIG: Dict[str, Any] = {
    "download_data": True,

    "use_benchmark_cache": True,
    "force_recompute_benchmark": False,
    "resume_benchmark": True,

    "fast_jps_weight": 1.15,
    "sa_turn_weight": 0.35,
    "fast_sa_heuristic_weight": 1.15,
    "hybrid_corridor_radius": 4,
    "fast_sa_corridor_radius": 4,

    "map_visibility_samples": 1000,
    "map_visibility_distance": 40,
    "map_type_open_density_max": 0.18,
    "map_type_open_visibility_min": 0.45,
    "map_type_open_corridor_max": 0.10,
    "map_type_maze_corridor_min": 0.22,
    "map_type_maze_visibility_max": 0.16,
    "map_type_maze_dead_end_min": 0.025,
    "map_type_urban_edge_min": 0.10,
    "map_type_urban_junction_min": 0.15,
    "run_structural_clustering": True,
    "map_cluster_count": 4,

    "run_statistical_analysis": True,
    "statistics_by_map_type": True,
    "statistical_alpha": 0.05,
    "generate_effect_size_tables": True,
    "display_effect_size_tables": True,
    "show_cd_diagrams": False,
    "statistical_algorithms": None,
    "statistical_metrics": [
        "Time ms",
        "Expanded nodes",
        "Generated successors",
        "Path length",
        "Gap vs A* percent",
        "Turns",
        "Total turning angle deg",
        "Waypoints",
    ],

    "sa_turn_weight_sweep": [
        0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00,
    ],

    "run_ablation_study": True,
    "ablation_fast_jps_weight": True,
    "ablation_sa_turn_weight": True,
    "ablation_fast_sa_heuristic_weight": True,
    "ablation_corridor_radius": True,
    "ablation_component_comparison": True,

    "ablation_fast_jps_weights": [
        1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40,
    ],
    "ablation_sa_turn_weights": [
        0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00,
    ],
    "ablation_fast_sa_heuristic_weights": [
        1.00, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40,
    ],
    "ablation_corridor_radii": [1, 2, 3, 4, 5, 6, 8],

    "run_smoke_test": True,
    "run_main_benchmark": True,
    "run_sa_weight_sweep": False,

    "display_tables": True,
    "save_bar_plots": True,
    "save_per_map_plots": True,
    "show_per_map_plots": False,
    "display_per_map_table": False,
    "save_example_plot": True,
    "example_plot_filename": "example_paths.png",

    # Save the exact rows used by every generated plot.
    "save_plot_data_csv": True,
    "save_summary_standard_deviation": True,
    # Append the active dataset name before every output extension.
    "append_dataset_suffix_to_all_files": True,
    "create_zip_archive": True,
    "archive_name": "astar_jps_all_maps_all_scenarios_ablation_results",
}


ALL_APPROACHES_org = [
    "A*",
    "Standard JPS",
    "Bidirectional Standard JPS",
    "Smooth JPS",
    "SA-JPS",
    "Bidirectional SA-JPS",
    "Fast Weighted SA-JPS",
    "Optimized Fast Weighted SA-JPS",
    "Bidirectional Fast Weighted SA-JPS",
    "JPS-Dijkstra Corridor",
    "Fast Weighted SA-JPS-Dijkstra Corridor",
]
ALL_APPROACHES = [
    #"A*",
    #"Standard JPS",
    #"Bidirectional Standard JPS",
    #"Smooth JPS",
    #"SA-JPS",
    #"Bidirectional SA-JPS",
    #"Fast Weighted SA-JPS",
    #"Bidirectional Fast Weighted SA-JPS",
    #"JPS-Dijkstra Corridor",
    "Fast Weighted SA-JPS-Dijkstra Corridor",
]




EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "description": "Small debugging run",
        "map_sizes": [256],
        "map_variant": list(range(30)),
        "map_limit": 2,
        "scenarios_per_map": 2,
        "scenario_selection": "uniform",
        "random_seed": 42,
        # "benchmark_approaches": [
        #     "A*",
        #     "Standard JPS",
        #     "SA-JPS",
        #     "Fast Weighted SA-JPS",
        # ],
        "benchmark_approaches": list(ALL_APPROACHES),

        "run_smoke_test": True,
        "run_statistical_analysis": True,
        "run_ablation_study": True,
        "create_zip_archive": False,
    },


    "full": {
        "description": "Full thesis benchmark",
        "map_sizes": None,#[256, 512, 1024],
        "map_variant": list(range(200)),
        "map_limit": None,
        "scenarios_per_map": 200,
        "scenario_selection": "uniform",
        "random_seed": 42,
        "benchmark_approaches": list(ALL_APPROACHES),
        "run_smoke_test": False,
        "run_statistical_analysis": True,
        "run_ablation_study": True,
        "create_zip_archive": True,
    },
}


EXPERIMENT_CONFIGS["optimized_vs_jps"] = {
    **EXPERIMENT_CONFIGS["debug"],
    "description": "Standard JPS versus original and optimized Fast Weighted SA-JPS",
    "download_data": False,
    "benchmark_approaches": [
        {"label": "Standard JPS", "algorithm": "Standard JPS",
         "parameters": {"heuristic_weight": 1.0, "smooth_tie_break": False}},
        {"label": "Fast Weighted SA-JPS", "algorithm": "Fast Weighted SA-JPS",
         "parameters": {"turn_weight": 0.35, "heuristic_weight": 1.15}},
        {"label": "Optimized Fast Weighted SA-JPS",
         "algorithm": "Optimized Fast Weighted SA-JPS",
         "parameters": {"turn_weight": 0.35, "heuristic_weight": 1.15}},
    ],
    "run_smoke_test": False,
    "run_ablation_study": False,
    "run_sa_weight_sweep": False,
    "run_inference_example": False,
    "generate_all_dataset_descriptions": False,
    "run_structural_clustering": False,
    "use_benchmark_cache": True,
    "force_recompute_benchmark": False,
    "resume_benchmark": True,
}


EXPERIMENT_CONFIGS["street_50x20"] = {
    **EXPERIMENT_CONFIGS["optimized_vs_jps"],
    "description": "50 street maps, 20 scenarios per map; JPS and both FWSJPS variants",
    "map_sizes": None,
    "map_limit": 50,
    "scenarios_per_map": 20,
    "scenario_selection": "uniform",
    "run_statistical_analysis": False,
    "save_bar_plots": False,
    "save_per_map_plots": False,
    "save_example_plot": False,
    "create_zip_archive": True,
}


EXPERIMENT_CONFIGS["street_smoothness_50x20"] = {
    **EXPERIMENT_CONFIGS["street_50x20"],
    "description": "50 street maps: test an additional per-turn penalty in optimized FWSJPS",
    "benchmark_approaches": [
        {"label": "Standard JPS", "algorithm": "Standard JPS",
         "parameters": {"heuristic_weight": 1.0, "smooth_tie_break": False}},
        {"label": "Fast Weighted SA-JPS", "algorithm": "Fast Weighted SA-JPS",
         "parameters": {"turn_weight": 0.35, "heuristic_weight": 1.15}},
        {"label": "Optimized Fast Weighted SA-JPS",
         "algorithm": "Optimized Fast Weighted SA-JPS",
         "parameters": {"turn_weight": 0.35, "heuristic_weight": 1.15,
                        "turn_count_weight": 5.0}},
    ],
}


EXPERIMENT_CONFIGS["street_all"] = {
    **EXPERIMENT_CONFIGS["street_smoothness_50x20"],
    "description": "All street maps and scenarios: standard JPS versus optimized FWSAJPS",
    "map_sizes": None,
    "map_limit": None,
    "scenarios_per_map": None,
    "benchmark_approaches": [
        {"label": "Standard JPS", "algorithm": "Standard JPS",
         "parameters": {"heuristic_weight": 1.0, "smooth_tie_break": False}},
        {"label": "Optimized Fast Weighted SA-JPS",
         "algorithm": "Optimized Fast Weighted SA-JPS",
         "parameters": {"turn_weight": 0.35, "heuristic_weight": 1.15,
                        "turn_count_weight": 5.0}},
    ],
}


EXPERIMENT_CONFIGS["test"] = {
    **EXPERIMENT_CONFIGS["street_50x20"],
    "description": "Two maps, one nontrivial scenario each; configurable optimized FWSJPS equation",
    "map_limit": 2,
    "scenarios_per_map": 1,
    # Uniform selection of a single scenario chooses the shortest, often zero.
    "scenario_selection": "longest",
    "create_zip_archive": False,
}


# One-at-a-time sensitivity study. Edit these lists to change the sweep.
SENSITIVITY_BASELINE = {'turn_weight': 0.35, 'turn_count_weight': 5.0, 'heuristic_weight': 1.15}
SENSITIVITY_VALUES = {
    'turn_weight': [0.0, 0.1, 0.35, 0.75, 1.5],
    'turn_count_weight': [0.0, 1.0, 2.5, 5.0, 10.0],
    'heuristic_weight': [1.0, 1.05, 1.15, 1.3, 1.5],
}


def sensitivity_approaches():
    """Run the shared baseline once, then vary only one coefficient per setting."""
    settings = [dict(SENSITIVITY_BASELINE)]
    for parameter, values in SENSITIVITY_VALUES.items():
        for value in values:
            if not math.isfinite(value) or value < 0 or (parameter == 'heuristic_weight' and value == 0):
                raise ValueError('Sensitivity coefficients must be finite and nonnegative; w must be positive.')
            setting = dict(SENSITIVITY_BASELINE, **{parameter: value})
            if setting not in settings:
                settings.append(setting)
    return [
        {'label': 'Optimized FWSAJPS [lambda={turn_weight:g}, beta={turn_count_weight:g}, w={heuristic_weight:g}]'.format(**setting),
         'algorithm': 'Optimized Fast Weighted SA-JPS', 'parameters': setting}
        for setting in settings
    ]


EXPERIMENT_CONFIGS['sensitivity'] = {
    **EXPERIMENT_CONFIGS['street_smoothness_50x20'],
    'description': 'Optimized FWSAJPS only: one-at-a-time lambda/beta/w sensitivity, all maps, 50 scenarios each',
    'map_sizes': None,
    'map_limit': None,
    'scenarios_per_map': 50,
    'scenario_selection': 'uniform',
    'benchmark_approaches': sensitivity_approaches(),
    'sensitivity_study': True,
}
EXPERIMENT_CONFIGS['sensitivity_test'] = {
    **EXPERIMENT_CONFIGS['sensitivity'],
    'description': 'Sensitivity smoke test: 2 maps, 1 nontrivial scenario each, optimized FWSAJPS only',
    'map_limit': 2,
    'scenarios_per_map': 1,
    'scenario_selection': 'longest',
    'create_zip_archive': False,
}
EXPERIMENT_CONFIGS['sensitivity_all'] = {
    **EXPERIMENT_CONFIGS['sensitivity'],
    'description': 'Optimized FWSAJPS sensitivity on every map and scenario',
    'scenarios_per_map': None,
}


def apply_equation_overrides(config, *, lambda_weight=None, beta_weight=None,
                             heuristic_weight=None):
    """Override only optimized FWSJPS; keep parameter experiments in separate folders."""
    supplied = {'turn_weight': lambda_weight, 'turn_count_weight': beta_weight,
                'heuristic_weight': heuristic_weight}
    supplied = {key: float(value) for key, value in supplied.items() if value is not None}
    if not supplied:
        return config
    if config.get('sensitivity_study', False):
        raise ValueError('Sensitivity profiles already sweep lambda/beta/w. Edit SENSITIVITY_VALUES in benchmark_config.py instead of using equation overrides.')
    for key, value in supplied.items():
        if not math.isfinite(value) or value < 0 or (key == 'heuristic_weight' and value == 0):
            raise ValueError('Lambda and beta must be finite and non-negative; w must be finite and positive.')
    result = deepcopy(config)
    approaches = []
    found = False
    for item in result.get('benchmark_approaches', []):
        algorithm = item.get('algorithm') if isinstance(item, dict) else item
        if algorithm == 'Optimized Fast Weighted SA-JPS':
            if isinstance(item, str):
                item = {'label': item, 'algorithm': item, 'parameters': {}}
            item.setdefault('parameters', {}).update(supplied)
            found = True
        approaches.append(item)
    if not found:
        raise ValueError('Equation overrides require Optimized Fast Weighted SA-JPS in benchmark_approaches.')
    result['benchmark_approaches'] = approaches
    suffix = '_'.join(f'{name}_{supplied[key]}' if key in supplied else f'{name}_profile'
                      for name, key in (('lambda', 'turn_weight'), ('beta', 'turn_count_weight'),
                                        ('w', 'heuristic_weight')))
    result['results'] = Path(result['results']) / suffix
    result['benchmark_cache_dir'] = result['results'] / 'cache'
    return result


def get_config(
    dataset_name: str,
    profile_name: str,
    *,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_name = str(dataset_name).strip().lower()
    profile_name = str(profile_name).strip().lower()

    # if dataset_name == "ado":
    #     dataset_name = "dao"

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available datasets: {sorted(DATASETS)}"
        )

    if profile_name not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Available profiles: {sorted(EXPERIMENT_CONFIGS)}"
        )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        # This module lives in <project>/server. Launch directory must not
        # change dataset, result, or checkpoint locations.
        else Path(__file__).resolve().parent.parent
    )

    config = deepcopy(COMMON_CONFIG)
    config.update(deepcopy(EXPERIMENT_CONFIGS[profile_name]))

    dataset = DATASETS[dataset_name]

    data_root = root / "datasets" / dataset_name
    results_root = root / "results_new_ablation" / dataset_name / profile_name

    config.update(
        {
            "dataset_name": dataset_name,
            "config_name": profile_name,
            "maps_url": dataset["maps_url"],
            "scenarios_url": dataset["scenarios_url"],
            "project_root": root,
            "data_dir": data_root,
            "maps_dir": data_root / "maps",
            "scenarios_dir": data_root / "scenarios",
            "results": results_root,
            "benchmark_cache_dir": results_root / "cache",
            "archive_name": (
                f"{dataset_name}_{profile_name}_"
                "astar_jps_results"
            ),
        }
    )

    return config
