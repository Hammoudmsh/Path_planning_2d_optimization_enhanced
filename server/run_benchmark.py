from pathlib import Path
import argparse

from benchmark_config import DATASETS, EXPERIMENT_CONFIGS, get_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("config", choices=sorted(EXPERIMENT_CONFIGS))
    args = parser.parse_args()

    config = get_config(
        args.dataset,
        args.config,
    )

    print("Selected configuration:")
    for key in (
        "dataset_name",
        "config_name",
        "maps_url",
        "scenarios_url",
        "results",
    ):
        print(f"{key:20s}: {config[key]}")

    print(
        "\nExport the notebook to benchmark.py, or import the benchmark "
        "module here, then call run_experiment(config). The selected profile "
        "now includes Fast Weighted SA-JPS-Dijkstra Corridor. All dataset-specific "
        "outputs use a dataset-name suffix, and every plot is accompanied by "
        "a reproducibility CSV containing the exact plotted data."
    )


if __name__ == "__main__":
    main()
