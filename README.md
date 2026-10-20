# Enhanced 2D Path Planning Optimization

Python research code for benchmarking A*, Jump Point Search (JPS), and smoothness-aware JPS variants on MovingAI grid maps. The project includes optimized Fast Weighted SA-JPS, configurable turning penalties, parallel benchmarks, resumable checkpoints, sensitivity experiments, and report generation.

## Project layout

| Path | Contents |
| --- | --- |
| `server/Astar_JPS_SA_JPS_config_profiles_ready.py` | Main benchmark entry point and search implementations |
| `server/benchmark_config.py` | Dataset definitions, experiment profiles, and parameter settings |
| `server/parallel_benchmark.py` | Parallel benchmark support |
| `server/exp_config.json` | Comparison of standard JPS and original/optimized Fast Weighted SA-JPS |
| `server/exp_smoothness.json` | Comparison with an additional per-turn penalty |
| `server/create_report.py` | Matched-scenario reports from SQLite checkpoints |
| `server/create_report_city.py` | Reports with source-city grouping support |
| `server/presentation_plots.py` | Presentation figures and path illustrations |
| `server/tests/` | Unit tests and benchmark validation utilities |
| `results_new_ablation/` | Existing experiment outputs, logs, and checkpoints |

`server/run_benchmark.py` only prints a selected configuration; use the main benchmark entry point shown below to run experiments.

## Installation

Python 3.10 is specified by the supplied Conda environment.

```bash
git clone https://github.com/Hammoudmsh/Path_planning_2d_optimization_enhanced.git
cd Path_planning_2d_optimization_enhanced
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r server/requirements.txt
```

Alternatively, from `server/`, run `conda env create -f environment.yml` and `conda activate astar_jps`.

Large result CSV, SQLite checkpoint, and pickle files use Git LFS. Install Git LFS and run `git lfs install` before cloning to retrieve these artifacts automatically, or run `git lfs pull` after cloning. They are not required to start a new benchmark.

## Datasets

Datasets are not included in this repository. Download the map and scenario archives specified in `DATASETS` in `server/benchmark_config.py` and extract them into:

```text
datasets/
  street/
    maps/        # .map files
    scenarios/   # .scen files
```

Use the same structure for other supported dataset names: `dao`, `wc3maps512`, `bg512`, `bgmaps`, `sc1`, `maze`, and `random`. Keep map filenames consistent with their scenario files.

Profiles with `download_data: true` can download the configured archives. The comparison, street, test, and sensitivity profiles disable downloads and expect local datasets. Consult the dataset provider's attribution and usage terms when using or redistributing benchmark data.

## Run a benchmark

To run all street maps and scenarios with 16 workers and then generate the report, first prepare the street dataset as described above. From the repository root:

```bash
cd server
python Astar_JPS_SA_JPS_config_profiles_ready.py street street_all --workers 16
python create_report.py
```

Wait for the benchmark to finish successfully before running `create_report.py`. The report script uses the default street checkpoint under `results_new_ablation/street/street_all/cache/` and creates a new report output directory automatically.

Return to the repository root with `cd ..` before using the examples below.

Run these commands from the repository root. Inspect a small experiment without running searches:

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street test --dry-run
```

After preparing the dataset, run the small comparison:

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street test
```

Run 50 maps with 20 scenarios per map, or all street maps and scenarios:

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street street_50x20 --workers 4
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street street_all --workers 4
```

The `street_50x20` profile compares standard JPS with original and optimized Fast Weighted SA-JPS. The `street_all` profile compares standard JPS with optimized Fast Weighted SA-JPS using an additional per-turn penalty.

Override the optimized algorithm's angle penalty (`--lambda`), per-turn penalty (`--beta`), and heuristic multiplier (`--w`):

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street test --lambda 0.35 --beta 5.0 --w 1.15
```

Use `--map-size 1024` to select square 1024-by-1024 maps and `--map-limit 2` to limit a run. Use `--help` for all command-line options.

JSON experiment configurations are also supported:

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street full --config-json server/exp_smoothness.json --dry-run
```

Remove `--dry-run` to execute the configured experiment.

## Sensitivity studies and outputs

The `sensitivity_test`, `sensitivity`, and `sensitivity_all` profiles vary one optimized search coefficient at a time. The small test uses two maps and one scenario per map:

```bash
python server/Astar_JPS_SA_JPS_config_profiles_ready.py street sensitivity_test
```

Default outputs are written under `results_new_ablation/<dataset>/<profile>/`. Parameter overrides or JSON configurations may change the output directory. Outputs depend on the profile and include per-run metrics, summaries, figures, plotting data, logs, and resumable checkpoints. Benchmark caching and resume behavior are controlled in `benchmark_config.py`.

## Generate reports

The report scripts default to the included street benchmark checkpoint. After retrieving large artifacts through Git LFS:

```bash
python server/create_report.py --output reports/street
python server/create_report_city.py --output reports/street_city
```

Use a new output directory for each report. Supply `--checkpoint PATH` for another SQLite checkpoint, `--maps-dir datasets/street/maps` for path illustrations, and `--export-pairs` to export scenario-level data. The city report accepts `--city-groups PATH` for a JSON mapping of map filenames to source-city groups.

## Tests

```bash
python -m unittest discover -s server/tests -p "test_*.py"
```

For cluster execution, `server/run_job.slurm` provides a Slurm template. Adjust its partition, resources, and Conda activation path for your system; create the `logs/` directory before submitting from `server/`.

## Research interpretation

Smoothness-aware and weighted variants are experimental. Heading-dependent turn penalties change the search objective, and weighted heuristics can trade path quality for speed. Evaluate runtime, path length, turning metrics, and search effort together; do not assume every variant retains standard JPS distance-optimality.

## Excluded local content

The repository ignores `datasets/`, `New folder/`, `exp.txt`, Python bytecode caches, the two specified street result ZIP archives, and `Astar_JPS_SA_JPS_dataset_suffix_plot_data_mean_median_std.ipynb`. The exported Python benchmark remains included.
