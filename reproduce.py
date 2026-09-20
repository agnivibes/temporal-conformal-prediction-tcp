"""Fit the study models and generate the tables and figures."""

from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ASSETS = "SP500,FTSE,N225,DAX,BTC-USD,ETH-USD,LTC-USD,Gold,WTI,NatGas,EURUSD=X,GBPUSD=X,JPYUSD=X"
DATA_HASH = "e86ef3ab9f343ddf3febb47dea682498144300f9a3aaab736f8bfbb3e3befb4a"


def run(script, *arguments):
    command = [sys.executable, str(ROOT / "code" / script), *map(str, arguments)]
    subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        help="Reuse forecasts from a completed run instead of fitting models again.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build",
        help="Destination for forecasts, tables and figures.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / "financial_returns.csv"
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_HASH:
        raise ValueError("financial_returns.csv differs from the study dataset.")
    if args.results is None:
        results = output / "forecasts"
        run(
            "run_benchmarks.py", "--assets", ASSETS, "--data", data, "--outdir", results
        )
        run("tail_levels.py", "--outdir", results)
        run("sensitivity.py", "--data", data, "--outdir", results)
    else:
        results = args.results.resolve()
    sensitivity = results / "sensitivity_2d.csv"
    if not sensitivity.is_file():
        raise FileNotFoundError(f"Missing sensitivity results: {sensitivity}")
    for script in ["make_results.py", "make_appendix.py"]:
        run(
            script,
            "--results",
            results,
            "--sensitivity",
            sensitivity,
            "--outdir",
            output,
        )
    versions = {}
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lightgbm",
        "arch",
        "statsmodels",
        "matplotlib",
        "tabulate",
    ]:
        versions[name] = importlib.metadata.version(name)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "refitted_forecasts": args.results is None,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f'Tables: {output / "tables"}')
    print(f'Figures: {output / "figures"}')


if __name__ == "__main__":
    main()
