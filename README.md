# Temporal Conformal Prediction (TCP)

Code and input data for rolling prediction intervals for financial returns. The study compares TCP, TCP-RM, CQR, clipped ACI, quantile regression and volatility models on thirteen financial series.

Paper: [arXiv:2507.05470](https://arxiv.org/abs/2507.05470)  
DOI: [10.48550/arXiv.2507.05470](https://doi.org/10.48550/arXiv.2507.05470)

## Run the study

Download this repository with **Code → Download ZIP**, extract it, and open a terminal in the extracted folder. Use Python 3.11, then run:

```powershell
python -m pip install -r requirements.txt
python reproduce.py
```

The second command fits the models on all thirteen series, runs the sensitivity grid, and generates the tables and all seven manuscript figures. It writes the outputs to:

- `build/forecasts/`: forecasts, evaluation summaries and fit diagnostics.
- `build/tables/`: CSV summaries and LaTeX tables.
- `build/figures/`: figure files in PNG format.
- `build/environment.json`: the Python and package versions used.

The full run takes time because the models are fitted repeatedly over rolling windows. Runtime depends on your computer. These commands also work on macOS and Linux with the appropriate Python executable.

After a completed run, you can rebuild the tables and figures without fitting the models again:

```powershell
python reproduce.py --results build/forecasts
```

## Files

| Path | Contents |
| --- | --- |
| `financial_returns.csv` | Input dataset |
| `reproduce.py` | Runs the study in order |
| `code/run_benchmarks.py` | Fits the main comparisons |
| `code/sensitivity.py` | Fits the sensitivity grid |
| `code/tail_levels.py` | Calculates the attainable-level one-sided backtests |
| `code/make_results.py` | Generates the main tables and figures |
| `code/make_appendix.py` | Generates the appendix tables and figures |
| `code/core.py`, `code/garch_family.py`, `code/evaluation.py` | Shared forecasting and evaluation calculations |
| `code/threadpin.py` | Sets numerical libraries to one thread |
| `tests/test_core.py` | Checks calibration updates, statistical calculations and forecast timing |
| `requirements.txt` | Package versions |

To run the numerical checks:

```powershell
python tests/test_core.py
```

## Data and calculations

The dataset contains 1,720 observations from 10 November 2017 to 20 May 2025. Returns are on a percentage scale. The main evaluation uses 1,448 common forecast dates from 23 January 2019 to 20 May 2025.

The thirteen series are SP500, FTSE, N225, DAX, BTC-USD, ETH-USD, LTC-USD, Gold, WTI, NatGas, EURUSD=X, GBPUSD=X and JPYUSD=X. VIX is present in the input file but is not used. The data contain gaps in calendar dates. Source prices and the dataset construction script are not included.

The SHA-256 of `financial_returns.csv` is:

```text
e86ef3ab9f343ddf3febb47dea682498144300f9a3aaab736f8bfbb3e3befb4a
```

Numerical optimizers can produce small differences across platforms, even with the same package versions. The calculations use one numerical thread.

There are ten forecasting methods and two diagnostic controls. `Reversed-sign-control` reverses the clipped ACI update. `Fixed-coefficient-control` uses a fixed variance recursion.

A nominal 95% two-sided interval does not imply a calibrated 2.5% lower tail. Lower endpoints from the two-sided conformal methods are evaluated as diagnostics. The separate one-sided CQR forecasts are labelled `CQR-1s`.

With 60 calibration observations, a finite one-sided conformal threshold cannot attain a requested 1% miscoverage level. The corresponding `SKIPPED` message is expected. The finest attainable reference is 1/61 under the exchangeable split-conformal argument. That argument does not automatically provide coverage guarantees for dependent financial returns.

## Citation

Please cite the paper if you use this work. Record the repository commit when reporting a reproduction.

```bibtex
@misc{Aich2025TCP,
  author        = {Aich, Agnideep and Aich, Ashit Baran and Jain, Dipak C.},
  title         = {Temporal Conformal Prediction (TCP): A Distribution-Free Statistical and Machine Learning Framework for Adaptive Risk Forecasting},
  year          = {2025},
  eprint        = {2507.05470},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2507.05470},
  url           = {https://arxiv.org/abs/2507.05470}
}
```

For code questions, open a GitHub issue. The corresponding author is Ashit Baran Aich, [aichnsou@gmail.com](mailto:aichnsou@gmail.com).

## License

The software is provided under the existing [MIT License](LICENSE).
