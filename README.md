# Temporal Conformal Prediction (TCP)

Code, study data and saved forecasts for rolling prediction intervals for financial returns. The study compares TCP, TCP-RM, CQR, clipped ACI, quantile regression and volatility models on thirteen financial series.

Paper: [arXiv:2507.05470](https://arxiv.org/abs/2507.05470)  
DOI: [10.48550/arXiv.2507.05470](https://doi.org/10.48550/arXiv.2507.05470)

## Run the code

Download this repository with **Code → Download ZIP**, extract it, and open a terminal in the extracted folder. Use Python 3.11 and install the packages:

```powershell
python -m pip install -r requirements.txt
```

To recreate the tables and all seven manuscript figures from the saved forecasts:

```powershell
python reproduce.py
```

This checks the saved interval summaries and tail backtests against the forecast arrays. It writes tables to `build/tables/` and figures to `build/figures/`. It does not fit models or overwrite the reference files.

To fit all thirteen series and the sensitivity grid, then rebuild the tables and figures:

```powershell
python reproduce.py --refit
```

New forecasts are written to `build/forecasts/`. Refitting all models and sensitivity settings takes longer than rebuilding the reports. The time depends on your computer.

These commands also work in macOS and Linux terminals with the appropriate Python executable.

## Files

| Path | Contents |
| --- | --- |
| `financial_returns.csv` | The input dataset used in the study |
| `reproduce.py` | Runs the calculations in order |
| `code/` | Forecasting, evaluation and figure scripts |
| `results/` | Reference forecasts, summaries, fit diagnostics and sensitivity results |
| `tables/` | CSV summaries and LaTeX tables generated from the reference forecasts |
| `figures/` | The seven figures used in the manuscript |
| `tests/test_core.py` | Checks of calibration updates, statistical calculations and forecast timing |
| `requirements.txt` | Package versions used for the calculations |

The code is split by function. `run_benchmarks.py` fits the main comparisons, `sensitivity.py` fits the sensitivity grid, and `tail_levels.py` adds the attainable-level one-sided backtests. `make_results.py` and `make_appendix.py` generate the tables and figures. The remaining modules contain the shared calculations.

Run the mathematical and numerical checks with:

```powershell
python tests/test_core.py
```

## Study data and reference results

The dataset contains 1,720 recorded observations from 10 November 2017 to 20 May 2025. Returns are on a percentage scale. The main evaluation uses 1,448 common forecast dates from 23 January 2019 to 20 May 2025.

The thirteen series are SP500, FTSE, N225, DAX, BTC-USD, ETH-USD, LTC-USD, Gold, WTI, NatGas, EURUSD=X, GBPUSD=X and JPYUSD=X. VIX is present in the input file but is not used by the models. The file contains gaps in calendar dates. Source prices and the dataset construction script are not included, so downloading current prices is not a documented way to reconstruct this exact input.

The SHA-256 of `financial_returns.csv` is:

```text
e86ef3ab9f343ddf3febb47dea682498144300f9a3aaab736f8bfbb3e3befb4a
```

The saved forecasts and sensitivity results are the inputs used to produce the reported tables and figures.

Each `raw_<asset>.npz` contains forecast dates, observed returns, interval endpoints, lower-tail forecasts and, where defined, expected-shortfall forecasts. `ALL_interval.csv` and `ALL_var.csv` collect the per-series summaries. The latter includes one-sided CQR backtests at both the requested and attainable reference levels. `garchdiag_<asset>.csv` records the selected volatility fits.

Refitting numerical optimizers on another platform can produce small differences, even with the same package versions. The saved forecasts are the reference for the reported tables and figures; a fresh run is not promised to be identical across platforms. The calculations use one numerical thread.

## Interpreting the comparisons

There are ten forecasting methods and two diagnostic controls. `Reversed-sign-control` reverses the clipped ACI update. `Fixed-coefficient-control` uses the stated fixed variance recursion. They are not fitted ACI or GARCH competitors.

TCP clips the CQR calibration threshold at zero. TCP-RM adds an online offset. The offset can make the total expansion negative on some dates. A nominal 95% two-sided interval does not imply a calibrated 2.5% lower tail. Lower endpoints from the two-sided conformal methods are therefore evaluated as diagnostics. The separate one-sided CQR forecasts are labelled `CQR-1s`.

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
