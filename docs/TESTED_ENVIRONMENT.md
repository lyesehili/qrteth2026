# Tested environment

Validated on 2026-11-24 using the pre-existing local `qrt` environment.
Python 3.11.18, Linux x86_64.

All runs used CPU estimators; no installation, data download or remote service was needed.
Notebook execution uses local Jupyter loopback sockets. In a restricted sandbox those
sockets require permission; normal offline use needs no external network access.

| Package | Installed version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| lightgbm | 4.7.0 |
| xgboost | 3.2.0 |
| catboost | 1.2.10 |
| optuna | 5.1.0 |
| matplotlib | 3.11.2 |
| statsmodels | 0.15.0 |
| pyarrow | 26.0.1 |
| jupyterlab | 4.6.6 |
| ipykernel | 7.4.0 |
| nbformat | 5.11.1 |
| nbclient | 0.11.0 |
| threadpoolctl | 3.8.0 |

`requirements.txt` specifies compatible version ranges; this table records the actual
environment tested, rather than claiming every version in those ranges was exercised.
Cache/install dependencies on the competition machine before going offline.

Validation commands:

```bash
python -m unittest discover -s tests -v
python scripts/check_notebooks.py
python -m qrtkit --demo classification
python -m qrtkit --demo regression
python -m qrtkit --demo financial
python -m qrtkit --demo multiclass
```

See [TEST_RESULTS.md](TEST_RESULTS.md) for the checks and illustrative synthetic scores.
