# anomaly-detection-compare

Compares z-score, IQR and Isolation Forest outlier detection on 500 synthetic 1-D points with 15 injected anomalies.

## Run

```bash
pip install -r requirements.txt
python anomaly.py
pytest
```

## Output (reproduced)

```
$ python3 anomaly.py
method             precision  recall     f1
zscore(3)               1.00    1.00   1.00
iqr(1.5)                0.71    1.00   0.83
isolation_forest        1.00    1.00   1.00

$ pytest -q
5 passed in 0.82s
```

## Limits

- Synthetic data with large, obvious anomalies, so the methods score high. IQR flags extra normal points as outliers (lower precision). Not representative of real data.
