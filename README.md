# T0 ML/MLOps Batch Job

Minimal Python batch job for a deterministic rolling-mean trading signal pipeline.

## Files

- `run.py` - CLI entry point
- `config.yaml` - run configuration
- `data.csv` - OHLCV input data
- `requirements.txt` - Python dependencies
- `Dockerfile` - containerized one-command run
- `metrics.json` - sample successful metrics output
- `run.log` - sample successful log output

## Local Run

```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

The program writes `metrics.json` in both success and error cases, writes detailed logs to `run.log`, prints the final metrics JSON to stdout, and exits non-zero on failure.

## Docker Run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The Docker image includes `run.py`, `config.yaml`, and `data.csv`. The default container command produces `metrics.json` and `run.log` inside the container and prints the final metrics JSON to stdout.

## Processing Choice

The rolling mean uses `window` rows and leaves the first `window - 1` rolling values as `NaN`. Those warm-up rows are excluded from signal-rate calculation because they do not have enough historical data for a full-window signal.

## Example `metrics.json`

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 24,
  "seed": 42,
  "status": "success"
}
```

`latency_ms` varies by machine and runtime environment.
