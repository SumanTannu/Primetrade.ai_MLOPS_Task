import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")
REQUIRED_DATA_COLUMN = "close"


class ValidationError(Exception):
    """Raised when input validation fails."""


def setup_logging(log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )


def write_metrics(output_path: str, metrics: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise ValidationError(f"Config file not found: {config_path}")
    if not path.is_file():
        raise ValidationError(f"Config path is not a file: {config_path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValidationError(f"Invalid YAML config: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Unable to read config file: {exc}") from exc

    if not isinstance(config, dict):
        raise ValidationError("Invalid config structure: expected a YAML mapping")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValidationError(f"Invalid config: missing required field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValidationError("Invalid config: seed must be an integer")
    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValidationError("Invalid config: window must be a positive integer")
    if not isinstance(config["version"], str) or not config["version"]:
        raise ValidationError("Invalid config: version must be a non-empty string")

    np.random.seed(config["seed"])
    logging.info(
        "Config loaded and validated: seed=%s window=%s version=%s",
        config["seed"],
        config["window"],
        config["version"],
    )
    return config


def load_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise ValidationError(f"Input file not found: {input_path}")
    if not path.is_file():
        raise ValidationError(f"Input path is not a file: {input_path}")
    if path.stat().st_size == 0:
        raise ValidationError(f"Input file is empty: {input_path}")

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValidationError("Input CSV is empty or has no columns") from exc
    except pd.errors.ParserError as exc:
        raise ValidationError(f"Invalid CSV format: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValidationError(f"Unable to decode CSV file: {exc}") from exc
    except OSError as exc:
        raise ValidationError(f"Unable to read input file: {exc}") from exc

    if data.empty:
        raise ValidationError("Input CSV contains no rows")
    if REQUIRED_DATA_COLUMN not in data.columns:
        raise ValidationError(f"Input CSV is missing required column: {REQUIRED_DATA_COLUMN}")

    try:
        data[REQUIRED_DATA_COLUMN] = pd.to_numeric(data[REQUIRED_DATA_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValidationError("Column 'close' must contain numeric values") from exc

    logging.info("Rows loaded: %s", len(data))
    return data


def compute_signal_rate(data: pd.DataFrame, config: dict) -> float:
    window = config["window"]
    if window > len(data):
        raise ValidationError(
            f"Invalid config: window ({window}) cannot exceed row count ({len(data)})"
        )

    logging.info("Computing rolling mean for close with window=%s", window)
    rolling_mean = data[REQUIRED_DATA_COLUMN].rolling(window=window, min_periods=window).mean()

    logging.info("Generating binary signal from close versus rolling mean")
    valid_rows = rolling_mean.notna()
    signal = (data.loc[valid_rows, REQUIRED_DATA_COLUMN] > rolling_mean.loc[valid_rows]).astype(int)
    signal_rate = float(signal.mean()) if not signal.empty else 0.0

    return signal_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal MLOps-style batch signal job")
    parser.add_argument("--input", required=True, help="Path to input OHLCV CSV")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output", required=True, help="Path to metrics JSON output")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()
    version = "unknown"
    logging_ready = False
    try:
        setup_logging(args.log_file)
        logging_ready = True
        logging.info("Job start")

        config = load_config(args.config)
        version = config["version"]
        data = load_dataset(args.input)

        signal_rate = compute_signal_rate(data, config)
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        metrics = {
            "version": config["version"],
            "rows_processed": int(len(data)),
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": int(latency_ms),
            "seed": int(config["seed"]),
            "status": "success",
        }
        logging.info("Metrics summary: %s", json.dumps(metrics, sort_keys=True))
        write_metrics(args.output, metrics)

        logging.info("Job end status=success")
        print(json.dumps(metrics, indent=2))
        return 0
    except Exception as exc:
        if logging_ready:
            logging.exception("Job failed: %s", exc)
        metrics = {
            "version": version,
            "status": "error",
            "error_message": str(exc),
        }
        write_metrics(args.output, metrics)
        if logging_ready:
            logging.info("Job end status=error")
        print(json.dumps(metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
