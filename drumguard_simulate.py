"""
drumguard_simulate.py
─────────────────────────────────────────────────────────────────────────────
DrumGuard – Desktop simulation & validation using the Mendeley dataset:
  "Vibration and Motor Current Dataset of Rolling Element Bearing
   Under Varying Speed Conditions for Fault Diagnosis"
  https://data.mendeley.com/datasets/vxkj334rzv/7
PURPOSE
  1. Mirrors the ESP32 algorithm exactly (Welford + Ring Buffer) in Python.
  2. Validates the algorithm on real healthy / faulty bearing data.
  3. Produces a labelled plot showing anomaly score vs ground-truth label.
DATASET STRUCTURE  (after download)
  Files exist in both root and dataset/ directories:
    vibration_normal_5.csv  – columns: bearingA_x, bearingA_y, bearingB_x, bearingB_y
    current_normal_5.csv    – columns: current_R, current_S, current_T
    vibration_inner_5.csv
    current_inner_5.csv
    vibration_outer_5.csv
    current_outer_5.csv
    vibration_ball_5.csv
    current_ball_5.csv
    dataset/vibration_normal_6.csv  (same structure, different speed)
    dataset/current_normal_6.csv
    ...
HOW TO RUN
  pip install numpy pandas matplotlib tqdm paho-mqtt
  python drumguard_simulate.py --dataset_dir ./dataset --calibration_files 1
ARGUMENTS
  --dataset_dir      Path to extracted dataset folder  (default: ./dataset)
  --calibration_files  Number of *normal* files used for calibration (default: 1)
  --sigma            σ multiplier for threshold            (default: 3.0)
  --ring_size        Ring buffer size                      (default: 100)
  --anomaly_ratio    Fraction of ring buffer above threshold for alert (default: 0.6)
  --plot             Show the result plot (default: True)
  --mqtt             Enable MQTT publishing (default: False)
  --broker           MQTT broker address (default: broker.hivemq.com)
  --port             MQTT broker port    (default: 1883)
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False
# ─────────────────────────── Welford's Algorithm ─────────────────────────────
class WelfordOnline:
    """
    Incrementally computes mean and variance without storing all samples.
    Identical to the C struct on the ESP32.
    """
    def __init__(self):
        self.count = 0
        self.mean  = 0.0
        self.M2    = 0.0
    def update(self, x: float):
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        delta2      = x - self.mean
        self.M2    += delta * delta2
    @property
    def variance(self) -> float:
        return self.M2 / (self.count - 1) if self.count >= 2 else 0.0
    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)
    def threshold(self, sigma: float = 3.0) -> float:
        return self.mean + sigma * self.stddev
# ─────────────────────────── Ring Buffer ─────────────────────────────────────
class RingBuffer:
    """Fixed-size circular queue of anomaly scores."""
    def __init__(self, size: int):
        self.size   = size
        self._buf   = deque(maxlen=size)
    def push(self, value: float):
        self._buf.append(value)
    def anomaly_detected(self, ratio: float = 0.6) -> bool:
        if not self._buf:
            return False
        above = sum(1 for v in self._buf if v > 1.0)
        return (above / len(self._buf)) >= ratio
    @property
    def is_full(self) -> bool:
        return len(self._buf) == self.size
# ─────────────────────────── Feature extraction (vectorized) ─────────────────
VIB_COLS = ['bearingA_x', 'bearingA_y', 'bearingB_x', 'bearingB_y']
CUR_COLS = ['current_R', 'current_S', 'current_T']
def vibration_rms_vec(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized: Euclidean magnitude of all vibration axes per row.
      sqrt(Ax² + Ay² + Bx² + By²) / 2   (average across two housings)
    """
    cols = [c for c in VIB_COLS if c in df.columns]
    vals = df[cols].values.astype(np.float64)
    return np.sqrt(np.sum(vals ** 2, axis=1)) / max(len(cols) / 2, 1)
def current_rms_vec(df: pd.DataFrame) -> np.ndarray:
    """Vectorized: RMS of three-phase motor current per row."""
    cols = [c for c in CUR_COLS if c in df.columns]
    vals = df[cols].values.astype(np.float64)
    return np.sqrt(np.mean(vals ** 2, axis=1))
# ─────────────────────────── Dataset loading ─────────────────────────────────
def load_csv(path: Path, nrows: int | None = None, skiprows: int = 0) -> pd.DataFrame:
    if skiprows > 0:
        df = pd.read_csv(path, skiprows=range(1, skiprows + 1), nrows=nrows)
    else:
        df = pd.read_csv(path, nrows=nrows)
    df.columns = [c.strip() for c in df.columns]
    return df
def find_files(dataset_dir: Path, prefix: str, condition: str) -> list[Path]:
    """Return sorted list of CSV paths matching prefix_condition_*.csv in dataset_dir."""
    pattern = f"{prefix}_{condition}_*.csv"
    return sorted(dataset_dir.glob(pattern))
# ─────────────────────────── Core simulation ─────────────────────────────────
def run_simulation(
    dataset_dir: Path,
    calibration_files: int = 1,
    sigma: float = 3.0,
    ring_size: int = 100,
    anomaly_ratio: float = 0.6,
    max_rows: int | None = None,
):
    """
    1. Calibrate on `calibration_files` normal vibration+current files.
    2. Run inference on normal, inner-race-fault, outer-race-fault, and ball-fault data.
    3. Return a dict with per-sample scores and ground-truth labels.
    max_rows: if set, only read this many rows per CSV (speeds up large files).
    """
    # ── 1. Calibration (vectorized) ──────────────────────────────────────────
    print("\n=== Phase 1: Calibration (Welford's Algorithm) ===")
    vib_welford = WelfordOnline()
    cur_welford = WelfordOnline()
    healthy_vib_files = find_files(dataset_dir, "vibration", "normal")
    healthy_cur_files = find_files(dataset_dir, "current",   "normal")
    if not healthy_vib_files:
        sys.exit(
            f"[ERROR] No files found matching 'vibration_normal_*.csv' in {dataset_dir}.\n"
            "        Download the dataset and point --dataset_dir at the extracted folder."
        )
    cal_vib = healthy_vib_files[:calibration_files]
    cal_cur = healthy_cur_files[:calibration_files]
    total_cal_samples = 0
    iterator = zip(cal_vib, cal_cur)
    if TQDM:
        iterator = tqdm(list(iterator), desc="Calibrating", unit="file")
    for vf, cf in iterator:
        print(f"    Loading {vf.name} ...")
        vdf = load_csv(vf, nrows=max_rows)
        print(f"    Loading {cf.name} ...")
        cdf = load_csv(cf, nrows=max_rows)
        n   = min(len(vdf), len(cdf))
        # Vectorized feature extraction
        vib_arr = vibration_rms_vec(vdf.iloc[:n])
        cur_arr = current_rms_vec(cdf.iloc[:n])
        # Bulk Welford update
        for val in vib_arr:
            vib_welford.update(val)
        for val in cur_arr:
            cur_welford.update(val)
        total_cal_samples += n
    vib_thresh = vib_welford.threshold(sigma)
    cur_thresh = cur_welford.threshold(sigma)
    print(f"  Calibrated on {total_cal_samples:,} samples from {len(cal_vib)} files.")
    print(f"  Vibration  mu={vib_welford.mean:.5f} g   sigma={vib_welford.stddev:.5f}  -> threshold={vib_thresh:.5f}")
    print(f"  Current    mu={cur_welford.mean:.5f} A   sigma={cur_welford.stddev:.5f}  -> threshold={cur_thresh:.5f}")
    # ── 2. Inference (vectorized) ────────────────────────────────────────────
    print("\n=== Phase 2: Inference ===")
    all_scores   = []
    all_labels   = []
    all_vib      = []
    all_cur      = []
    conditions = [
        ("normal",   0, "green"),
        ("inner",    1, "orange"),
        ("outer",    2, "red"),
        ("ball",     3, "purple"),
    ]
    for condition, label, _ in conditions:
        vib_files = find_files(dataset_dir, "vibration", condition)
        cur_files = find_files(dataset_dir, "current",   condition)
        skiprows_inference = 0
        if condition == "normal":
            if len(vib_files) <= calibration_files:
                # Keep the file but skip the calibration samples
                skiprows_inference = max_rows if max_rows is not None else 0
            else:
                vib_files = vib_files[calibration_files:]
                cur_files = cur_files[calibration_files:]
        if not vib_files:
            print(f"  [SKIP] No inference files for condition '{condition}'.")
            continue
        file_pairs = list(zip(vib_files, cur_files))
        if TQDM:
            file_pairs = tqdm(file_pairs, desc=f"  Condition: {condition}", unit="file")
        for vf, cf in file_pairs:
            print(f"    Loading {vf.name} ...")
            vdf = load_csv(vf, nrows=max_rows, skiprows=skiprows_inference)
            print(f"    Loading {cf.name} ...")
            cdf = load_csv(cf, nrows=max_rows, skiprows=skiprows_inference)
            n   = min(len(vdf), len(cdf))
            # Vectorized feature extraction
            vib_arr = vibration_rms_vec(vdf.iloc[:n])
            cur_arr = current_rms_vec(cdf.iloc[:n])
            # Vectorized score computation
            vib_scores = vib_arr / vib_thresh if vib_thresh > 0 else np.zeros(n)
            cur_scores = cur_arr / cur_thresh if cur_thresh > 0 else np.zeros(n)
            scores     = np.maximum(vib_scores, cur_scores)
            all_scores.append(scores)
            all_labels.append(np.full(n, label, dtype=int))
            all_vib.append(vib_arr)
            all_cur.append(cur_arr)
        # Per-condition summary
        cond_scores = np.concatenate([s for s, l in zip(all_scores, all_labels) if l[0] == label]) if all_scores else np.array([])
        n_cond = len(cond_scores)
        n_above = int(np.sum(cond_scores > 1.0)) if n_cond > 0 else 0
        print(f"  {condition:<10}  samples={n_cond:>10,}  above_threshold={n_above:>10,}  rate={n_above/max(n_cond,1)*100:.1f}%")
    # Concatenate all results
    if not all_scores:
        sys.exit("[ERROR] No inference data processed.")
    scores_arr = np.concatenate(all_scores)
    labels_arr = np.concatenate(all_labels)
    vib_arr    = np.concatenate(all_vib)
    cur_arr    = np.concatenate(all_cur)
    n_total    = len(scores_arr)
    # Ring buffer pass for alert flags (must be sequential)
    print(f"  Running ring buffer alert detection on {n_total:,} samples ...")
    ring = RingBuffer(ring_size)
    alert_flags = np.zeros(n_total, dtype=int)
    for i in range(n_total):
        ring.push(float(scores_arr[i]))
        if ring.anomaly_detected(anomaly_ratio):
            alert_flags[i] = 1
    # Generate ISO timestamps
    base_ts = datetime.now(timezone.utc)
    iso_timestamps = [base_ts.isoformat()] * n_total  # Same base; unique per-sample not needed for bulk
    results = {
        "timestamps":     np.arange(n_total).tolist(),
        "iso_timestamps": iso_timestamps,
        "anomaly_scores": scores_arr.tolist(),
        "labels":         labels_arr.tolist(),
        "alert_flags":    alert_flags.tolist(),
        "vib_values":     vib_arr.tolist(),
        "cur_values":     cur_arr.tolist(),
    }
    # Final alert summary
    for condition, label, _ in conditions:
        mask    = labels_arr == label
        n_cond  = int(np.sum(mask))
        n_alert = int(np.sum(alert_flags[mask]))
        if n_cond > 0:
            print(f"  {condition:<10}  samples={n_cond:>10,}  alerts={n_alert:>10,}  alert_rate={n_alert/max(n_cond,1)*100:.1f}%")
    return results, vib_thresh, cur_thresh, vib_welford, cur_welford
# ─────────────────────────── Metrics ─────────────────────────────────────────
def compute_metrics(results: dict) -> dict:
    labels = np.array(results["labels"])
    alerts = np.array(results["alert_flags"])
    # Binary: fault = label > 0
    ground_truth = (labels > 0).astype(int)
    TP = int(np.sum((alerts == 1) & (ground_truth == 1)))
    TN = int(np.sum((alerts == 0) & (ground_truth == 0)))
    FP = int(np.sum((alerts == 1) & (ground_truth == 0)))
    FN = int(np.sum((alerts == 0) & (ground_truth == 1)))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / len(ground_truth) if len(ground_truth) > 0 else 0.0
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)
# ─────────────────────────── Plot ────────────────────────────────────────────
def plot_results(results: dict, vib_thresh: float, cur_thresh: float):
    if not HAS_PLOT:
        print("[PLOT] matplotlib not available. Install it to see the plot.")
        return
    scores = np.array(results["anomaly_scores"])
    labels = np.array(results["labels"])
    alerts = np.array(results["alert_flags"])
    t      = np.arange(len(scores))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("DrumGuard – Anomaly Score vs Ground Truth", fontsize=14, fontweight="bold")
    # ── Anomaly score ──────────────────────────────────────────────────────
    ax = axes[0]
    colours = {0: "#4caf50", 1: "#ff9800", 2: "#f44336", 3: "#9c27b0"}
    label_names = {0: "Normal", 1: "Inner-race fault", 2: "Outer-race fault", 3: "Ball fault"}
    for lbl, colour in colours.items():
        mask = labels == lbl
        ax.scatter(t[mask], scores[mask], c=colour, s=0.3, alpha=0.5,
                   label=label_names[lbl])
    ax.axhline(y=1.0, color="black", linewidth=1.2, linestyle="--", label="Threshold (μ+3σ)")
    ax.set_ylabel("Normalised Anomaly Score")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", markerscale=8)
    ax.grid(axis="y", alpha=0.3)
    # ── Alert flags ────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(t, alerts, step="post", color="#f44336", alpha=0.7, label="Alert fired")
    ax2.fill_between(t, (labels > 0).astype(int), step="post",
                     color="#ff9800", alpha=0.4, label="Ground truth fault")
    ax2.set_ylabel("Alert / GT")
    ax2.set_yticks([0, 1])
    ax2.legend(loc="upper left")
    # ── Vibration RMS ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.plot(t, results["vib_values"], color="#2196f3", linewidth=0.4, alpha=0.7)
    ax3.axhline(y=vib_thresh, color="black", linewidth=1, linestyle=":", label=f"Vib threshold {vib_thresh:.4f}")
    ax3.set_ylabel("Vibration RMS (g)")
    ax3.set_xlabel("Sample index")
    ax3.legend(loc="upper left")
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path("drumguard_results.png")
    plt.savefig(out, dpi=150)
    print(f"\n[PLOT] Saved to {out.resolve()}")
    plt.show()
# ─────────────────────────── MQTT Publishing ─────────────────────────────────
def publish_mqtt(broker: str, port: int, results: dict,
                 vib_thresh: float, cur_thresh: float):
    """Publish simulation results to MQTT with ISO timestamps."""
    if not HAS_MQTT:
        print("[MQTT] paho-mqtt not installed. Skipping MQTT publish.")
        return
    client = mqtt.Client(client_id="drumguard-simulator")
    try:
        client.connect(broker, port, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")
        return
    label_names = {0: "normal", 1: "inner_fault", 2: "outer_fault", 3: "ball_fault"}
    n = len(results["timestamps"])
    # Publish a summary every N samples to avoid flooding
    step = max(1, n // 100)
    print(f"\n[MQTT] Publishing {n // step} summary messages to {broker}:{port} ...")
    for i in range(0, n, step):
        iso_ts = results["iso_timestamps"][i]
        score  = results["anomaly_scores"][i]
        vib    = results["vib_values"][i]
        cur    = results["cur_values"][i]
        label  = results["labels"][i]
        alert  = results["alert_flags"][i]
        # ── drumguard/data payload ──
        data_payload = {
            "timestamp":     iso_ts,
            "device_id":     "drumguard-sim",
            "vibration_rms": round(vib, 6),
            "current_amps":  round(cur, 6),
            "anomaly_score": round(score, 4),
            "status":        label_names.get(label, "unknown"),
            "calibrated":    True,
        }
        client.publish("drumguard/data", json.dumps(data_payload), qos=1)
        # ── drumguard/alert payload (only when alert fires) ──
        if alert:
            alert_payload = {
                "timestamp":     iso_ts,
                "device_id":     "drumguard-sim",
                "vibration_rms": round(vib, 6),
                "current_amps":  round(cur, 6),
                "anomaly_score": round(score, 4),
                "message":       f"Anomaly detected – condition={label_names.get(label, 'unknown')}",
            }
            client.publish("drumguard/alert", json.dumps(alert_payload), qos=1)
    # ── drumguard/status payload ──
    status_payload = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "device_id":  "drumguard-sim",
        "status":     "simulation_complete",
        "total_samples": n,
        "vib_threshold": round(vib_thresh, 6),
        "cur_threshold": round(cur_thresh, 6),
    }
    client.publish("drumguard/status", json.dumps(status_payload), qos=1)
    client.loop_stop()
    client.disconnect()
    print("[MQTT] Done.")
# ─────────────────────────── CLI ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DrumGuard - dataset simulation")
    parser.add_argument("--dataset_dir",       type=Path,  default=Path("./dataset"))
    parser.add_argument("--calibration_files", type=int,   default=1)
    parser.add_argument("--sigma",             type=float, default=3.0)
    parser.add_argument("--ring_size",         type=int,   default=100)
    parser.add_argument("--anomaly_ratio",     type=float, default=0.6)
    parser.add_argument("--max_rows",          type=int,   default=50000,
                        help="Max rows to read per CSV file (default: 50000). Use 0 for all rows.")
    parser.add_argument("--no_plot",           action="store_true")
    parser.add_argument("--mqtt",              action="store_true",
                        help="Publish results to MQTT broker")
    parser.add_argument("--broker",            default="broker.hivemq.com")
    parser.add_argument("--port",              type=int, default=1883)
    args = parser.parse_args()
    if not args.dataset_dir.exists():
        sys.exit(f"[ERROR] Dataset directory not found: {args.dataset_dir}\n"
                 "        Download and extract the dataset, then pass --dataset_dir <path>.")
    row_limit = args.max_rows if args.max_rows > 0 else None
    results, vib_thresh, cur_thresh, vib_w, cur_w = run_simulation(
        dataset_dir        = args.dataset_dir,
        calibration_files  = args.calibration_files,
        sigma              = args.sigma,
        ring_size          = args.ring_size,
        anomaly_ratio      = args.anomaly_ratio,
        max_rows           = row_limit,
    )
    metrics = compute_metrics(results)
    print("\n=== Classification Metrics ===")
    print(f"  Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision  : {metrics['precision']*100:.2f}%")
    print(f"  Recall     : {metrics['recall']*100:.2f}%")
    print(f"  F1-Score   : {metrics['f1']*100:.2f}%")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  FP={metrics['FP']}  FN={metrics['FN']}")
    if not args.no_plot:
        plot_results(results, vib_thresh, cur_thresh)
    if args.mqtt:
        publish_mqtt(args.broker, args.port, results, vib_thresh, cur_thresh)
if __name__ == "__main__":
    main()
