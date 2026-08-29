"""Shared HAR pipeline: loading, resampling, windowing, feature extraction,
discretization, unknown-activity detection and trend/forecast helpers.

Imported by both 01_research_notebook.ipynb and 02_inference_notebook.ipynb so the
exact same preprocessing is used for training and for inference.
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "activities_measurments")
DEFAULT_TEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_test_file", "Raw Data.csv")

TARGET_RATE_HZ = 50.0
WINDOW_SEC = 2.56          # 128 samples @ 50Hz -- standard HAR (UCI-HAR) window length
OVERLAP = 0.5
WINDOW_SAMPLES = int(round(WINDOW_SEC * TARGET_RATE_HZ))
STEP_SAMPLES = int(round(WINDOW_SAMPLES * (1 - OVERLAP)))

AXES = ["vertical", "horizontal", "mag"]
GRAVITY_WINDOW_SEC = 1.0  # low-pass window for estimating gravity direction
MOVEMENT_ACTIVITIES = {"walking", "running", "stairs_up", "stairs_down"}
KNOWN_ACTIVITIES = ["walking", "running", "stairs_up", "stairs_down", "still"]
UNKNOWN_LABEL = "unknown"
DISPLAY_NAMES = {
    "walking": "Walking",
    "running": "Running",
    "stairs_up": "Stairs up",
    "stairs_down": "Stairs down",
    "still": "Still",
    UNKNOWN_LABEL: "Unknown",
}

PHYPHOX_COLUMN_MAP = {
    "Time (s)": "time",
    "Acceleration x (m/s^2)": "x",
    "Acceleration y (m/s^2)": "y",
    "Acceleration z (m/s^2)": "z",
    "Absolute acceleration (m/s^2)": "mag",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_recording(path):
    """Load one Phyphox 'Acceleration with g' export (xls/xlsx or csv) into a
    DataFrame with columns: time, x, y, z, mag."""
    df = None
    try:
        df = pd.read_excel(path)
    except Exception:
        df = None
    if df is None or df.shape[1] < 4:
        df = pd.read_csv(path)

    df = df.rename(columns=PHYPHOX_COLUMN_MAP)
    missing = {"time", "x", "y", "z"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {missing}")
    if "mag" not in df.columns:
        df["mag"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)

    df = df[["time", "x", "y", "z", "mag"]].dropna().sort_values("time").reset_index(drop=True)
    return df


# Filenames are free-form (e.g. "Walking_Raw_Data.csv", "Climbing_Stairs_Up_Raw_Data.csv"),
# so files are matched to an activity by keyword-in-filename rather than an exact prefix --
# this keeps the loader working regardless of which teammate named the file what.
_ACTIVITY_KEYWORDS = {
    "walking": ["walking"],
    "running": ["running"],
    "stairs_up": ["stairs_up", "stairsup", "climbing_stairs_up", "stairs_up", "up"],
    "stairs_down": ["stairs_down", "stairsdown", "climbing_stairs_down", "down"],
    "still": ["still", "standing"],
}


def _matches_activity(filename, activity):
    name = filename.lower()
    if "stairs" in name or "climbing" in name:
        # disambiguate up/down before falling back to the generic "up"/"down" keywords
        if activity == "stairs_up":
            return "up" in name
        if activity == "stairs_down":
            return "down" in name
        return False
    return any(kw in name for kw in _ACTIVITY_KEYWORDS[activity] if kw not in ("up", "down"))


def load_activity_files(data_dir=DATA_DIR):
    """Scan activities_measurments/ for csv|xls|xlsx files and return
    {activity_name: [DataFrame, ...]} -- multiple recordings per activity
    (e.g. from different teammates) are all picked up, matched by keyword
    in the filename rather than an exact name."""
    all_files = sorted(
        glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "*.xls"))
        + glob.glob(os.path.join(data_dir, "*.xlsx"))
    )
    recordings = {activity: [] for activity in KNOWN_ACTIVITIES}
    for path in all_files:
        filename = os.path.basename(path)
        for activity in KNOWN_ACTIVITIES:
            if _matches_activity(filename, activity):
                recordings[activity].append(load_recording(path))
                break
    return recordings


# ---------------------------------------------------------------------------
# Resampling + windowing
# ---------------------------------------------------------------------------

def resample_uniform(df, rate_hz=TARGET_RATE_HZ):
    """Linearly interpolate onto a fixed-rate time grid. Phyphox's native
    sampling rate depends on the phone (the sample file measured ~478Hz), so
    every recording is normalized to the same rate before feature extraction."""
    t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
    n = int(np.floor((t1 - t0) * rate_hz)) + 1
    grid = t0 + np.arange(n) / rate_hz
    out = pd.DataFrame({"time": grid})
    for col in ["x", "y", "z", "mag"]:
        out[col] = np.interp(grid, df["time"], df[col])
    return out


def add_orientation_invariant_axes(df, gravity_window_sec=GRAVITY_WINDOW_SEC, rate_hz=TARGET_RATE_HZ):
    """Decompose x/y/z into a gravity-aligned 'vertical' component (signed;
    gravity + linear vertical acceleration) and a 'horizontal' magnitude
    (motion perpendicular to gravity -- forward/sideways, direction
    discarded). Raw x/y/z encode phone *orientation*, not just motion: a
    recording held at a different fixed tilt gets wildly different x/y/z
    means even for identical activities. Vertical/horizontal are invariant
    to how the phone is held, so they generalize across recordings/phones.

    Gravity direction is estimated with a rolling low-pass filter (longer
    than a stride cycle) over the whole recording, not per-window, so the
    estimate stays stable across window boundaries and can still track slow
    orientation drift within one continuous recording."""
    win = max(int(round(gravity_window_sec * rate_hz)), 1)
    gx = df["x"].rolling(win, center=True, min_periods=1).mean()
    gy = df["y"].rolling(win, center=True, min_periods=1).mean()
    gz = df["z"].rolling(win, center=True, min_periods=1).mean()
    g_norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).replace(0, np.nan)
    ux, uy, uz = gx / g_norm, gy / g_norm, gz / g_norm

    vertical = df["x"] * ux + df["y"] * uy + df["z"] * uz
    horiz_x = df["x"] - vertical * ux
    horiz_y = df["y"] - vertical * uy
    horiz_z = df["z"] - vertical * uz
    horizontal = np.sqrt(horiz_x ** 2 + horiz_y ** 2 + horiz_z ** 2)

    out = df.copy()
    out["vertical"] = vertical.bfill().ffill()
    out["horizontal"] = horizontal.fillna(0.0)
    return out


def make_windows(df, window_samples=WINDOW_SAMPLES, step_samples=STEP_SAMPLES):
    """Yield (start_time, end_time, window_df) slices of a resampled recording."""
    n = len(df)
    for start in range(0, max(n - window_samples, 0) + 1, step_samples):
        chunk = df.iloc[start:start + window_samples]
        if len(chunk) < window_samples:
            break
        yield chunk["time"].iloc[0], chunk["time"].iloc[-1], chunk


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_PER_AXIS_FEATURES = ["mean", "std", "min", "max", "range", "rms", "mad", "zcr", "sma", "dom_freq", "spec_energy"]
FEATURE_NAMES = [f"{axis}_{feat}" for axis in AXES for feat in _PER_AXIS_FEATURES] + ["corr_vert_horiz"]


def _axis_features(values, rate_hz=TARGET_RATE_HZ):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    std = values.std()
    vmin, vmax = values.min(), values.max()
    rms = np.sqrt(np.mean(values ** 2))
    mad = np.mean(np.abs(values - mean))
    centered = values - mean
    zcr = np.sum(np.diff(np.sign(centered)) != 0) / max(n - 1, 1)
    sma = np.mean(np.abs(values))

    fft_mag = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(n, d=1.0 / rate_hz)
    if len(fft_mag) > 1:
        dom_idx = 1 + np.argmax(fft_mag[1:])  # skip DC component
        dom_freq = freqs[dom_idx]
    else:
        dom_freq = 0.0
    spec_energy = np.sum(fft_mag ** 2) / n

    return [mean, std, vmin, vmax, vmax - vmin, rms, mad, zcr, sma, dom_freq, spec_energy]


def extract_features(window_df):
    """window_df -> dict of ~47 named features (per-axis stats + cross-axis correlation)."""
    feats = {}
    for axis in AXES:
        values = window_df[axis].to_numpy()
        for name, val in zip(_PER_AXIS_FEATURES, _axis_features(values)):
            feats[f"{axis}_{name}"] = val

    vertical, horizontal = window_df["vertical"].to_numpy(), window_df["horizontal"].to_numpy()
    with np.errstate(invalid="ignore"):
        feats["corr_vert_horiz"] = np.nan_to_num(np.corrcoef(vertical, horizontal)[0, 1])
    return feats


def featurize_recording(df, label=None):
    """Resample + gravity-align + window + extract features for one recording.
    Returns a DataFrame of features, one row per window, plus start/end time
    columns and (if label given) an 'activity' column."""
    resampled = resample_uniform(df)
    resampled = add_orientation_invariant_axes(resampled)
    rows, starts, ends = [], [], []
    for start_t, end_t, window in make_windows(resampled):
        rows.append(extract_features(window))
        starts.append(start_t)
        ends.append(end_t)
    feats_df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    feats_df["window_start"] = starts
    feats_df["window_end"] = ends
    # Windows overlap (OVERLAP=0.5), so window_end already reaches into the
    # next window. For duration/segment bookkeeping each window should only
    # own its non-overlapping slice, ending where the *next* window starts;
    # only the last window of the recording has no successor and keeps its
    # true end. Using window_start/window_end (instead of this) for segment
    # boundaries double-counts the overlap at every segment transition.
    feats_df["slice_end"] = feats_df["window_start"].shift(-1).fillna(feats_df["window_end"])
    if label is not None:
        feats_df["activity"] = label
    return feats_df


def build_training_table(recordings_by_activity):
    """recordings_by_activity: {activity: [DataFrame, ...]} -> one big features
    DataFrame with an 'activity' label column."""
    parts = []
    for activity, recs in recordings_by_activity.items():
        for rec in recs:
            parts.append(featurize_recording(rec, label=activity))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Discretization (for the from-scratch, categorical-only ID3 / Naive Bayes)
# ---------------------------------------------------------------------------

def fit_discretizer(features_df, n_bins=4, feature_names=FEATURE_NAMES):
    """Quantile-bin edges per feature, fit on training data only."""
    bin_edges = {}
    for col in feature_names:
        edges = np.quantile(features_df[col], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)  # guard against constant/near-constant features
        bin_edges[col] = edges.tolist()
    return bin_edges


def apply_discretizer(features_df, bin_edges, feature_names=FEATURE_NAMES):
    """Turn continuous features into categorical bin labels ('Q1', 'Q2', ...)
    using previously-fit bin edges, returning a list of dict rows ready for
    the from-scratch ID3 / Naive Bayes (which expect discrete attribute values)."""
    rows = []
    cols = {}
    for col in feature_names:
        edges = np.array(bin_edges[col])
        if len(edges) < 2:
            cols[col] = ["Q1"] * len(features_df)
            continue
        idx = np.clip(np.digitize(features_df[col].to_numpy(), edges[1:-1]), 0, len(edges) - 2)
        cols[col] = [f"Q{i + 1}" for i in idx]
    for i in range(len(features_df)):
        rows.append({col: cols[col][i] for col in feature_names})
    return rows


def discrete_attr_values(n_bins=4):
    return {col: [f"Q{i + 1}" for i in range(n_bins)] for col in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Unknown-activity detection (k-means distance rejection)
# ---------------------------------------------------------------------------

def unknown_distances(scaled_features, kmeans, cluster_to_activity):
    """Distance from each row to the centroid of the cluster matching its
    *predicted* activity (not just nearest centroid), used both to pick the
    rejection threshold during training and to score new windows at inference."""
    cluster_ids = kmeans.predict(scaled_features)
    centroids = kmeans.cluster_centers_
    dists = np.linalg.norm(scaled_features - centroids[cluster_ids], axis=1)
    activities = np.array([cluster_to_activity[c] for c in cluster_ids])
    return activities, dists


def fit_unknown_threshold(scaled_features, true_labels, kmeans, cluster_to_activity, percentile=99):
    _, dists = unknown_distances(scaled_features, kmeans, cluster_to_activity)
    return float(np.percentile(dists, percentile))


# ---------------------------------------------------------------------------
# Gaussian Process regression: smoothing / trend classification / forecasting
# ---------------------------------------------------------------------------

def fit_gpr(times, values):
    times = np.asarray(times, dtype=float).reshape(-1, 1)
    values = np.asarray(values, dtype=float)
    t_mean, t_std = times.mean(), times.std() + 1e-9
    v_mean, v_std = values.mean(), values.std() + 1e-9
    x = (times - t_mean) / t_std
    y = (values - v_mean) / v_std
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=2)
    gpr.fit(x, y)
    return {"gpr": gpr, "t_mean": t_mean, "t_std": t_std, "v_mean": v_mean, "v_std": v_std}


def gpr_predict(model, times):
    times = np.asarray(times, dtype=float).reshape(-1, 1)
    x = (times - model["t_mean"]) / model["t_std"]
    mean, std = model["gpr"].predict(x, return_std=True)
    return mean * model["v_std"] + model["v_mean"], std * model["v_std"]


def classify_trend(times, values, t_threshold=2.0):
    """Classify a movement segment's speed-proxy signal as Accelerate /
    Decelerate / Stable speed (the project's required 'Remark' values) using
    ordinary least-squares ("Gaussian") linear regression: the slope is only
    trusted as a real trend if it clears a significance test (|slope| more
    than `t_threshold` standard errors from zero) -- otherwise it's noise and
    the segment is called stable. See `forecast_signal` for the kernelized
    Gaussian Process regressor used for actual signal forecasting."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(times)
    if n < 3 or np.std(values) < 1e-9:
        return "Stable speed"

    slope, intercept = np.polyfit(times, values, 1)
    residuals = values - (slope * times + intercept)
    dof = n - 2
    residual_var = np.sum(residuals ** 2) / dof
    t_var = np.sum((times - times.mean()) ** 2)
    slope_se = np.sqrt(residual_var / t_var) + 1e-12

    if abs(slope) / slope_se < t_threshold:
        return "Stable speed"
    return "Accelerate" if slope > 0 else "Decelerate"


def forecast_signal(times, values, horizon_sec, n_points=20):
    """Fit a Gaussian Process (RBF kernel) to a signal and forecast its
    behavior `horizon_sec` beyond the last observed time -- the project's
    'predict the future behavior of the sensor signal' requirement."""
    model = fit_gpr(times, values)
    last_t = np.asarray(times, dtype=float)[-1]
    future_times = last_t + np.linspace(1e-6, horizon_sec, n_points)
    mean, std = gpr_predict(model, future_times)
    return future_times, mean, std


# ---------------------------------------------------------------------------
# Segment building (window labels -> contiguous Activity/Time/Remark rows)
# ---------------------------------------------------------------------------

def majority_smooth(labels, k=3):
    """Rolling majority-vote filter over predicted window labels to remove
    single-window flicker before merging into segments."""
    labels = list(labels)
    n = len(labels)
    out = []
    half = k // 2
    for i in range(n):
        window = labels[max(0, i - half):min(n, i + half + 1)]
        # iterate the list (not a set) so a tied vote deterministically picks
        # the first-occurring label -- str hashing (and thus set iteration
        # order) is randomized per-process, which made ties non-reproducible
        out.append(max(window, key=window.count))
    return out


def merge_segments(labels, window_starts, window_ends):
    """Merge consecutive windows sharing a label into one segment:
    [{'activity', 'start', 'end', 'duration'}, ...]"""
    segments = []
    cur_label, cur_start, cur_end = None, None, None
    for label, start, end in zip(labels, window_starts, window_ends):
        if label == cur_label:
            cur_end = end
        else:
            if cur_label is not None:
                segments.append({"activity": cur_label, "start": cur_start, "end": cur_end})
            cur_label, cur_start, cur_end = label, start, end
    if cur_label is not None:
        segments.append({"activity": cur_label, "start": cur_start, "end": cur_end})
    for seg in segments:
        seg["duration"] = seg["end"] - seg["start"]
    return segments


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def classify_windows(features_df, artifacts):
    """Full per-window inference: Random Forest activity prediction,
    overridden to 'unknown' when the point's distance to its nearest
    k-means centroid (the same distance used to fit `unknown_threshold`
    during training) is too large to trust any known-activity label."""
    X = features_df[FEATURE_NAMES]
    preds = artifacts["random_forest"].predict(X)
    X_scaled = artifacts["scaler"].transform(X)
    _, dists = unknown_distances(X_scaled, artifacts["kmeans"], artifacts["cluster_to_activity"])
    return [
        UNKNOWN_LABEL if dist > artifacts["unknown_threshold"] else pred
        for pred, dist in zip(preds, dists)
    ]


def segments_to_table(segments, features_df):
    """Turn merged segments + their windows' features into the required
    Activity / Time (seconds) / Remark output table. Only movement
    activities get a trend remark; Still/Unknown get a blank one, matching
    the project's example table."""
    mid_times = (features_df["window_start"] + features_df["window_end"]) / 2
    rows = []
    # Round durations cumulatively (largest-remainder style) rather than
    # independently per segment: rounding each of ~dozens of segments to the
    # nearest second on its own lets the individual +/-0.5s errors drift in
    # one direction, so the reported total can end up several seconds off
    # from the recording's real length even though each row looks fine.
    cumulative_true = 0.0
    cumulative_rounded = 0
    for seg in segments:
        mask = (mid_times >= seg["start"]) & (mid_times <= seg["end"])
        remark = ""
        if seg["activity"] in MOVEMENT_ACTIVITIES:
            remark = classify_trend(mid_times[mask].to_numpy(), features_df.loc[mask, "mag_rms"].to_numpy())
        cumulative_true += seg["duration"]
        new_cumulative_rounded = int(round(cumulative_true))
        rows.append({
            "Activity": DISPLAY_NAMES.get(seg["activity"], seg["activity"]),
            "Time (seconds)": new_cumulative_rounded - cumulative_rounded,
            "Remark": remark,
        })
        cumulative_rounded = new_cumulative_rounded
    return pd.DataFrame(rows, columns=["Activity", "Time (seconds)", "Remark"])


def save_artifacts(artifacts, artifacts_dir=ARTIFACTS_DIR):
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(artifacts, os.path.join(artifacts_dir, "model_bundle.joblib"))


def load_artifacts(artifacts_dir=ARTIFACTS_DIR):
    return joblib.load(os.path.join(artifacts_dir, "model_bundle.joblib"))
