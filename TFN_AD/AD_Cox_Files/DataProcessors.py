# AD_Cox_Files/DataProcessors.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import os, pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle


# --------------------------
# 0) Config / Types
# --------------------------
WHITELIST_STRUCT_COLS = [
    "AGE", "PTEDUCAT",
    "PTGENDER_encoded", "PTETHCAT_encoded",
    "PTRACCAT_encoded", "PTMARRY_encoded",
    "MMSE", "ADAS11", "ADAS13", "ADASQ4",
    # NOTE: Years_bl is intentionally EXCLUDED here (used for time axis Δt)
]

# --- dataclass ---
@dataclass
class PatientSeq:
    pid: str
    times: np.ndarray        # [T]  (Δt from MCI for included visits)
    event: int               # 0/1  (patient-level)
    struct: np.ndarray       # [T,F]
    img_paths: List[str]     # len T
    t_event: float           # ★ Cox target time: AD time if event=1, else last observed time
    meta: Dict[str, Any]


# --------------------------
# 1) Manifest I/O
# --------------------------
def read_manifest(csv_path: str,
                  path_col: str = "path",
                  pid_col: str = "patient_id") -> pd.DataFrame:
    """
    Load per-patient PKL paths from a CSV. Each row represents one patient.
    Required cols: [path_col, pid_col]
    """
    mf = pd.read_csv(csv_path)
    if path_col not in mf or pid_col not in mf:
        raise ValueError(f"CSV must contain '{path_col}' and '{pid_col}' columns.")
    return mf[[pid_col, path_col]].rename(columns={pid_col: "pid", path_col: "pkl_path"})

# --------------------------
# 2) Per-patient sequence builder
# --------------------------
def build_patient_sequence(
    df: pd.DataFrame,
    pid: str,
    whitelist_cols: List[str] = WHITELIST_STRUCT_COLS,
    require_images: bool = True,
    img_prefix_replace: Optional[Tuple[str, str]] = ("/home/mason/ADNI_Dataset/", "../ADNI_Dataset/")
) -> Optional[PatientSeq]:
    """
    Convert a single patient's DataFrame → PatientSeq.
    - df contains per-visit rows (Columns: DX, Years_bl, image_path, whitelist_cols, ...)
    - Inclusion window: from the first MCI visit up to (but not including) the first AD/Dementia visit.
      If AD is never reached, include through the last visit.
    - Δt = Years_bl - Years_bl@MCI
    - Image path remapping and existence check are optional (require_images)
    """
    if "DX" not in df or "Years_bl" not in df:
        return None

    # 1) Sort visits within the patient by time
    df = df.sort_values("Years_bl").reset_index(drop=True)

    dx_list = df["DX"].tolist()
    if "MCI" not in dx_list:
        return None

    mci_idx = dx_list.index("MCI")

    # Index of the first AD/Dementia occurrence
    ad_idx = -1
    for i in range(mci_idx + 1, len(dx_list)):
        if dx_list[i] in ["AD", "Dementia"]:
            ad_idx = i
            break

    # Determine the index range to include
    if ad_idx != -1:
        idxs = list(range(mci_idx, ad_idx))  # up to just before AD
        event = 1
    else:
        idxs = list(range(mci_idx, len(df))) # censored
        event = 0
    if len(idxs) == 0:
        return None

    # 2) Time axis Δt
    years = df.loc[idxs, "Years_bl"].to_numpy(dtype=float)
    times = years - years[0]  # Δt from MCI
    if (np.diff(times) < -1e-8).any():
        # Guard against monotonicity violations (rare cases)
        order = np.argsort(times)
        idxs = [idxs[i] for i in order]
        years = years[order]
        times = years - years[0]

    # 3) Structured features
    missing_cols = [c for c in whitelist_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in patient df: {missing_cols}")
    struct = df.loc[idxs, whitelist_cols].to_numpy(dtype=np.float32)  # [T, F]

    # 4) Image paths
    if "image_path" not in df.columns:
        img_paths = [""] * len(idxs)
    else:
        img_paths = df.loc[idxs, "image_path"].astype(str).tolist()
        if img_prefix_replace:
            old, new = img_prefix_replace
            img_paths = [p.replace(old, new) for p in img_paths]

        if require_images:
            keep = []
            for k, p in enumerate(img_paths):
                if os.path.exists(p):
                    keep.append(k)
            # Remove timesteps without images among visits
            if len(keep) == 0:
                return None
            times  = times[keep]
            struct = struct[keep, :]
            img_paths = [img_paths[k] for k in keep]

    # --- Additional computations inside build_patient_sequence() ---
    mci_year = float(df.loc[mci_idx, "Years_bl"])
    if ad_idx != -1:
        ad_year = float(df.loc[ad_idx, "Years_bl"])  # AD time (visit itself excluded)
        t_event = ad_year - mci_year                  # ★ true event time
        idxs = list(range(mci_idx, ad_idx))          # include up to just before AD
        event = 1
    else:
        # Censoring: last observed time
        last_year = float(df.loc[len(df) - 1, "Years_bl"])
        t_event = last_year - mci_year               # ★ censoring time
        idxs = list(range(mci_idx, len(df)))
        event = 0

    # ... After computing times/struct/img_paths, return ...
    return PatientSeq(
        pid=pid,
        times=times,
        event=event,
        struct=struct,
        img_paths=img_paths,
        t_event=float(t_event),  # ★ keep
        meta={"mci_idx": int(mci_idx), "ad_idx": int(ad_idx)}
    )


def build_all_sequences(
    manifest_df: pd.DataFrame,
    whitelist_cols: List[str] = WHITELIST_STRUCT_COLS,
    require_images: bool = True,
    img_prefix_replace: Optional[Tuple[str, str]] = ("/home/mason/ADNI_Dataset/", "../ADNI_Dataset/")
) -> List[PatientSeq]:
    """
    Iterate over the manifest and build sequences for all patients.
    """
    out: List[PatientSeq] = []
    for _, row in manifest_df.iterrows():
        pid, pkl_path = str(row["pid"]), str(row["pkl_path"])
        try:
            df = pd.read_pickle(pkl_path)
            seq = build_patient_sequence(
                df=df, pid=pid, whitelist_cols=whitelist_cols,
                require_images=require_images, img_prefix_replace=img_prefix_replace
            )
            if seq is not None and len(seq.times) >= 1:
                out.append(seq)
        except Exception as e:
            print(f"[Skip] pid={pid} ({pkl_path}) → {e}")
            continue
    return out

# --------------------------
# 3) Train-only fit → transform (Preprocessor shell)
# --------------------------
class StructPreprocessor:
    """
    Preprocessor for structured features.
    - fit: Learn imputer/scaler from TRAIN visit rows only (prevents leakage)
    - transform: Apply the same transforms to each sequence's struct [T, F]
    - Options: Feature augmentation with deltas and velocity (slope)
    """
    def __init__(self, add_deltas: bool = False, add_velocity: bool = False, vel_window: int = 3):
        self.add_deltas = add_deltas
        self.add_velocity = add_velocity
        self.vel_window = vel_window
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self._f_in = None
        self._f_out = None

    def _augment(self, X: np.ndarray) -> np.ndarray:
        # X: [T, F]
        feats = [X]
        if self.add_deltas:
            dX = np.diff(X, axis=0, prepend=X[[0], :])
            feats.append(dX)
        if self.add_velocity:
            # Simple moving slope (difference/window approximation instead of linear regression)
            V = np.zeros_like(X)
            w = self.vel_window
            for t in range(1, X.shape[0]):
                t0 = max(0, t - w)
                V[t] = (X[t] - X[t0]) / max(1, (t - t0))
            feats.append(V)
        return np.concatenate(feats, axis=1)

    def fit(self, train_seqs: List[PatientSeq]) -> "StructPreprocessor":
        # Stack train visit rows → fit imputer/scaler
        mats = []
        for s in train_seqs:
            X = self._augment(s.struct.astype(np.float32))
            mats.append(X)
        X_all = np.vstack(mats) if mats else np.empty((0, len(train_seqs[0].struct[0])))
        self._f_in = X_all.shape[1] if X_all.size else None
        X_imp = self.imputer.fit_transform(X_all)
        X_std = self.scaler.fit_transform(X_imp)
        self._f_out = X_std.shape[1] if X_std.size else None
        return self

    def transform(self, seqs: List[PatientSeq]) -> List[PatientSeq]:
        out: List[PatientSeq] = []
        for s in seqs:
            X = self._augment(s.struct.astype(np.float32))
            X_imp = self.imputer.transform(X)
            X_std = self.scaler.transform(X_imp)
            out.append(PatientSeq(
                pid=s.pid, times=s.times.copy(), event=int(s.event),
                struct=X_std.astype(np.float32),
                img_paths=list(s.img_paths),
                t_event=float(s.t_event),  # ★ keep
                meta=dict(s.meta)
            ))

        return out

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            # Serialize the whole object as-is (simplest/fastest)
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "StructPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
