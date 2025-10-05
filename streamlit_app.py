
import io
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="MICMAC – Overall Only (De-duplicated)", layout="wide")

# ----------------------- Helpers -----------------------
def clean_label(s: str) -> str:
    # normalize labels so small formatting differences don't create duplicates
    return " ".join(str(s).strip().split())

def consolidate_duplicates(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    If a file has duplicate labels (rows/columns with the same name),
    consolidate them so each label appears exactly once.
    """
    df = df.copy()
    df.index = df.index.map(clean_label)
    df.columns = df.columns.map(clean_label)

    # group columns first, then rows, with chosen aggregation
    agg = "mean" if how == "mean" else "sum"
    df = df.groupby(level=0, axis=1).agg(agg)  # columns
    df = df.groupby(level=0, axis=0).agg(agg)  # rows

    # keep only common row/col labels
    common = df.index.intersection(df.columns)
    df = df.loc[common, common]
    return df

def read_matrix(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read labeled square matrix (first col=labels, header=labels)."""
    if filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), header=0, index_col=0)
    else:
        df = pd.read_csv(io.BytesIO(file_bytes), header=0, index_col=0)

    # numeric & NA
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    # consolidate duplicates and force square
    df = consolidate_duplicates(df, how="mean")
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise ValueError(f"{filename}: not enough items after consolidation.")
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"{filename}: matrix must be square after consolidation. Got {df.shape}.")

    return df

def micmac_sum(A: np.ndarray, alpha: float, K: int, include_identity: bool=False) -> np.ndarray:
    """S(α,K) = (I if include_identity) + Σ_{p=1..K} α^(p-1) A^p"""
    n = A.shape[0]
    S = np.eye(n) if include_identity else np.zeros((n, n), dtype=float)
    Ap = A.copy().astype(float)
    for p in range(1, K + 1):
        S += (alpha ** (p - 1)) * Ap
        Ap = Ap @ A
    return S

def driving_dependence(S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return S.sum(axis=1), S.sum(axis=0)

def classify_quadrants(drive, depend, labels, method="Median"):
    if method == "Median":
        x_cut = float(np.median(drive)); y_cut = float(np.median(depend))
    else:
        x_cut = float(np.mean(drive));   y_cut = float(np.mean(depend))

    cats = []
    for x, y in zip(drive, depend):
        if x >= x_cut and y < y_cut:       cats.append("Driving")
        elif x >= x_cut and y >= y_cut:    cats.append("Linkage")
        elif x < x_cut and y >= y_cut:     cats.append("Dependent")
        else:                               cats.append("Autonomous")

    df = pd.DataFrame({"Item": labels, "Driving": drive, "Dependence": depend, "Category": cats})
    return df, x_cut, y_cut

# ----------------------- Sidebar -----------------------
st.sidebar.header("Upload & Overall Settings")
files = st.sidebar.file_uploader(
    "Upload one or more matrices (.csv/.xlsx)", type=["csv", "xlsx", "xls"], accept_multiple_files=True
)

alpha = st.sidebar.slider("α (damping factor)", 0.0, 1.0, 0.5, 0.01)
K = st.sidebar.slider("Max power K", 1, 30, 6, 1)
include_identity = st.sidebar.checkbox("Include Identity (I)", False)
split_rule = st.sidebar.radio("Quadrant split by", ["Median", "Mean"], index=0)

binarize = st.sidebar.checkbox("Binarize inputs", value=False)
bin_threshold = st.sidebar.slider("Binarize threshold", 0.0, 5.0, 0.5, 0.1, disabled=not binarize)

normalize_view = st.sidebar.checkbox("Normalize S̄ for display", value=False)
reorder = st.sidebar.checkbox("Reorder by Driving (desc) for display", value=True)

label_mode = st.sidebar.selectbox("Chart labels", ["All", "Top-K", "None"], index=0)
topk = st.sidebar.number_input("Top-K (if selected)", 1, 999, 20, 1)

st.title("MICMAC – OVERALL (Average across selected files)")
st.caption("This plots a single point per item—averaged across the selected files. Duplicate labels inside files are consolidated.")

if not files:
    st.info("Upload at least one file to begin.")
    st.stop()

# ----------------------- Load all files -----------------------
cases: Dict[str, Dict] = {}
skips = []
for f in files:
    try:
        df = read_matrix(f.read(), f.name)
        labels = [clean_label(x) for x in df.index.tolist()]
        A = df.values.astype(float)
        if binarize:
            A = (A >= bin_threshold).astype(float)
        cases[f.name] = {"labels": labels, "A": A}
    except Exception as e:
        skips.append(f"{f.name}: {e}")

if skips:
    st.warning("Skipped files:\n\n- " + "\n- ".join(skips))
if not cases:
    st.stop()

# choose which files to include
names = list(cases.keys())
selected = st.multiselect("Include files in OVERALL:", names, default=names)
if len(selected) < 1:
    st.info("Select at least one file.")
    st.stop()

# ----------------------- Intersection & Average -----------------------
# intersection of labels after consolidation
label_sets = [set(cases[n]["labels"]) for n in selected]
common = set.intersection(*label_sets)
if len(common) < 2:
    st.error("Not enough common items across selected files (need ≥ 2). Try a different subset.")
    st.stop()

# keep order from the first selected file
first_labels = cases[selected[0]]["labels"]
common_labels = [lbl for lbl in first_labels if lbl in common]
n = len(common_labels)

# build S for each file on aligned common label order
stack = []
for nme in selected:
    labels = cases[nme]["labels"]
    A = cases[nme]["A"]
    # map labels -> indices in this file
    idx = [labels.index(lbl) for lbl in common_labels]
    A_sub = A[np.ix_(idx, idx)]
    S = micmac_sum(A_sub, alpha=alpha, K=K, include_identity=include_identity)
    stack.append(S)

S_avg = np.mean(np.stack(stack, axis=0), axis=0)  # the ONLY matrix we’ll plot/analyze

# ----------------------- Driving/Dependence & Plot -----------------------
drive, depend = driving_dependence(S_avg)

order = np.argsort(-drive) if reorder else np.arange(n)
labels_ord = [common_labels[i] for i in order]
S_ord = S_avg[order][:, order]
drive_ord = drive[order]
depend_ord = depend[order]

S_show = S_ord / (S_ord.max() if S_ord.max() != 0 else 1.0) if normalize_view else S_ord
class_df, x_cut, y_cut = classify_quadrants(drive_ord, depend_ord, labels_ord, method=split_rule)

# optional matrix view
if st.toggle("Show overall matrix table (S̄)", value=False):
    st.dataframe(pd.DataFrame(S_show, index=labels_ord, columns=labels_ord).round(4), use_container_width=True)

st.subheader("Overall MICMAC Map (one point per item)")
fig = px.scatter(
    class_df, x="Driving", y="Dependence", text="Item", color="Category",
    title=f"Overall MICMAC (α={alpha:.2f}, K={K}) — Files: {len(selected)} | Items: {len(labels_ord)}",
    labels={"Driving": "Driving Power (row sum)", "Dependence": "Dependence Power (col sum)"}
)
fig.add_vline(x=x_cut, line_dash="dash", line_color="gray")
fig.add_hline(y=y_cut, line_dash="dash", line_color="gray")

if label_mode == "None":
    fig.update_traces(text=None)
elif label_mode == "Top-K":
    score = (class_df["Driving"] - x_cut) ** 2 + (class_df["Dependence"] - y_cut) ** 2
    keep = set(class_df.iloc[score.sort_values(ascending=False).index[:topk]]["Item"])
    fig.update_traces(text=[t if t in keep else "" for t in class_df["Item"]])

fig.update_traces(textposition="top center")
fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"- **Files included:** {len(selected)}  \n"
    f"- **Common items (unique):** {len(labels_ord)}  \n"
    f"- **Cuts:** Driving = `{x_cut:.3f}`, Dependence = `{y_cut:.3f}`"
)
