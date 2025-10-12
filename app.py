
# -*- coding: utf-8 -*-
"""
Streamlit AHP (Steps 1–2 visible, Steps 3–5 hidden)
- Experts only enter pairwise comparisons for Importance and Likelihood (Saaty 1–9).
- App computes AHP weights, consistency, and combined risk internally (hidden).
- Output is an Excel file with raw and % weights, rankings, consistency, and aggregated matrices.
- No JSON uploads/downloads involved.

Run: streamlit run ahp_steps1_2_export_excel.py
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------ Fixed transmission routes ------------------------ #
ROUTES = [
    "Introduction of virus through introduction of day old chick",
    "Introduction of virus trough animal transport vehicle / equipment",
    "Introduction of virus through professional visitors at the farm (vet, truck driver, catching team)",
    "Introduction of virus through feed trucks",
    "Introduction of virus through vermin or birds",
    "Introduction of virus through water",
    "Introduction of virus through truck of the rendering company",
    "Introduction of virus through shared equipment",
    "Introduction of virus through other farm animals",
    "Introduction of virus through the air over short distance (<1000m)",
    "Introduction of virus through spreading of manure originating from infected farms in close vicinity of the farm",
]
N = len(ROUTES)

# ------------------------ Saaty Random Index (RI) -------------------------- #
SAATY_RI = {
    1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
    9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59
}

# --------------------------- AHP core functions ---------------------------- #
def eigen_priority(M: np.ndarray):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()
    return w, vals[idx].real

def consistency_ratio(M: np.ndarray):
    n = M.shape[0]
    w, lam = eigen_priority(M)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return CR, CI, lam, w

def matrix_from_upper_triangle(n: int, pairs: dict[tuple[int,int], float]) -> np.ndarray:
    """Build a reciprocal AHP matrix from values given for i<j."""
    M = np.ones((n, n), dtype=float)
    for (i, j), v in pairs.items():
        if i == j: 
            continue
        v = float(v)
        if v <= 0:
            raise ValueError("Saaty values must be > 0")
        M[i, j] = v
        M[j, i] = 1.0 / v
    return M

# ----------------------------- UI helpers --------------------------------- #
SAATY_SCALE = {
    "1 (equal)": 1,
    "2 (between 1–3)": 2,
    "3 (moderate)": 3,
    "4 (between 3–5)": 4,
    "5 (strong)": 5,
    "6 (between 5–7)": 6,
    "7 (very strong)": 7,
    "8 (between 7–9)": 8,
    "9 (extreme)": 9,
}

def saaty_selectbox(key_prefix: str, label: str, default=1):
    options = list(SAATY_SCALE.keys())
    default_idx = options.index(next(k for k,v in SAATY_SCALE.items() if v==default))
    choice = st.selectbox(label, options, index=default_idx, key=key_prefix)
    return SAATY_SCALE[choice]

def pairwise_form(criterion_key: str, title: str):
    st.subheader(title)
    st.caption("Pick a value **>1** if the left route is more important/likely than the right; "
               "use the *Reciprocal* toggle to flip direction (1/value).")

    pairs = {}
    cols = st.columns([2, 2, 1.2, 1.2, 2])  # headers
    with cols[0]: st.markdown("**Left route**")
    with cols[1]: st.markdown("**Right route**")
    with cols[2]: st.markdown("**Saaty**")
    with cols[3]: st.markdown("**Reciprocal?**")
    with cols[4]: st.markdown("**Preview**")

    for i in range(N - 1):
        for j in range(i + 1, N):
            row = st.columns([2, 2, 1.2, 1.2, 2])
            with row[0]:
                st.write(f"{i+1}. {ROUTES[i]}")
            with row[1]:
                st.write(f"{j+1}. {ROUTES[j]}")
            with row[2]:
                val = saaty_selectbox(f"{criterion_key}_saaty_{i}_{j}", "", default=1)
            with row[3]:
                recip = st.checkbox(" ", key=f"{criterion_key}_rec_{i}_{j}", value=False)
            with row[4]:
                if not recip:
                    st.write(f"{val} (left > right)")
                    pairs[(i, j)] = float(val)
                else:
                    st.write(f"1/{val} (right > left)")
                    pairs[(i, j)] = 1.0 / float(val)

    return pairs

# ------------------------------- App body --------------------------------- #
st.set_page_config(page_title="AHP – Transmission Routes (Steps 1–2)", layout="wide")

st.title("AHP – Transmission Routes")
st.markdown("""
**Steps 1 & 2 (visible to experts):**  
Provide pairwise comparisons for **Importance** and **Likelihood** of each transmission route (Saaty 1–9 scale).

**Steps 3–5 (hidden):**  
The app internally aggregates your inputs into AHP matrices, computes **weights** and **consistency**, and produces a **Combined Risk** (Importance × Likelihood) ranking.  
Click **Export Excel** to download the results.
""")

with st.expander("Show transmission routes", expanded=False):
    for idx, r in enumerate(ROUTES, start=1):
        st.write(f"{idx}. {r}")

st.divider()
st.header("Step 1 — Pairwise comparisons: Importance")
pairs_I = pairwise_form("I", "Fill comparisons for **Importance**")

st.divider()
st.header("Step 2 — Pairwise comparisons: Likelihood")
pairs_L = pairwise_form("L", "Fill comparisons for **Likelihood**")

st.divider()
st.subheader("Export results")
st.caption("The Excel file includes: raw & % weights, rankings, CR/CI, and the two aggregated matrices.")

# ---------------------------- Compute & export ---------------------------- #
def compute_and_package_excel(pairs_I, pairs_L) -> bytes:
    # Build matrices
    M_I = matrix_from_upper_triangle(N, pairs_I)
    M_L = matrix_from_upper_triangle(N, pairs_L)

    # Consistency & weights
    CR_I, CI_I, lam_I, w_I = consistency_ratio(M_I)
    CR_L, CI_L, lam_L, w_L = consistency_ratio(M_L)

    # Combined risk (normalized again for clarity)
    risk = w_I * w_L
    risk = risk / risk.sum()

    # Assemble results table
    df = pd.DataFrame({
        "Route": ROUTES,
        "Importance_w": w_I,
        "Likelihood_w": w_L,
        "Risk_w": risk
    })
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False, method="min").astype(int)
    df["Rank_Likelihood"] = df["Likelihood_w"].rank(ascending=False, method="min").astype(int)
    df["Rank_Risk"]       = df["Risk_w"].rank(ascending=False, method="min").astype(int)

    # Percentage columns
    for col in ("Importance_w", "Likelihood_w", "Risk_w"):
        df[col + " (%)"] = (df[col] * 100).round(4)

    # Sort by risk rank
    df = df.sort_values("Rank_Risk").reset_index(drop=True)

    # Build Excel in-memory
    buf = io.BytesIO()
    # Prefer openpyxl; fall back to xlsxwriter if not available
    engine = None
    try:
        import openpyxl  # noqa
        engine = "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            engine = None
    if engine is None:
        raise RuntimeError("Please install 'openpyxl' or 'xlsxwriter' to export Excel.")

    with pd.ExcelWriter(buf, engine=engine) as writer:
        # Results sheet
        cols = [
            "Rank_Risk", "Route",
            "Risk_w", "Risk_w (%)",
            "Importance_w", "Importance_w (%)",
            "Likelihood_w", "Likelihood_w (%)",
            "Rank_Importance", "Rank_Likelihood",
        ]
        df[cols].to_excel(writer, sheet_name="Results", index=False)

        # Consistency sheet
        con = pd.DataFrame({
            "Criterion": ["Importance", "Likelihood"],
            "n (routes)": [N, N],
            "lambda_max": [lam_I, lam_L],
            "CI": [CI_I, CI_L],
            "CR": [CR_I, CR_L]
        })
        con.to_excel(writer, sheet_name="Consistency", index=False)

        # Aggregated matrices sheet
        df_MI = pd.DataFrame(M_I, index=ROUTES, columns=ROUTES)
        df_ML = pd.DataFrame(M_L, index=ROUTES, columns=ROUTES)
        df_MI.to_excel(writer, sheet_name="Aggregated_Matrices", startrow=0, startcol=0)
        startcol_right = df_MI.shape[1] + 2
        df_ML.to_excel(writer, sheet_name="Aggregated_Matrices", startrow=0, startcol=startcol_right)

    buf.seek(0)
    return buf.read(), (CR_I, CR_L)

# Button to export
if st.button("Export Excel"):
    try:
        data, (CRI, CRL) = compute_and_package_excel(pairs_I, pairs_L)
        st.success("AHP computed. If CR > 0.10, consider revisiting some comparisons.")
        st.write(f"Importance CR: **{CRI:.3f}**  •  Likelihood CR: **{CRL:.3f}**")
        st.download_button(
            label="Download AHP results (Excel)",
            data=data,
            file_name="ahp_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# (Optional) Minimal preview of what will be exported (no details of steps 3–5)
with st.expander("Preview top routes by combined risk (for your validation)"):
    try:
        data, _ = compute_and_package_excel(pairs_I, pairs_L)
        # Read back the Results sheet only for display
        with io.BytesIO(data) as b:
            xls = pd.ExcelFile(b)
            df_prev = pd.read_excel(xls, sheet_name="Results")
        st.dataframe(df_prev.head(10))
    except Exception:
        st.info("Fill some comparisons and click Export to see a preview.")

