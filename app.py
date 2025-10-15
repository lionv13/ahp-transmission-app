# -*- coding: utf-8 -*-
"""
HPAI – Transmission Routes 
- Experts enter pairwise comparisons for Importance (Score 1–9).
- Experts enter their Name and Credentials.
- Clear instructions box for how to fill & save.
- Exports Excel named with the expert's name (metadata included).

Run: streamlit run app.py
"""

from __future__ import annotations
import io, re, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Config ---------------- #
SUBMIT_EMAIL = "your-submission@organization.org"   # <-- set your address here

# ---------------- Routes ---------------- #
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

# -------------- Saaty RI -------------- #
SAATY_RI = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48,13:1.56,14:1.57,15:1.59}

# -------------- AHP core -------------- #
def eigen_priority(M: np.ndarray):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    return w / w.sum(), vals[idx].real

def consistency_ratio(M: np.ndarray):
    n = M.shape[0]
    w, lam = eigen_priority(M)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return CR, CI, lam, w

def matrix_from_upper_triangle(n: int, pairs: dict[tuple[int,int], float]) -> np.ndarray:
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

# -------------- UI helpers -------------- #
SAATY_SCALE = {
    "1 (equal)": 1, "2 (between 1–3)": 2, "3 (moderate)": 3, "4 (between 3–5)": 4,
    "5 (strong)": 5, "6 (between 5–7)": 6, "7 (very strong)": 7, "8 (between 7–9)": 8, "9 (extreme)": 9,
}
def saaty_selectbox(key_prefix: str, label: str, default=1):
    options = list(SAATY_SCALE.keys())
    default_idx = options.index(next(k for k, v in SAATY_SCALE.items() if v == default))
    choice = st.selectbox(label, options, index=default_idx, key=key_prefix)
    return SAATY_SCALE[choice]

def pairwise_form(criterion_key: str, title: str):
    st.subheader(title)

    # 🔸 Clear instructions (bold + colored box)
    st.markdown(
        """
        <div style="background:#FFF3E0;border-left:6px solid #FF9800;padding:12px 14px;margin:8px 0 14px 0;">
          <div style="font-weight:700;color:#E65100;margin-bottom:6px;">How to fill & save</div>
          <ol style="margin:0 0 4px 18px;">
            <li>For each <b>pair</b> of transmission routes, add a score from <b>1 to 9</b> using the Saaty scale.</li>
            <li>If the <b>left route</b> is more important, choose a value <b>&gt; 1</b>. If the <b>right route</b> is more important, tick <b>Reciprocal</b> to use <b>1/value</b>.</li>
            <li>Fill <b>all</b> pairwise comparisons.</li>
            <li>Click <b>Export Excel</b>.</li>
            <li>Send the downloaded file to <b>""" + SUBMIT_EMAIL + """</b>.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True
    )

    pairs = {}
    head = st.columns([2, 2, 1.2, 1.2, 2])
    with head[0]: st.markdown("**Left route**")
    with head[1]: st.markdown("**Right route**")
    with head[2]: st.markdown("**Saaty**")
    with head[3]: st.markdown("**Reciprocal?**")
    with head[4]: st.markdown("**Preview**")

    for i in range(N - 1):
        for j in range(i + 1, N):
            row = st.columns([2, 2, 1.2, 1.2, 2])
            with row[0]: st.write(f"{i+1}. {ROUTES[i]}")
            with row[1]: st.write(f"{j+1}. {ROUTES[j]}")
            with row[2]: val = saaty_selectbox(f"{criterion_key}_saaty_{i}_{j}", "", default=1)
            with row[3]: recip = st.checkbox(" ", key=f"{criterion_key}_rec_{i}_{j}", value=False)
            with row[4]:
                if not recip:
                    st.write(f"{val} (left > right)")
                    pairs[(i, j)] = float(val)
                else:
                    st.write(f"1/{val} (right > left)")
                    pairs[(i, j)] = 1.0 / float(val)
    return pairs

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "expert"

# -------------- App body -------------- #
st.set_page_config(page_title="AHP – Transmission Routes (Importance only)", layout="wide")
st.title("AHP – Transmission Routes (Importance only)")

with st.expander("Show transmission routes", expanded=False):
    for idx, r in enumerate(ROUTES, start=1):
        st.write(f"{idx}. {r}")

st.divider()
st.header("Expert info")
colA, colB = st.columns([1.2, 1.8])
with colA:
    expert_name = st.text_input("Your full name*", placeholder="e.g., Dr. Jane Doe")
with colB:
    expert_credentials = st.text_input("Credentials / Institution", placeholder="e.g., DVM, PhD – Example University")

st.divider()
st.header("Pairwise comparisons: Importance")
pairs_I = pairwise_form("I", "Fill comparisons for **Importance**")

st.divider()
st.subheader("Export results")
st.caption("Excel includes: raw & % weights, ranking, CR/CI, aggregated matrix, and your metadata.")

def compute_and_package_excel(pairs_I, expert_name, expert_credentials) -> bytes:
    M_I = matrix_from_upper_triangle(N, pairs_I)
    CR_I, CI_I, lam_I, w_I = consistency_ratio(M_I)

    df = pd.DataFrame({"Route": ROUTES, "Importance_w": w_I})
    df["Importance_w (%)"] = (df["Importance_w"] * 100).round(4)
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("Rank_Importance").reset_index(drop=True)

    buf = io.BytesIO()
    try:
        import openpyxl; engine = "openpyxl"
    except Exception:
        import xlsxwriter; engine = "xlsxwriter"

    with pd.ExcelWriter(buf, engine=engine) as writer:
        df[["Rank_Importance", "Route", "Importance_w", "Importance_w (%)"]].to_excel(
            writer, sheet_name="Results", index=False
        )
        pd.DataFrame({
            "Criterion": ["Importance"], "n (routes)": [N],
            "lambda_max": [lam_I], "CI": [CI_I], "CR": [CR_I]
        }).to_excel(writer, sheet_name="Consistency", index=False)
        pd.DataFrame(M_I, index=ROUTES, columns=ROUTES).to_excel(writer, sheet_name="Aggregated_Matrix")
        pd.DataFrame({
            "Field": ["Expert name", "Credentials/Institution", "Submission email", "Timestamp (UTC)"],
            "Value": [expert_name, expert_credentials, SUBMIT_EMAIL, dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"],
        }).to_excel(writer, sheet_name="Metadata", index=False)

    buf.seek(0)
    return buf.read(), CR_I

export_disabled = (expert_name.strip() == "")
if st.button("Export Excel", disabled=export_disabled):
    if export_disabled:
        st.warning("Please enter your name to enable export.")
    else:
        try:
            data, CRI = compute_and_package_excel(pairs_I, expert_name.strip(), expert_credentials.strip())
            st.success(f"AHP (Importance) computed. Email the file to {SUBMIT_EMAIL}.")
            st.write(f"Importance CR: **{CRI:.3f}**")
            filename = f"ahp_importance_results_{slugify(expert_name)}.xlsx"
            st.download_button(
                label=f"Download results – {filename}",
                data=data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
