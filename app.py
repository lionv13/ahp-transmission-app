# -*- coding: utf-8 -*-
"""
HPAI Transmission Routes – AHP (Importance only)
Wizard: Intro → one page per pairwise comparison → Finish & Export

- Experts enter pairwise comparisons for Importance (Score 1–9).
- Steps 3–5 (weights, consistency, matrix, ranking) are computed internally.
- Exports an Excel file named with the expert's name.
- Emails results automatically to smtp.report_to (from Streamlit secrets).

Run locally:  streamlit run app.py
"""

from __future__ import annotations
import io
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ============================== CONFIG =============================== #
st.set_page_config(page_title="HPAI Transmission Routes – Importance", layout="wide")

ROUTES: List[str] = [
    "Introduction of virus through introduction of day old chick",  # 1
    "Introduction of virus trough animal transport vehicle / equipment",  # 2
    "Introduction of virus through professional visitors at the farm (vet, truck driver, catching team)",  # 3
    "Introduction of virus through feed trucks",  # 4
    "Introduction of virus through vermin or birds",  # 5
    "Introduction of virus through water",  # 6
    "Introduction of virus through truck of the rendering company",  # 7
    "Introduction of virus through shared equipment",  # 8
    "Introduction of virus through other farm animals",  # 9
    "Introduction of virus through the air over short distance (<1000m)",  # 10
    "Introduction of virus through spreading of manure originating from infected farms in close vicinity of the farm",  # 11
]
N = len(ROUTES)

# Saaty Random Index (for CR). We keep this name even though UI says "Score"
SAATY_RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
            9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59}

# ============================= HELPERS =============================== #
def all_pairs(n: int) -> List[Tuple[int, int]]:
    """Return list of index pairs (i, j) with i<j."""
    return [(i, j) for i in range(n-1) for j in range(i+1, n)]

@st.cache_data(show_spinner=False)
def matrix_from_upper_triangle(n: int, pairs: Dict[Tuple[int, int], float]) -> np.ndarray:
    """Build reciprocal AHP matrix from i<j values."""
    M = np.ones((n, n), dtype=float)
    for (i, j), v in pairs.items():
        if i == j:
            continue
        vv = float(v)
        if vv <= 0:
            raise ValueError("Score values must be > 0")
        M[i, j] = vv
        M[j, i] = 1.0 / vv
    return M

@st.cache_data(show_spinner=False)
def eigen_priority(M: np.ndarray):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()
    return w, float(vals[idx].real)

@st.cache_data(show_spinner=False)
def consistency_ratio(M: np.ndarray):
    n = M.shape[0]
    w, lam = eigen_priority(M)
    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = SAATY_RI.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    return CR, CI, lam, w

@st.cache_resource(show_spinner=False)
def get_excel_engine() -> str:
    """Pick an Excel writer engine once per session."""
    try:
        import openpyxl  # noqa: F401
        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401
            return "xlsxwriter"
        except Exception as e:
            raise RuntimeError(
                "No Excel writer installed. Add 'openpyxl' or 'xlsxwriter' to requirements.txt."
            ) from e

@st.cache_resource(show_spinner=False)
def get_mailer():
    """Create a reusable SMTP client from Streamlit secrets."""
    import smtplib
    host = st.secrets["smtp"]["host"]
    port = int(st.secrets["smtp"].get("port", 587))
    user = st.secrets["smtp"]["user"]
    password = st.secrets["smtp"]["password"]
    use_tls = bool(st.secrets["smtp"].get("use_tls", True))
    s = smtplib.SMTP(host, port, timeout=30)
    if use_tls:
        s.starttls()
    s.login(user, password)
    return s, user  # return sender as well

def send_results_email(to_email: str, subject: str, body: str, attachment_bytes: bytes, filename: str):
    """Send email with Excel results."""
    from email.message import EmailMessage
    (smtp, sender) = get_mailer()
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )
    smtp.send_message(msg)

def build_excel(expert_name: str, pairs_I: Dict[Tuple[int, int], float]) -> bytes:
    """Compute results and return an Excel file as bytes."""
    M_I = matrix_from_upper_triangle(N, pairs_I)
    CR_I, CI_I, lam_I, w_I = consistency_ratio(M_I)

    # Single-criterion “Risk” equals Importance weights (kept for naming continuity)
    risk = w_I / np.sum(w_I)

    df = pd.DataFrame({
        "Route": ROUTES,
        "Importance_w": w_I,
        "Risk_w": risk,  # identical to Importance here
    })

    # Ranks
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False, method="min").astype(int)
    df["Rank_Risk"]       = df["Risk_w"].rank(ascending=False, method="min").astype(int)

    # Percent columns
    df["Importance_w (%)"] = (df["Importance_w"] * 100).round(4)
    df["Risk_w (%)"]       = (df["Risk_w"] * 100).round(4)

    # Sort by risk rank
    df = df.sort_values("Rank_Risk").reset_index(drop=True)

    # Excel
    buf = io.BytesIO()
    engine = get_excel_engine()
    with pd.ExcelWriter(buf, engine=engine) as writer:
        cols = [
            "Rank_Risk", "Route",
            "Risk_w", "Risk_w (%)",
            "Importance_w", "Importance_w (%)",
            "Rank_Importance",
        ]
        df[cols].to_excel(writer, sheet_name="Results", index=False)

        con = pd.DataFrame({
            "Criterion": ["Importance"],
            "n (routes)": [N],
            "lambda_max": [lam_I],
            "CI": [CI_I],
            "CR": [CR_I],
        })
        con.to_excel(writer, sheet_name="Consistency", index=False)

        # Aggregated matrix
        pd.DataFrame(M_I, index=[f"{i+1}. {r}" for i, r in enumerate(ROUTES)],
                        columns=[f"{i+1}. {r}" for i, r in enumerate(ROUTES)]
                    ).to_excel(writer, sheet_name="Aggregated_Matrix", index=True)

        # Metadata
        meta = pd.DataFrame({
            "Expert Name": [expert_name],
        })
        meta.to_excel(writer, sheet_name="Meta", index=False)

    buf.seek(0)
    return buf.read()

# ============================ STATE SETUP ============================ #
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0  # 0=intro, then 1..len(pairs), then finish
if "pairs_list" not in st.session_state:
    st.session_state.pairs_list = all_pairs(N)  # sequence of (i,j)
if "pairs_values" not in st.session_state:
    st.session_state.pairs_values: Dict[Tuple[int, int], float] = {}
if "expert_name" not in st.session_state:
    st.session_state.expert_name = ""
if "expert_credentials" not in st.session_state:
    st.session_state.expert_credentials = ""

pairs_seq = st.session_state.pairs_list
page_idx = st.session_state.page_idx

# =============================== UI ================================= #
def intro_page():
    st.title("HPAI Transmission Routes")
    st.markdown("### Importance (Score 1–9)")

    st.markdown(
        """
**How to score each pairwise comparison**

- **Score 1**: no difference between the two routes.  
- **Scores 3, 5, 7**: *moderate*, *strong*, and *very strong* differences (left route more important than right).  
- **Score 9**: *extreme* difference (left ≫ right).  
- **Scores 2, 6, 8**: in-between values (*equal to moderate*, *moderate→strong*, *strong→very strong*).  
- **If the right route is more important**, **tick the _Reciprocal_ box** (this sets the value to **1/score**).

<div style="padding:0.6rem 0.8rem; border-left:6px solid #444; background:#f7f7f7;">
<b>Read before starting:</b> <span style="color:#c0392b;"><b>Pick a score &gt; 1 if the left route is more important than the right;</b></span>  
use the <b>Reciprocal</b> checkbox to flip direction (becomes <b>1/score</b>).
</div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Show transmission routes", expanded=False):
        for idx, r in enumerate(ROUTES, start=1):
            st.write(f"**{idx}.** {r}")

    st.divider()
    st.subheader("Expert identification")
    colA, colB = st.columns(2)
    with colA:
        st.session_state.expert_name = st.text_input("Your name", value=st.session_state.expert_name).strip()
    with colB:
        st.session_state.expert_credentials = st.text_input("Your credentials / affiliation (optional)", value=st.session_state.expert_credentials).strip()

    st.divider()
    disabled = len(st.session_state.expert_name) == 0
    st.button("Start scoring", type="primary", disabled=disabled, on_click=lambda: _advance())

def _advance():
    st.session_state.page_idx += 1

def _back():
    st.session_state.page_idx = max(0, st.session_state.page_idx - 1)

def pair_page(k: int, ij: Tuple[int, int]):
    i, j = ij
    left_route = ROUTES[i]
    right_route = ROUTES[j]

    st.markdown(f"### **{i+1}. {left_route}**")
    st.caption("Compare the left route against the route shown on the right.")

    lcol, rcol = st.columns([1.6, 1.4], vertical_alignment="top")

    with lcol:
        with st.container(border=True):
            st.markdown("**Left route**")
            st.write(f"{i+1}. {left_route}")

    with rcol:
        with st.container(border=True):
            st.markdown("**Compare against (Right route)**")
            st.write(f"{j+1}. {right_route}")

            # Score selector (1–9, integers only)
            score = st.selectbox(
                "Score",
                options=list(range(1, 10)),
                index=0,
                key=f"score_{i}_{j}"
            )
            # Reciprocal toggle
            recip = st.checkbox(
                "Reciprocal (tick if the RIGHT route is more important than the LEFT)",
                value=False,
                key=f"recip_{i}_{j}"
            )

            # Store value immediately
            val = 1.0 / float(score) if recip else float(score)
            st.session_state.pairs_values[(i, j)] = val

            # Preview
            st.caption("Preview of stored value:")
            if recip:
                st.write(f"**1/{score}** (right > left)")
            else:
                st.write(f"**{score}** (left > right)")

    st.divider()
    nav_left, nav_mid, nav_right = st.columns([1, 5, 1])
    with nav_left:
        st.button("Back", on_click=_back)
    with nav_right:
        st.button("Next", type="primary", on_click=_advance)

def finish_page():
    st.header("Finish")
    st.write("You have completed all pairwise comparisons.")

    # Compute and preview
    try:
        excel_bytes = build_excel(st.session_state.expert_name, st.session_state.pairs_values)
        with io.BytesIO(excel_bytes) as b:
            xls = pd.ExcelFile(b)
            df_prev = pd.read_excel(xls, sheet_name="Results")
        st.subheader("Preview of results")
        st.dataframe(df_prev.head(10))
    except Exception as e:
        st.error(f"Computation error: {e}")
        st.stop()

    st.divider()
    # Export & auto-email (to secrets)
    export_name_slim = st.session_state.expert_name.strip().replace(" ", "_")
    file_name = f"HPAI_AHP_Importance_{export_name_slim}.xlsx"

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Excel results",
            data=excel_bytes,
            file_name=file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with col2:
        to_email = st.secrets.get("smtp", {}).get("report_to", "")
        if to_email:
            if st.button(f"Send results to {to_email}", type="primary"):
                try:
                    subject = f"AHP Results – HPAI Transmission Routes (Importance) – {st.session_state.expert_name}"
                    creds_info = f" ({st.session_state.expert_credentials})" if st.session_state.expert_credentials else ""
                    body = (
                        f"Dear team,\n\n"
                        f"Attached are the AHP (Importance only) results for HPAI Transmission Routes.\n"
                        f"Expert: {st.session_state.expert_name}{creds_info}\n\n"
                        f"Best regards."
                    )
                    send_results_email(to_email, subject, body, excel_bytes, file_name)
                    st.success("Email sent.")
                except Exception as e:
                    st.error(f"Email failed: {e}")
        else:
            st.info("Configure `smtp.report_to` in Streamlit secrets to enable automatic emailing.")

    st.divider()
    st.button("Start over", on_click=lambda: _reset())

def _reset():
    st.session_state.page_idx = 0
    st.session_state.pairs_values = {}
    # Keep name/credentials as-is for convenience


# =========================== PAGE ROUTER =========================== #
pairs = pairs_seq  # list of (i,j)
total_pair_pages = len(pairs)

if page_idx == 0:
    intro_page()
elif 1 <= page_idx <= total_pair_pages:
    # 1-based indexing for page number; fetch pair by index-1
    current_pair = pairs[page_idx - 1]
    st.progress(page_idx / total_pair_pages, text=f"Pair {page_idx} of {total_pair_pages}")
    pair_page(page_idx, current_pair)
else:
    finish_page()
