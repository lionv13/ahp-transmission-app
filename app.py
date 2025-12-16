# -*- coding: utf-8 -*-
"""
ASF Transmission Routes – AHP (Importance only)
Wizard: Intro → one page per pairwise comparison → Finish & Export

Run: streamlit run app_asf.py
"""

from __future__ import annotations
import io, json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

# ============================== CONFIG =============================== #
st.set_page_config(page_title="ASF Transmission Routes – Importance", layout="wide")

ROUTES: List[str] = [
    "Introduction of ASF virus through breeding pigs, weaned piglets and semen",
    "Introduction of ASF virus through wild boar in the neighbourhood",
    "Introduction of ASF virus through persons (farmer, vet, truck driver, ...)",
    "Introduction of ASF virus through equipment",
    "Introduction of ASF virus through animal transport vehicle / tools",
    "Introduction of ASF virus through feed trucks",
    "Introduction of ASF virus through swill feeding",
    "Introduction of ASF virus through regular feeding",
    "Introduction of ASF virus through water",
    "Introduction of ASF virus through the air over short distance (<1000m)",
    "Introduction of ASF virus through other animals (pets, cattle, …)",
    "Introduction of ASF virus through truck of the rendering company",
    "Introduction of ASF virus through manure from other farms (hoses, manure spread in neighbourhood)",
    "Introduction of ASF virus through vermin, birds, and insects",
]
N = len(ROUTES)
APP_VERSION = "ASF-1.0-importance"

SAATY_RI = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
}

# ============================= HELPERS =============================== #
def all_pairs(n: int):
    return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

def matrix_from_upper_triangle(n, pairs):
    M = np.ones((n, n), dtype=float)
    for (i, j), v in pairs.items():
        M[i, j] = v
        M[j, i] = 1 / v
    return M

def eigen_priority(M):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    return w / w.sum(), float(vals[idx].real)

def consistency_ratio(M):
    n = M.shape[0]
    w, lam = eigen_priority(M)
    CI = (lam - n) / (n - 1)
    CR = CI / SAATY_RI.get(n, 1.59)
    return CR, CI, lam, w

# ============================= EMAIL (FIXED) =============================== #
def send_results_email(to_email, subject, body, attachment_bytes, filename):
    import smtplib
    from email.message import EmailMessage

    smtp_cfg = st.secrets["smtp"]
    host = smtp_cfg["host"]
    port = int(smtp_cfg.get("port", 587))
    user = smtp_cfg["user"]
    password = smtp_cfg["password"]
    from_email = smtp_cfg.get("from_email", user)
    use_tls = smtp_cfg.get("use_tls", True)

    msg = EmailMessage()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)

# ============================ EXCEL ============================ #
def build_excel(expert_name, pairs):
    M = matrix_from_upper_triangle(N, pairs)
    CR, CI, lam, w = consistency_ratio(M)

    df = pd.DataFrame({
        "Route": ROUTES,
        "Importance_w": w,
        "Importance_w (%)": w * 100
    }).sort_values("Importance_w", ascending=False)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as wtr:
        df.to_excel(wtr, "Results", index=False)
        pd.DataFrame({"CR": [CR], "CI": [CI], "λmax": [lam]}).to_excel(
            wtr, "Consistency", index=False
        )
    buf.seek(0)
    return buf.read()

# ============================ STATE ============================ #
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = {}
if "errors" not in st.session_state:
    st.session_state.errors = {}

pairs_list = all_pairs(N)

# ============================ UI ============================ #
def intro_page():
    st.title("ASF Transmission Routes – Importance")

    st.markdown("""
### Instructions (confidential – internal use only)

1. **Enter your expert ID / credentials** below.  
   This information is used **only for internal scientific purposes** and handled **strictly confidentially**.
2. You will compare **pairs of transmission routes**.
3. For each pair, select a **score from 1 to 9**.
4. **0 means no selection** → you cannot continue until a score is chosen.
5. You may go **Back** at any time; your previous answers are preserved.
    """)

    st.session_state.expert_name = st.text_input(
        "Expert ID / Name", st.session_state.get("expert_name", "")
    )

    st.button(
        "Start scoring",
        disabled=not st.session_state.expert_name,
        on_click=lambda: st.session_state.update(page_idx=1)
    )

def pair_page(k):
    i, j = pairs_list[k - 1]
    left, right = ROUTES[i], ROUTES[j]

    st.markdown(f"### {left}")

    score = st.selectbox(
        "Score (0–9)",
        list(range(10)),
        key=f"s_{i}_{j}",
        format_func=lambda x: "0 – select" if x == 0 else str(x)
    )

    rec = st.checkbox("Reciprocal (right more important)", key=f"r_{i}_{j}")

    if score != 0:
        value = 1 / score if rec else score
        st.caption(f"Current value: **{value:.3f}**")

    err_key = f"err_{i}_{j}"
    if err_key in st.session_state.errors:
        st.error(st.session_state.errors[err_key])

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=lambda: st.session_state.update(page_idx=k - 1))
    with col2:
        if st.button("Next"):
            if score == 0:
                st.session_state.errors[err_key] = "Please select a score (1–9)."
            else:
                st.session_state.errors.pop(err_key, None)
                st.session_state.pairs[(i, j)] = value
                st.session_state.page_idx = k + 1

def finish_page():
    st.success("All comparisons completed.")
    excel = build_excel(st.session_state.expert_name, st.session_state.pairs)

    st.download_button(
        "⬇️ Download results",
        excel,
        f"ASF_AHP_{st.session_state.expert_name}.xlsx"
    )

    to = st.secrets["smtp"].get("report_to")
    if to and st.button(f"📤 Send results to {to}"):
        send_results_email(
            to,
            f"ASF AHP Results – {st.session_state.expert_name}",
            "Attached are the ASF AHP results.",
            excel,
            f"ASF_AHP_{st.session_state.expert_name}.xlsx"
        )
        st.success("Email sent successfully.")

    st.button(
        "Start over",
        on_click=lambda: st.session_state.clear()
    )

# ============================ ROUTER ============================ #
if st.session_state.page_idx == 0:
    intro_page()
elif 1 <= st.session_state.page_idx <= len(pairs_list):
    st.progress(st.session_state.page_idx / len(pairs_list))
    pair_page(st.session_state.page_idx)
else:
    finish_page()
