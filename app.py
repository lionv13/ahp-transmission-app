# -*- coding: utf-8 -*-
"""
HPAI Transmission Routes – AHP (Importance only)
Wizard: Intro → one page per pairwise comparison → Finish & Export

Includes:
- Detailed instructions for experts
- Save draft (JSON) and Load to resume
- Optional Draft Excel (fills missing pairs with 1)
- Final export with automatic email sending

Run: streamlit run app.py
"""

from __future__ import annotations
import io, json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

# ============================== CONFIG =============================== #
st.set_page_config(page_title="HPAI Transmission Routes – Importance", layout="wide")

ROUTES: List[str] = [
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
APP_VERSION = "1.4-instructions-refined"

# Saaty Random Index (for CR)
SAATY_RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
            9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59}

# ============================= HELPERS =============================== #
def all_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

@st.cache_data(show_spinner=False)
def matrix_from_upper_triangle(n: int, pairs: Dict[Tuple[int, int], float]) -> np.ndarray:
    M = np.ones((n, n), dtype=float)
    for (i, j), v in pairs.items():
        M[i, j] = float(v)
        M[j, i] = 1.0 / float(v)
    return M

def matrix_from_upper_triangle_allow_missing(n: int, pairs: Dict[Tuple[int, int], float]) -> np.ndarray:
    M = np.ones((n, n), dtype=float)
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = float(pairs.get((i, j), 1.0))
            M[i, j] = val
            M[j, i] = 1.0 / val
    return M

@st.cache_data(show_spinner=False)
def eigen_priority(M: np.ndarray):
    vals, vecs = np.linalg.eig(M)
    idx = np.argmax(vals.real)
    w = np.abs(vecs[:, idx].real)
    return w / w.sum(), float(vals[idx].real)

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
    try:
        import openpyxl  # noqa
        return "openpyxl"
    except Exception:
        import xlsxwriter
        return "xlsxwriter"

@st.cache_resource(show_spinner=False)
def get_mailer():
    import smtplib
    host = st.secrets["smtp"]["host"]
    port = int(st.secrets["smtp"].get("port", 587))
    user = st.secrets["smtp"]["user"]
    password = st.secrets["smtp"]["password"]
    s = smtplib.SMTP(host, port, timeout=30)
    s.starttls()
    s.login(user, password)
    return s, user

def send_results_email(to_email: str, subject: str, body: str, attachment_bytes: bytes, filename: str):
    from email.message import EmailMessage
    smtp, sender = get_mailer()
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    msg.add_attachment(attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename)
    smtp.send_message(msg)

# ============================ EXCEL BUILDERS ============================ #
def build_excel(expert_name: str, pairs: Dict[Tuple[int, int], float]) -> bytes:
    M = matrix_from_upper_triangle(N, pairs)
    CR, CI, lam, w = consistency_ratio(M)
    risk = w / np.sum(w)
    df = pd.DataFrame({"Route": ROUTES, "Importance_w": w, "Risk_w": risk})
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False).astype(int)
    df["Rank_Risk"] = df["Risk_w"].rank(ascending=False).astype(int)
    df["Importance_w (%)"] = df["Importance_w"] * 100
    df["Risk_w (%)"] = df["Risk_w"] * 100
    df = df.sort_values("Rank_Risk").reset_index(drop=True)
    buf = io.BytesIO()
    engine = get_excel_engine()
    with pd.ExcelWriter(buf, engine=engine) as wtr:
        df.to_excel(wtr, "Results", index=False)
        pd.DataFrame({"Criterion": ["Importance"], "λmax": [lam], "CI": [CI], "CR": [CR]}).to_excel(
            wtr, "Consistency", index=False
        )
        pd.DataFrame(
            M,
            index=[f"{i+1}. {r}" for i, r in enumerate(ROUTES)],
            columns=[f"{i+1}. {r}" for i, r in enumerate(ROUTES)],
        ).to_excel(wtr, "Matrix")
        pd.DataFrame({"Expert": [expert_name], "Version": [APP_VERSION]}).to_excel(wtr, "Meta", index=False)
    buf.seek(0)
    return buf.read()

def build_excel_draft(expert_name: str, pairs_partial: Dict[Tuple[int, int], float]) -> bytes:
    M = matrix_from_upper_triangle_allow_missing(N, pairs_partial)
    CR, CI, lam, w = consistency_ratio(M)
    df = pd.DataFrame({"Route": ROUTES, "Importance_w": w, "Risk_w": w})
    df["Rank_Risk"] = df["Risk_w"].rank(ascending=False).astype(int)
    buf = io.BytesIO()
    engine = get_excel_engine()
    with pd.ExcelWriter(buf, engine=engine) as wtr:
        df.to_excel(wtr, "Results (DRAFT)", index=False)
        pd.DataFrame(
            {"Note": ["Some pairs missing, set to 1"], "λmax": [lam], "CI": [CI], "CR": [CR]}
        ).to_excel(wtr, "Consistency", index=False)
    buf.seek(0)
    return buf.read()

# ============================ STATE SETUP ============================ #
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if "pairs_list" not in st.session_state:
    st.session_state.pairs_list = all_pairs(N)
if "pairs_values" not in st.session_state:
    st.session_state.pairs_values = {}
if "expert_name" not in st.session_state:
    st.session_state.expert_name = ""
if "expert_credentials" not in st.session_state:
    st.session_state.expert_credentials = ""

pairs_seq = st.session_state.pairs_list
page_idx = st.session_state.page_idx

# =========================== DRAFT SAVE/LOAD =========================== #
def serialize_draft() -> str:
    data = {
        "v": APP_VERSION,
        "name": st.session_state.expert_name,
        "cred": st.session_state.expert_credentials,
        "page": st.session_state.page_idx,
        "pairs": {f"{i},{j}": v for (i, j), v in st.session_state.pairs_values.items()},
    }
    return json.dumps(data, indent=2)

def load_draft(txt: str):
    d = json.loads(txt)
    st.session_state.expert_name = d.get("name", "")
    st.session_state.expert_credentials = d.get("cred", "")
    st.session_state.page_idx = d.get("page", 0)
    pairs = {}
    for k, v in d.get("pairs", {}).items():
        i, j = map(int, k.split(","))
        pairs[(i, j)] = float(v)
    st.session_state.pairs_values = pairs
    st.success("Draft loaded successfully.")

with st.sidebar:
    st.subheader("💾 Save / Resume progress")
    st.caption("If you cannot complete the evaluation in one session, save your progress and resume later.")
    st.download_button("⬇️ Save draft (JSON)", serialize_draft().encode(),
                       "hpai_ahp_draft.json", "application/json", use_container_width=True)
    uploaded = st.file_uploader("Load draft", type="json")
    if uploaded:
        load_draft(uploaded.read().decode())
    st.markdown("---")
    if st.toggle("Generate Draft Excel (fill missing with 1)", value=False):
        try:
            data = build_excel_draft(st.session_state.expert_name or "Anonymous", st.session_state.pairs_values)
            st.download_button("⬇️ Download Draft Excel", data, "HPAI_AHP_DRAFT.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Draft error: {e}")

# =============================== UI ================================= #
def intro_page():
    st.title("HPAI Transmission Routes")
    st.markdown("### Importance (Score 1–9)")

    st.markdown(
        """
## 🧭 Instructions for completing the evaluation

1. **To start the evaluation**, click on **Start scoring** below.  
2. You will see **pairs of transmission routes**.  
   For **each pair**, assign a **score (1–9)** following the scale explained below:
   - **1** → no difference between the two routes.  
   - **3, 5, 7** → moderate, strong, and very strong difference (left > right).  
   - **9** → extreme difference (left ≫ right).  
   - **2, 6, 8** → in-between values.  
   - If the **right route** is more important → **tick the “Reciprocal” box** (sets the value to 1/score).
3. After scoring each pair, click **Next** to continue.  
4. If you **cannot finish in one session**, open the left sidebar and:
   - Click **“Save draft (JSON)”** to download your progress file.  
   - Later, reopen the app and **upload that file** to resume where you left off.
5. Once you finish all comparisons:
   - You’ll reach a **Finish page** where you can:
     - **Download** a copy of your results (Excel file).  
     - Click **“Send results”** — your answers will be automatically emailed to the evaluation team.
6. Your results are saved only after you export or send them.

---

<div style="padding:0.6rem 0.8rem; border-left:6px solid #444; background:#f7f7f7;">
<b>Important:</b> Pick a <b>Score &gt; 1</b> if the left route is more important than the right.  
Use the <b>Reciprocal</b> checkbox if the right route is more important.
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
        st.session_state.expert_credentials = st.text_input(
            "Your credentials / affiliation (optional)", value=st.session_state.expert_credentials
        ).strip()

    st.divider()
    st.button("Start scoring", type="primary", disabled=len(st.session_state.expert_name) == 0, on_click=lambda: _advance())

def _advance():
    st.session_state.page_idx += 1

def _back():
    st.session_state.page_idx = max(0, st.session_state.page_idx - 1)

def pair_page(k: int, ij: Tuple[int, int]):
    i, j = ij
    left, right = ROUTES[i], ROUTES[j]

    st.markdown(f"### **{i+1}. {left}**")
    st.caption("Compare the importance (likelihood of occurrence) of the left transmission route with the right transmission route.")

    lcol, rcol = st.columns([1.6, 1.4])
    with lcol:
        with st.container(border=True):
            st.markdown("**Left route**")
            st.write(f"{i+1}. {left}")
    with rcol:
        with st.container(border=True):
            st.markdown("**Right route**")
            st.write(f"{j+1}. {right}")
            score = st.selectbox("Score (1–9)", range(1, 10), index=0, key=f"s_{i}_{j}")
            rec = st.checkbox("Reciprocal (if RIGHT route is more important)", key=f"r_{i}_{j}")
            st.session_state.pairs_values[(i, j)] = 1 / score if rec else float(score)
            st.caption("Stored value:")
            st.write(f"**{1/score if rec else score:.3f}**")
    st.divider()
    c1, _, c3 = st.columns([1, 4, 1])
    with c1:
        st.button("Back", on_click=_back)
    with c3:
        st.button("Next", type="primary", on_click=_advance)

def finish_page():
    st.header("Finish")
    st.success("You have completed all pairwise comparisons.")
    try:
        excel = build_excel(st.session_state.expert_name, st.session_state.pairs_values)
        df = pd.read_excel(io.BytesIO(excel), sheet_name="Results")
        st.subheader("Preview of results")
        st.dataframe(df.head(10))
    except Exception as e:
        st.error(f"Error computing results: {e}")
        return
    st.divider()
    name = st.session_state.expert_name.replace(" ", "_")
    filename = f"HPAI_AHP_Importance_{name}.xlsx"
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download Excel results", excel, filename,
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with col2:
        to = st.secrets.get("smtp", {}).get("report_to", "")
        if to and st.button(f"📤 Send results to {to}", type="primary"):
            try:
                subj = f"HPAI AHP Results – {st.session_state.expert_name}"
                body = f"Dear team,\n\nAttached are the AHP Importance results.\nExpert: {st.session_state.expert_name}\n\nBest regards."
                send_results_email(to, subj, body, excel, filename)
                st.success("Results sent successfully.")
            except Exception as e:
                st.error(f"Email failed: {e}")
    st.divider()
    st.button("Start over", on_click=lambda: _reset())

def _reset():
    st.session_state.page_idx = 0
    st.session_state.pairs_values = {}

# =========================== PAGE ROUTER =========================== #
pairs = pairs_seq
if page_idx == 0:
    intro_page()
elif 1 <= page_idx <= len(pairs):
    st.progress(page_idx / len(pairs), f"Pair {page_idx} of {len(pairs)}")
    pair_page(page_idx, pairs[page_idx - 1])
else:
    finish_page()
