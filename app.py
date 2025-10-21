# -*- coding: utf-8 -*-
"""
HPAI Transmission Routes – Expert Scoring Wizard (Importance only)
Flow:
  0) Intro
  1..P) One page per pair (i<j)  where P = N*(N-1)//2
  P+1) Finish (export + optional email)

Run: streamlit run app.py
"""

from __future__ import annotations
import io, re, datetime as dt, subprocess, sys, smtplib
from email.message import EmailMessage
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Auto-install Excel engines ---------------- #
try:
    import openpyxl  # noqa
    ENGINE = "openpyxl"
except ImportError:
    try:
        import xlsxwriter  # noqa
        ENGINE = "xlsxwriter"
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
        import openpyxl  # type: ignore
        ENGINE = "openpyxl"

# ---------------- Configuration ---------------- #
DEFAULT_TO_EMAIL = "your-submission@organization.org"

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
PAIRS = [(i, j) for i in range(N - 1) for j in range(i + 1, N)]
P = len(PAIRS)

SAATY_RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
            9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59}

SCORE_SCALE = {
    "1 (no difference)": 1,
    "2 (between 1–3)": 2,
    "3 (moderate)": 3,
    "4 (between 3–5)": 4,
    "5 (strong)": 5,
    "6 (between 5–7)": 6,
    "7 (very strong)": 7,
    "8 (between 7–9)": 8,
    "9 (extreme)": 9,
}

# ---------------- AHP core ---------------- #
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

def matrix_from_pairs(n: int, pairs_dict: dict[tuple[int,int], float]) -> np.ndarray:
    M = np.ones((n, n), dtype=float)
    for (i, j), v in pairs_dict.items():
        if i == j:
            continue
        v = float(v)
        if v <= 0:
            raise ValueError("Score values must be > 0")
        M[i, j] = v
        M[j, i] = 1.0 / v
    return M

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "expert"

# ---------------- Email helper ---------------- #
def send_results_email(attachment_bytes: bytes, filename: str, expert_name: str, expert_credentials: str) -> tuple[bool, str]:
    cfg = st.secrets.get("email", {})
    host = cfg.get("host")
    port = int(cfg.get("port", 587))
    user = cfg.get("user")
    password = cfg.get("password")
    to_addr = cfg.get("to", DEFAULT_TO_EMAIL)

    if not host or not user or not password:
        return False, "SMTP settings missing (host/user/password). Add them to st.secrets['email'] to enable auto-email."

    msg = EmailMessage()
    msg["Subject"] = f"HPAI AHP Importance Results – {expert_name or 'Unnamed Expert'}"
    msg["From"] = user
    msg["To"] = to_addr
    body = (
        f"Automatic submission from the HPAI Transmission Routes app.\n\n"
        f"Expert: {expert_name or '(not provided)'}\n"
        f"Credentials/Institution: {expert_credentials or '(not provided)'}\n"
        f"Timestamp (UTC): {dt.datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        f"Attached: {filename}\n"
    )
    msg.set_content(body)
    msg.add_attachment(
        attachment_bytes,
        maintype="application",
        subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )
    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return True, f"Results emailed to {to_addr}"
    except Exception as e:
        return False, f"Email send failed: {e}"

# ---------------- Streamlit state ---------------- #
st.set_page_config(page_title="HPAI Transmission Routes – Wizard (Pairs)", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = 0
if "pairs" not in st.session_state:
    st.session_state.pairs = {}
if "expert_name" not in st.session_state:
    st.session_state.expert_name = ""
if "expert_credentials" not in st.session_state:
    st.session_state.expert_credentials = ""

# ---------------- Navigation helpers ---------------- #
def goto_page(p: int):
    st.session_state.page = max(0, min(P + 1, p))
    st.rerun()

def next_page():
    goto_page(st.session_state.page + 1)

def back_page():
    goto_page(st.session_state.page - 1)

def header_bar():
    total_pages = P + 2
    st.markdown(f"**Step {st.session_state.page+1} / {total_pages}**")
    st.progress((st.session_state.page+1) / total_pages)

def score_selectbox(key: str, default=1):
    options = list(SCORE_SCALE.keys())
    idx = options.index(next(k for k, v in SCORE_SCALE.items() if v == default))
    choice = st.selectbox("Score (1–9)", options, index=idx, key=key, label_visibility="collapsed")
    return SCORE_SCALE[choice]

def nav_buttons(show_back=True, show_next=True, next_label="Next"):
    cols = st.columns([1,1,6,2])
    with cols[0]:
        if show_back and st.button("◀ Back", key=f"back_{st.session_state.page}"):
            back_page()
    with cols[3]:
        if show_next and st.button(next_label, key=f"next_{st.session_state.page}"):
            next_page()

# ---------------- Pages ---------------- #
def page_intro():
    st.title("HPAI Transmission Routes")
    st.markdown("Experts enter pairwise comparisons for **Importance (Score 1–9)**.")

    # Clear, highlighted instructions with detailed scale meaning
    st.markdown(
        """
        <div style="background:#FFF3E0;border-left:6px solid #FF9800;padding:12px 14px;margin:8px 0 14px 0;">
          <div style="font-weight:800;color:#E65100;margin-bottom:8px;">Read before starting</div>
          <ul style="margin:0 0 8px 18px;">
            <li>You will be shown <b>two routes at a time</b>. Choose which is more important using a <b>Score 1–9</b>.</li>
            <li><b>Score meanings:</b>
              <ul>
                <li><b>1</b> = no difference between routes</li>
                <li><b>3</b>, <b>5</b>, <b>7</b> = moderate, strong, very strong</li>
                <li><b>2</b>, <b>6</b>, <b>8</b> = between anchors (1–3, 5–7, 7–9) → respectively: <i>equal to moderate</i>, <i>moderate to strong</i>, <i>strong to very strong</i></li>
                <li><b>9</b> = extreme difference</li>
              </ul>
            </li>
            <li>If the <b>header route</b> (top bar) is more important, choose a <b>Score &gt; 1</b>.</li>
            <li>If the <b>left route</b> (the one in the box) is more important, tick <b>Reciprocal</b> to apply <code>1/Score</code>.</li>
            <li>Complete all pairs, then click <b>Finish</b> to export results.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Show all routes", expanded=False):
        for i, r in enumerate(ROUTES, start=1):
            st.write(f"{i}. {r}")

    st.divider()
    st.subheader("Expert information (optional to start)")
    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        st.session_state.expert_name = st.text_input(
            "Your full name", value=st.session_state.expert_name, placeholder="e.g., Dr. Jane Doe"
        )
    with c2:
        st.session_state.expert_credentials = st.text_input(
            "Credentials / Institution", value=st.session_state.expert_credentials,
            placeholder="e.g., DVM, PhD – Example University"
        )

    st.info("Click **Start ▶** to begin. You can enter/edit your name later on the Finish page.")
    nav_buttons(show_back=False, show_next=True, next_label="Start ▶")

def page_pair(k: int):
    """k from 0..P-1 maps to a pair (i,j)."""
    (i, j) = PAIRS[k]
    header_bar()

    st.markdown(
        f"""
        <div style="padding:10px 14px;border-left:6px solid #1976D2;background:#E3F2FD;margin-bottom:10px;">
          <div style="font-size:18px;font-weight:800;color:#0D47A1;">
            Route {i+1}: {ROUTES[i]}
          </div>
          <div style="margin-top:6px;color:#0D47A1;">
            <b>How to score:</b> Pick a <b>Score &gt; 1</b> if the header route (above) is more important than the route shown on the left.<br/>
            If the left route is more important, tick <b>Reciprocal</b> to apply <code>1/Score</code>.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader(f"Pair {k+1} of {P}")

    col_left, col_right = st.columns([2.8, 1.6])

    default_score, default_recip = 1, False
    if (i, j) in st.session_state.pairs:
        v = float(st.session_state.pairs[(i, j)])
        if v >= 1:
            default_score, default_recip = int(round(v)), False
        else:
            default_score, default_recip = int(round(1.0 / v)), True

    with col_left:
        st.markdown("**Route to compare**")
        st.container(border=True).write(f"Route {j+1}: {ROUTES[j]}")

    with col_right:
        st.markdown("**Score & Reciprocal**")
        st.caption("Score (1–9)")
        score_val = score_selectbox(f"score_{i}_{j}", default=default_score)
        st.caption("Reciprocal?")
        recip = st.checkbox(" ", key=f"rec_{i}_{j}", value=default_recip)
        stored = (1.0 / score_val) if recip else float(score_val)
        st.markdown(
            f"<div style='margin-top:10px;color:#424242;'>Stored value: <code>{stored:.5f}</code></div>",
            unsafe_allow_html=True
        )

    # Persist choice
    st.session_state.pairs[(i, j)] = stored

    nav_buttons(show_back=True, show_next=True)

def page_finish():
    header_bar()
    st.header("Finish")

    # Fill any missing with neutral 1.0
    for (i, j) in PAIRS:
        st.session_state.pairs.setdefault((i, j), 1.0)

    # Compute AHP
    M = matrix_from_pairs(N, st.session_state.pairs)
    CR, CI, lam, w = consistency_ratio(M)

    df = pd.DataFrame({"Route": ROUTES, "Importance_w": w})
    df["Importance_w (%)"] = (df["Importance_w"] * 100).round(4)
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("Rank_Importance").reset_index(drop=True)

    st.success("All pairs completed. Click **Export & Send** to save your results.")
    st.write(f"Consistency ratio (CR): **{CR:.3f}**")

    # Ensure name/credentials can be set before export
    st.subheader("Your details for the file/email")
    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        st.session_state.expert_name = st.text_input(
            "Full name", value=st.session_state.expert_name, placeholder="e.g., Dr. Jane Doe"
        )
    with c2:
        st.session_state.expert_credentials = st.text_input(
            "Credentials / Institution", value=st.session_state.expert_credentials,
            placeholder="e.g., DVM, PhD – Example University"
        )

    def build_excel() -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine=ENGINE) as writer:
            df[["Rank_Importance", "Route", "Importance_w", "Importance_w (%)"]].to_excel(
                writer, sheet_name="Results", index=False
            )
            pd.DataFrame(M, index=ROUTES, columns=ROUTES).to_excel(writer, sheet_name="Aggregated_Matrix")
            pd.DataFrame({
                "Criterion": ["Importance"], "n (routes)": [N],
                "lambda_max": [lam], "CI": [CI], "CR": [CR]
            }).to_excel(writer, sheet_name="Consistency", index=False)
            pd.DataFrame({
                "Field": ["Expert name", "Credentials/Institution", "Timestamp (UTC)"],
                "Value": [
                    st.session_state.expert_name or "(not provided)",
                    st.session_state.expert_credentials or "(not provided)",
                    dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                ],
            }).to_excel(writer, sheet_name="Metadata", index=False)
        buf.seek(0)
        return buf.read()

    expert_name = st.session_state.expert_name.strip()
    expert_credentials = st.session_state.expert_credentials.strip()
    filename = f"hpai_importance_results_{slugify(expert_name)}.xlsx"

    cols = st.columns([2, 2])
    with cols[0]:
        if st.button("💾 Export & Send", key="export_send"):
            try:
                data = build_excel()
                ok, msg = send_results_email(data, filename, expert_name, expert_credentials)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)
                st.download_button(
                    label=f"Download results – {filename}",
                    data=data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_btn"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    nav_buttons(show_back=True, show_next=False)

# ---------------- Router ---------------- #
page = st.session_state.page
if page == 0:
    page_intro()
elif 1 <= page <= P:
    page_pair(page - 1)
else:
    page_finish()
