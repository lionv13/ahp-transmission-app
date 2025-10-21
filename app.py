# -*- coding: utf-8 -*-
"""
HPAI Transmission Routes – Expert Scoring Wizard (Importance only)
Pages:
  0) Intro
  1..N) Route i vs remaining routes (pairwise pages)
  N+1) Finish (export + email)

Run: streamlit run app.py
"""

from __future__ import annotations
import io, os, re, datetime as dt, subprocess, sys, smtplib
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
        import openpyxl
        ENGINE = "openpyxl"

# ---------------- Configuration ---------------- #
# Put your default recipient here; can be overridden by st.secrets["email"]["to"]
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

SAATY_RI = {
    1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41,
    9:1.45, 10:1.49, 11:1.51, 12:1.48, 13:1.56, 14:1.57, 15:1.59
}

SCORE_SCALE = {
    "1 (equal)": 1, "2 (between 1–3)": 2, "3 (moderate)": 3, "4 (between 3–5)": 4,
    "5 (strong)": 5, "6 (between 5–7)": 6, "7 (very strong)": 7, "8 (between 7–9)": 8, "9 (extreme)": 9,
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
    """pairs_dict has only i<j. Build full reciprocal matrix."""
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
    """
    Sends email with attachment using st.secrets["email"] config if present:
      host, port, user, password, to
    Returns (ok, message).
    """
    cfg = st.secrets.get("email", {})
    host = cfg.get("host")
    port = int(cfg.get("port", 587))
    user = cfg.get("user")
    password = cfg.get("password")
    to_addr = cfg.get("to", DEFAULT_TO_EMAIL)

    if not host or not user or not password:
        return False, "SMTP settings missing (host/user/password). Add them to st.secrets['email'] to enable auto-email."

    msg = EmailMessage()
    msg["Subject"] = f"HPAI AHP Importance Results – {expert_name}"
    msg["From"] = user
    msg["To"] = to_addr
    body = (
        f"Automatic submission from the HPAI Transmission Routes app.\n\n"
        f"Expert: {expert_name}\n"
        f"Credentials/Institution: {expert_credentials}\n"
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
st.set_page_config(page_title="HPAI Transmission Routes – Wizard", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = 0  # 0=intro, 1..N routes, N+1 finish

if "pairs" not in st.session_state:
    # store float values for i<j
    st.session_state.pairs = {}

if "expert_name" not in st.session_state:
    st.session_state.expert_name = ""

if "expert_credentials" not in st.session_state:
    st.session_state.expert_credentials = ""

# ---------------- UI helpers ---------------- #
def header_bar():
    total_pages = N + 2
    st.markdown(
        f"**Step {st.session_state.page+1} / {total_pages}**",
        help="Use Next/Back to navigate. Your inputs are saved automatically."
    )
    st.progress((st.session_state.page+1)/total_pages)

def score_selectbox(key: str, default=1):
    options = list(SCORE_SCALE.keys())
    idx = options.index(next(k for k, v in SCORE_SCALE.items() if v == default))
    choice = st.selectbox("Score (1–9)", options, index=idx, key=key, label_visibility="collapsed")
    return SCORE_SCALE[choice]

def nav_buttons(show_back=True, show_next=True, next_label="Next"):
    cols = st.columns([1,1,6,2])
    with cols[0]:
        if show_back and st.button("◀ Back"):
            st.session_state.page = max(0, st.session_state.page - 1)
            st.rerun()
    with cols[3]:
        if show_next and st.button(next_label):
            st.session_state.page = min(N + 1, st.session_state.page + 1)
            st.rerun()

# ---------------- Pages ---------------- #
def page_intro():
    st.title("HPAI Transmission Routes")
    st.markdown("Experts enter pairwise comparisons for **Importance (Score 1–9)**.")

    st.markdown(
        """
        <div style="background:#FFF3E0;border-left:6px solid #FF9800;padding:12px 14px;margin:8px 0 14px 0;">
          <div style="font-weight:700;color:#E65100;margin-bottom:6px;">Read before starting</div>
          <ul style="margin:0 0 4px 18px;">
            <li>On each page you’ll evaluate one route against the remaining routes.</li>
            <li>If the <b>left route</b> is more important, choose a score <b>&gt; 1</b>.
                If the <b>right route</b> is more important, tick <b>Reciprocal</b> (this applies <b>1/score</b>).</li>
            <li>Complete all pages, then click <b>Finish</b> to export the results.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Show all routes", expanded=False):
        for i, r in enumerate(ROUTES, start=1):
            st.write(f"{i}. {r}")

    st.divider()
    st.subheader("Expert information")
    c1, c2 = st.columns([1.4, 1.6])
    with c1:
        st.session_state.expert_name = st.text_input(
            "Your full name*", value=st.session_state.expert_name, placeholder="e.g., Dr. Jane Doe"
        )
    with c2:
        st.session_state.expert_credentials = st.text_input(
            "Credentials / Institution", value=st.session_state.expert_credentials,
            placeholder="e.g., DVM, PhD – Example University"
        )

    disabled = (st.session_state.expert_name.strip() == "")
    st.info("Click **Next** to begin scoring. You can go back anytime.")
    nav_buttons(show_back=False, show_next=True, next_label="Start ▶" if not disabled else "Enter name to continue ▶")
    if disabled:
        st.stop()

def page_route(i: int):
    """Page for focal route i: compare i vs j>i."""
    st.header_bar = header_bar()
    st.subheader(f"Route {i+1} of {N}")
    st.markdown(f"**Left route (focal):** {ROUTES[i]}")

    st.markdown(
        "<div style='color:#BF360C;font-weight:700;margin:6px 0 10px 0;'>"
        "Pick a score >1 if the left route is more important; tick Reciprocal if the right route is more important."
        "</div>",
        unsafe_allow_html=True
    )

    # Build rows for j=i+1..N-1
    for j in range(i + 1, N):
        with st.container(border=True):
            st.markdown(f"**Compare with (right route):** {ROUTES[j]}")
            key_val = f"score_{i}_{j}"
            key_rec = f"rec_{i}_{j}"

            # Defaults: existing session value or 1
            default_score = 1
            default_recip = False
            if (i, j) in st.session_state.pairs:
                # pairs store actual value; infer if it's reciprocal by <=1
                v = float(st.session_state.pairs[(i, j)])
                if v >= 1:
                    default_score = int(round(v))
                    default_recip = False
                else:
                    # v = 1/score
                    inv = round(1.0 / v)
                    default_score = int(inv)
                    default_recip = True

            c = st.columns([2.2, 1.2, 1.2, 2.2])
            with c[0]:
                st.caption("Left route")
                st.write(ROUTES[i])
            with c[1]:
                st.caption("Score (1–9)")
                score_val = score_selectbox(key_val, default=default_score)
            with c[2]:
                st.caption("Reciprocal?")
                recip = st.checkbox(" ", key=key_rec, value=default_recip)
            with c[3]:
                st.caption("Right route")
                st.write(ROUTES[j])

            # Save to pairs as an always-upper value (i<j)
            st.session_state.pairs[(i, j)] = (1.0 / score_val) if recip else float(score_val)

    nav_buttons(show_back=True, show_next=True)

def page_finish():
    st.header_bar = header_bar()
    st.header("Finish")

    # Build matrix & results
    pairs = st.session_state.pairs
    # Ensure all pairs exist (default 1 if missing)
    for i in range(N - 1):
        for j in range(i + 1, N):
            pairs.setdefault((i, j), 1.0)

    M = matrix_from_pairs(N, pairs)
    CR, CI, lam, w = consistency_ratio(M)

    df = pd.DataFrame({"Route": ROUTES, "Importance_w": w})
    df["Importance_w (%)"] = (df["Importance_w"] * 100).round(4)
    df["Rank_Importance"] = df["Importance_w"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("Rank_Importance").reset_index(drop=True)

    st.success("All steps are complete. Click **Export & Send** to save your results.")
    st.write(f"Consistency ratio (CR): **{CR:.3f}**")

    # build Excel
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
                    st.session_state.expert_name,
                    st.session_state.expert_credentials,
                    dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
                ],
            }).to_excel(writer, sheet_name="Metadata", index=False)
        buf.seek(0)
        return buf.read()

    expert_name = st.session_state.expert_name.strip()
    expert_credentials = st.session_state.expert_credentials.strip()
    filename = f"hpai_importance_results_{slugify(expert_name)}.xlsx"

    c = st.columns([2, 2])
    with c[0]:
        if st.button("💾 Export & Send"):
            try:
                data = build_excel()
                # Attempt email (optional)
                ok, msg = send_results_email(data, filename, expert_name, expert_credentials)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

                st.download_button(
                    label=f"Download results – {filename}",
                    data=data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")

    nav_buttons(show_back=True, show_next=False)

# -------------- Router -------------- #
header_bar()
page = st.session_state.page
if page == 0:
    page_intro()
elif 1 <= page <= N:
    page_route(page - 1)
else:
    page_finish()
