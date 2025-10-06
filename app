import json, math
from typing import Dict, List, Tuple
import numpy as np
import streamlit as st

# ---------------- AHP core ---------------- #
RI = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49}

def matrix_from_pairs(n:int, pairs:Dict[str,float])->np.ndarray:
    A = np.ones((n,n), dtype=float)
    for key,val in pairs.items():
        i,j = map(int, key.split(','))
        v = float(val); A[i,j]=v; A[j,i]=1.0/v
    return A

def principal_eigenvector(A:np.ndarray)->Tuple[np.ndarray,float]:
    vals, vecs = np.linalg.eig(A)
    k = np.argmax(np.real(vals)); lam = float(np.real(vals[k]))
    v = np.abs(np.real(vecs[:,k])); w = v / v.sum()
    return w, lam

def consistency_metrics(A:np.ndarray):
    n = A.shape[0]; _, lam = principal_eigenvector(A)
    CI = (lam - n)/(n-1) if n>2 else 0.0; CR = CI / (RI.get(n, RI[10]) or 1.0)
    return lam, CI, CR

def geom_mean_mats(mats):
    logsum = np.zeros_like(mats[0], dtype=float)
    for M in mats: logsum += np.log(M)
    G = np.exp(logsum/len(mats)); np.fill_diagonal(G,1.0)
    for i in range(G.shape[0]):
        for j in range(i+1,G.shape[1]): G[j,i] = 1.0 / G[i,j]
    return G

def default_pairs(n:int): return {f"{i},{j}":1.0 for i in range(n) for j in range(i+1,n)}

# ---------------- App config ---------------- #
st.set_page_config(page_title="AHP: Transmission Routes", layout="wide")

# Mode from query string (default: expert)
def get_mode()->str:
    try:
        qp = st.query_params            # Streamlit ≥1.30
        mode = qp.get("mode", ["expert"])[0]
    except Exception:
        qp = st.experimental_get_query_params()  # older API
        mode = (qp.get("mode", ["expert"])[0])
    return mode.lower().strip()

MODE = get_mode()          # "expert" or "coord"
IS_COORD = MODE in ("coord", "coordinator", "admin")

# Optional lightweight guard for accidental exposure:
code_ok = True
if IS_COORD:
    with st.sidebar:
        code = st.text_input("Coordinator code (optional)", type="password", help="Leave blank if not set.")
        # Set a code here if you want to restrict coordinator view, e.g.: REQUIRED="mycode"
        REQUIRED = ""  # e.g., "mysecret123"
        code_ok = (REQUIRED == "") or (code == REQUIRED)

st.title("AHP for Virus Transmission Routes")

routes = [
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
labels = routes; n = len(labels)

# Session state for pairwise inputs
if "pairs_I" not in st.session_state: st.session_state["pairs_I"] = default_pairs(n)
if "pairs_L" not in st.session_state: st.session_state["pairs_L"] = default_pairs(n)

# --------- STEP 1 (visible to experts) --------- #
st.header("1) Transmission routes (fixed list)")
st.dataframe({"#": list(range(1, n+1)), "Route": labels}, use_container_width=True, hide_index=True)

# --------- STEP 2 (visible to experts) --------- #
def pairwise_editor(title, key_prefix, state_key):
    st.header(title)
    st.caption("Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme. "
               "Always read as the **first** item vs the **second**.")
    pairs = st.session_state[state_key]
    cols = st.columns(3); per_col = (n*(n-1)//2 + 2)//3
    idx = 0; c = 0
    for i in range(n):
        for j in range(i+1, n):
            if idx and idx % per_col == 0: c = min(c+1, 2)
            with cols[c]:
                v = float(pairs.get(f"{i},{j}", 1.0))
                pairs[f"{i},{j}"] = st.number_input(
                    f"{labels[i]}  vs  {labels[j]}",
                    min_value=1.0, max_value=9.0, step=1.0, value=v,
                    key=f"{key_prefix}_{i}_{j}")
            idx += 1
    st.session_state[state_key] = pairs

pairwise_editor("2) Pairwise comparisons — IMPORTANCE", "imp", "pairs_I")
pairwise_editor("2) Pairwise comparisons — LIKELIHOOD", "lik", "pairs_L")

# --------- Expert export (still visible in expert mode) --------- #
def dl_json(pairs, criterion):
    data = {"criterion": criterion, "labels": labels, "pairs": pairs,
            "scale": "Saaty 1–9 (reciprocal implied)"}
    return json.dumps(data, indent=2).encode("utf-8")

st.subheader("Export your answers (send these two files to the coordinator)")
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Importance JSON",
        data=dl_json(st.session_state["pairs_I"], "Importance"),
        file_name="expert_importance.json", mime="application/json")
with c2:
    st.download_button("Download Likelihood JSON",
        data=dl_json(st.session_state["pairs_L"], "Likelihood"),
        file_name="expert_likelihood.json", mime="application/json")

# ==================== COORDINATOR VIEW ONLY ==================== #
if IS_COORD and code_ok:
    st.divider()
    st.markdown("### Coordinator view (Steps 3–5) — hidden from experts")

    # Step 3: Results per criterion (from uploaded expert JSONs aggregated)
    st.subheader("Upload JSONs from experts")
    up_imp = st.file_uploader("Upload *Importance* JSON files", type=["json"], accept_multiple_files=True, key="agg_imp")
    up_lik = st.file_uploader("Upload *Likelihood* JSON files", type=["json"], accept_multiple_files=True, key="agg_lik")

    def load_group(files):
        if not files: return None
        mats=[]; base=None
        for f in files:
            d = json.loads(f.read().decode("utf-8"))
            if base is None: base = d["labels"]
            if d["labels"] != base:
                st.error("All uploaded files must have the same labels and ordering.")
                return None
            mats.append(matrix_from_pairs(len(base), d["pairs"]))
        return geom_mean_mats(mats), base

    G_I, base_I = load_group(up_imp) if up_imp else (None, None)
    G_L, base_L = load_group(up_lik) if up_lik else (None, None)

    def show_weights(name, G):
        if G is None: 
            st.info(f"Upload {name} JSON files to compute group weights.")
            return None
        w, _ = principal_eigenvector(G); lam, CI, CR = consistency_metrics(G)
        st.write(f"**Aggregated {name}** — λmax={lam:.4f} · CI={CI:.4f} · CR={CR:.4f}")
        st.dataframe({"label": labels, "weight": w})
        return w

    st.header("3) Aggregated weights (group)")
    wI_g = show_weights("Importance", G_I)
    wL_g = show_weights("Likelihood", G_L)

    # Step 4 & 5: Combined risk
    if (wI_g is not None) and (wL_g is not None):
        st.header("4) Combined risk = normalized (Importance × Likelihood)")
        prod = wI_g * wL_g; risk = prod / prod.sum()
        st.dataframe({"label": labels, "importance": wI_g, "likelihood": wL_g, "risk": risk})
        import io, csv
        def tocsv(rows, header):
            s = io.StringIO(); w = csv.writer(s); w.writerow(header); w.writerows(rows); return s.getvalue().encode()
        risk_rows = [[lbl, float(a), float(b), float(r)] for lbl,a,b,r in zip(labels, wI_g, wL_g, risk)]
        st.download_button("Download Combined Risk CSV",
                           tocsv(risk_rows, ["label","importance","likelihood","risk"]),
                           "risk_scores.csv", mime="text/csv")

elif IS_COORD and not code_ok:
    st.warning("Enter the correct coordinator code to view Steps 3–5.")

