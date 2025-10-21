def page_pair(k: int):
    """k from 0..P-1 maps to a pair (i,j). UI shows header route only once."""
    (i, j) = PAIRS[k]
    header_bar()

    # ---- Header with anchor (left) route ----
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

    # ---- Two-column layout: left shows the compared route; right shows controls ----
    col_left, col_right = st.columns([2.8, 1.6], vertical_alignment="top")

    # Remember prior choice (if any)
    default_score, default_recip = 1, False
    if (i, j) in st.session_state.pairs:
        v = float(st.session_state.pairs[(i, j)])
        if v >= 1:
            default_score, default_recip = int(round(v)), False
        else:
            default_score, default_recip = int(round(1.0 / v)), True

    with col_left:
        # Compared route in a box
        st.markdown("**Route to compare**")
        st.container(border=True).write(f"Route {j+1}: {ROUTES[j]}")

    with col_right:
        # Controls
        st.markdown("**Score & Reciprocal**")
        # (collapsed label to keep a clean look; caption labels above)
        st.caption("Score (1–9)")
        score_val = score_selectbox(f"score_{i}_{j}", default=default_score)

        st.caption("Reciprocal?")
        recip = st.checkbox(" ", key=f"rec_{i}_{j}", value=default_recip)

        # Live preview of the stored numeric value
        preview = (1.0 / score_val) if recip else float(score_val)
        st.markdown(
            f"<div style='margin-top:10px;color:#424242;'>Stored value: "
            f"<code>{preview:.5f}</code></div>",
            unsafe_allow_html=True
        )

    # Persist normalized value for (i,j) (we always store i<j)
    st.session_state.pairs[(i, j)] = (1.0 / score_val) if recip else float(score_val)

    # Nav
    nav_buttons(show_back=True, show_next=True)
