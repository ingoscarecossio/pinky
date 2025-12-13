import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Preference Collapse",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #0f0f14, #151520);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Preference Collapse")
st.caption("Cuando una preferencia binaria destruye un mundo continuo")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🎛 Control")
    n = st.slider("Número de alternativas", 100, 1500, 600, 50)
    p_share = st.slider("Proporción Pinky (P)", 0.1, 0.9, 0.5, 0.05)
    animate = st.checkbox("Animar colapso", value=True)
    seed = st.number_input("Semilla", value=42, step=1)

# -----------------------------
# Generate world
# -----------------------------
rng = np.random.default_rng(seed)
X = rng.normal(0, 1, size=(n, 2))   # mundo continuo
is_pinky = rng.random(n) < p_share

u = np.where(is_pinky, 1.0, 0.0)

# -----------------------------
# Initial state
# -----------------------------
st.subheader("1️⃣ Mundo original (aparente complejidad)")

fig0 = go.Figure()
fig0.add_trace(go.Scatter(
    x=X[:,0], y=X[:,1],
    mode="markers",
    marker=dict(
        size=6,
        color=X[:,0]**2 + X[:,1]**2,
        colorscale="Viridis",
        opacity=0.85
    ),
    hoverinfo="skip"
))

fig0.update_layout(
    height=500,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig0, use_container_width=True)

# -----------------------------
# Collapse animation
# -----------------------------
st.subheader("2️⃣ Aplicar preferencia Pinky")

collapse = st.slider(
    "Intensidad de colapso",
    0.0, 1.0, 0.0, 0.01
)

Y_collapsed = (1 - collapse) * X[:,1] + collapse * (u * 2 - 1)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=X[:,0],
    y=Y_collapsed,
    mode="markers",
    marker=dict(
        size=6,
        color=u,
        colorscale=[[0, "#ff4d4d"], [1, "#7CFF00"]],
        opacity=0.9
    ),
    hoverinfo="skip"
))

fig1.update_layout(
    height=500,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Information collapse
# -----------------------------
st.subheader("3️⃣ Colapso de información")

info_loss = 1 - (2 / n)

col1, col2, col3 = st.columns(3)

col1.metric("Alternativas originales", f"{n}")
col2.metric("Clases inducidas", "2")
col3.metric("Pérdida de resolución", f"{info_loss:.1%}")

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; font-size:26px; color:#dddddd;">
Esto no es una preferencia rica.<br>
<b>Es un interruptor.</b>
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div style="text-align:center; font-size:22px; margin-top:10px; color:#aaaaaa;">
u(x) ∈ {0, 1}
</div>
""",
unsafe_allow_html=True
)
