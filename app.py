import io
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# ============================================================
# Preferencias dicotómicas P / Pᶜ — Demo académica
# ============================================================

st.set_page_config(
    page_title="Preferencias dicotómicas — demo académica",
    page_icon="🧠",
    layout="wide"
)

# -------------------- Estilo --------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .card {
        border: 1px solid rgba(200,200,200,0.15);
        border-radius: 14px;
        padding: 16px;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }
    .muted { color: rgba(220,220,220,0.75); }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Preferencias dicotómicas (P / Pᶜ)")
st.caption("Formalización limpia, visualización sobria y lectura económica.")

# ============================================================
# Modelo de preferencias
# ============================================================

def u(in_P: bool) -> int:
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def symbol(i: int, j: int, P: List[bool]) -> str:
    if i == j:
        return "∼"
    if strict_pref(i, j, P):
        return "≻"
    if indifferent(i, j, P):
        return "∼"
    return "⪰" if weak_pref(i, j, P) else "⪯"

def symbol_to_num(s: str) -> float:
    return {"≻": 2.0, "⪰": 1.5, "∼": 1.0, "⪯": 0.5}[s]

def build_relation(names: List[str], P: List[bool]):
    n = len(names)
    S = np.empty((n, n), dtype=object)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = symbol(i, j, P)
            S[i, j] = s
            Z[i, j] = symbol_to_num(s)
    return S, Z

def complete(P: List[bool]) -> bool:
    n = len(P)
    return all(
        weak_pref(i, j, P) or weak_pref(j, i, P)
        for i in range(n) for j in range(n)
    )

def transitive(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and not weak_pref(i, k, P):
                    return False
    return True

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuración")

    n = st.slider("Número de alternativas", 4, 20, 10, 1)
    names = [f"x{i+1}" for i in range(n)]

    st.subheader("Definir P")
    P = [st.checkbox(f"{names[i]} ∈ P", value=(i % 2 == 0)) for i in range(n)]

# ============================================================
# Datos derivados
# ============================================================

S, Z = build_relation(names, P)
P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

# ============================================================
# KPIs
# ============================================================

c1, c2, c3, c4 = st.columns(4)
c1.metric("|P|", len(P_set))
c2.metric("|Pᶜ|", len(Pc_set))
c3.metric("Completitud", "OK" if complete(P) else "NO")
c4.metric("Transitividad", "OK" if transitive(P) else "NO")

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs(["Modelo", "Visualización", "Comparación"])

# ---------------- Modelo ----------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        """
        **Definición**

        - El conjunto de alternativas se particiona en dos bloques: **P** y **Pᶜ**
        - La preferencia estricta es:
          \n\n`x ≻ y  ⇔  x ∈ P  y  y ∈ Pᶜ`
        - Dentro de cada bloque hay **indiferencia**

        **Representación**
        \n\n`u(x) = 1` si `x ∈ P`, `u(x) = 0` si `x ∈ Pᶜ`  
        \n`x ⪰ y  ⇔  u(x) ≥ u(y)`
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Visualización ----------------
with tab2:
    colA, colB = st.columns([1, 1.1])

    with colA:
        st.subheader("Grafo bipartito (P → Pᶜ)")
        pos = {}
        top = [i for i in range(n) if P[i]]
        bot = [i for i in range(n) if not P[i]]

        if top:
            xs = np.linspace(0.1, 0.9, len(top))
            for k, i in enumerate(top):
                pos[i] = (xs[k], 0.75)
        if bot:
            xs = np.linspace(0.1, 0.9, len(bot))
            for k, i in enumerate(bot):
                pos[i] = (xs[k], 0.25)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in pos],
            y=[pos[i][1] for i in pos],
            mode="markers+text",
            text=[names[i] for i in pos],
            textposition="top center",
            marker=dict(
                size=22,
                color=[1 if P[i] else 0 for i in pos],
                colorscale="Plasma"
            ),
            showlegend=False
        ))

        for i in top:
            for j in bot:
                fig.add_trace(go.Scatter(
                    x=[pos[i][0], pos[j][0]],
                    y=[pos[i][1], pos[j][1]],
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                ))

        fig.update_layout(
            height=450,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Heatmap de la relación (hover)")
        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(go.Heatmap(
            z=Z,
            x=names,
            y=names,
            text=hover,
            hoverinfo="text"
        ))
        fig_h.update_layout(height=450)
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("**Tabla simbólica**")
        st.dataframe(pd.DataFrame(S, index=names, columns=names))

# ---------------- Comparación ----------------
with tab3:
    st.markdown(
        """
        **Lectura económica**

        - Este modelo induce **exactamente dos clases de equivalencia**
        - No hay ranking intra-bloque
        - Cualquier refinamiento exige supuestos adicionales

        **Comparación**
        - *Dicotómica*: máxima rigidez (este caso)
        - *Lexicográfica*: jerarquía de criterios
        - *Umbral*: indiferencia en banda
        - *Orden total*: ranking completo (suposición fuerte)
        """
    )

    msg = """Formalicé tu construcción como una preferencia dicotómica P/Pᶜ.

La estructura es deliberadamente rígida: induce dos clases de equivalencia
y no permite refinamiento ordinal sin introducir supuestos adicionales
(lexicografía, umbrales o atributos).

Eso es precisamente lo interesante del modelo.
"""
    st.text_area("Mensaje listo para enviar", msg, height=180)
