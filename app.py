# Streamlit WOW App: Preferencias Pinky
# Run with: streamlit run app.py

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Preferencias WOW", page_icon="🧠", layout="wide")
st.title("🧠 De esas fotos a una demo que humilla")
st.caption("Preferencias dicotómicas: P domina a P^c. Cine + tablero + pruebas formales.")

def u(is_pinky: bool) -> int:
    return 1 if is_pinky else 0

def weak_pref(i, j, is_pinky):
    return is_pinky[i] or (not is_pinky[j])

def strict_pref(i, j, is_pinky):
    return is_pinky[i] and (not is_pinky[j])

def indifferent(i, j, is_pinky):
    return (is_pinky[i] and is_pinky[j]) or ((not is_pinky[i]) and (not is_pinky[j]))

def relation_symbol(i, j, is_pinky):
    if i == j:
        return "∼"
    if strict_pref(i, j, is_pinky):
        return "≻"
    if indifferent(i, j, is_pinky):
        return "∼"
    return "⪰" if weak_pref(i, j, is_pinky) else "⪯"

def build_rel(names, is_pinky):
    n = len(names)
    rel = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            rel[i, j] = relation_symbol(i, j, is_pinky)
    return rel

def rel_numeric(rel):
    mapping = {"≻": 2.0, "∼": 1.0, "⪰": 1.5, "⪯": 0.5}
    return np.vectorize(lambda s: mapping.get(s, 0.0))(rel)

def check_completeness(is_pinky):
    n = len(is_pinky)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, is_pinky) or weak_pref(j, i, is_pinky)):
                return False
    return True

def check_transitivity(is_pinky):
    n = len(is_pinky)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, is_pinky) and weak_pref(j, k, is_pinky) and (not weak_pref(i, k, is_pinky)):
                    return False
    return True

with st.sidebar:
    st.header("🎛️ Control del show")
    n = st.slider("Número de alternativas |X|", 4, 24, 10, 1)
    default_names = ", ".join([f"x{i+1}" for i in range(n)])
    names_str = st.text_input("Nombres (coma)", value=default_names)
    names = [s.strip() for s in names_str.split(",") if s.strip()]
    if len(names) != n:
        names = [f"x{i+1}" for i in range(n)]
        st.warning("Nombres inválidos. Uso x1..xn.")

    st.subheader("🟣 Define P (Pinky)")
    mode = st.radio("Cómo definir P", ["Manual", "Aleatorio"], index=0)
    if mode == "Aleatorio":
        p_share = st.slider("Proporción en P", 0.1, 0.9, 0.5, 0.05)
        seed = st.number_input("Semilla", value=7, step=1)
        rng = np.random.default_rng(int(seed))
        is_pinky = list(rng.random(n) < p_share)
    else:
        is_pinky = []
        for i, nm in enumerate(names):
            is_pinky.append(st.checkbox(f"{nm} ∈ P", value=(i % 2 == 0)))

    st.subheader("🎬 Animación")
    autoplay = st.checkbox("Autoplay", value=True)
    speed = st.slider("Velocidad", 1, 10, 6, 1)
    max_edges = st.slider("Máximo de flechas", 20, 500, 180, 10)
    label_edges = st.checkbox("Poner '≻' en flechas", value=False)

P = [names[i] for i in range(n) if is_pinky[i]]
Pc = [names[i] for i in range(n) if not is_pinky[i]]

c1, c2 = st.columns([1.0, 1.2], gap="large")
with c1:
    st.subheader("Traducción matemática")
    st.info(f"P={P if P else '∅'} | P^c={Pc if Pc else '∅'}")
    comp = check_completeness(is_pinky)
    trans = check_transitivity(is_pinky)
    st.write(f"Completitud: {'✅' if comp else '❌'}")
    st.write(f"Transitividad: {'✅' if trans else '❌'}")

    util_df = pd.DataFrame({
        "Alternativa": names,
        "Grupo": ["P" if b else "P^c" for b in is_pinky],
        "u(x)": [u(b) for b in is_pinky],
    }).sort_values(["u(x)", "Alternativa"], ascending=[False, True])
    st.dataframe(util_df, use_container_width=True, hide_index=True)

with c2:
    st.subheader("Grafo animado (P → P^c)")
    P_idx = [i for i in range(n) if is_pinky[i]]
    Pc_idx = [i for i in range(n) if not is_pinky[i]]
    if len(P_idx) and len(Pc_idx):
        xP = np.linspace(0.1, 0.9, len(P_idx))
        xC = np.linspace(0.1, 0.9, len(Pc_idx))
        pos = {}
        for k, i in enumerate(P_idx): pos[i] = (xP[k], 0.82)
        for k, i in enumerate(Pc_idx): pos[i] = (xC[k], 0.18)
        edges = [(i, j) for i in P_idx for j in Pc_idx][:max_edges]
        total = len(edges)
        step = st.slider("Paso", 1, max(1, total), min(40, total) if total else 1, 1)
        ph = st.empty()

        def render(num):
            fig = go.Figure()
            node_x = [pos[i][0] for i in pos]
            node_y = [pos[i][1] for i in pos]
            node_color = [1 if is_pinky[i] else 0 for i in pos]
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                     text=[names[i] for i in pos], textposition="top center",
                                     marker=dict(size=22, color=node_color, colorscale="Plasma"),
                                     showlegend=False))
            for (i, j) in edges[:num]:
                x0, y0 = pos[i]; x1, y1 = pos[j]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                         line=dict(width=2), showlegend=False))
                if label_edges:
                    fig.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(y0+y1)/2], mode="text",
                                             text=["≻"], showlegend=False))
            fig.update_layout(height=520, xaxis=dict(visible=False), yaxis=dict(visible=False))
            return fig

        if autoplay and total:
            for s in range(1, min(step + speed*3, total) + 1, speed):
                ph.plotly_chart(render(s), use_container_width=True)
                time.sleep(max(0.02, 0.18 - 0.012*speed))
        else:
            ph.plotly_chart(render(step), use_container_width=True)

    st.subheader("Matriz")
    rel = build_rel(names, is_pinky)
    heat = rel_numeric(rel)
    fig_h = go.Figure(data=go.Heatmap(z=heat, x=names, y=names))
    for i in range(n):
        for j in range(n):
            fig_h.add_annotation(x=names[j], y=names[i], text=rel[i, j], showarrow=False)
    fig_h.update_layout(height=560)
    st.plotly_chart(fig_h, use_container_width=True)
