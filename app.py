from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# Preferencias dicotómicas P / Pᶜ — Demo limpia (sin matplotlib)
# ============================================================

st.set_page_config(
    page_title="Preferencias dicotómicas — demo limpia",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .card { border: 1px solid rgba(250,250,250,0.12); border-radius: 14px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
    .muted { color: rgba(250,250,250,0.72); font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Preferencias dicotómicas (P / Pᶜ)")
st.caption("Formalización sobria + estructura visible. Nada de ruido visual, todo argumento.")

# ----------------- Modelo -----------------
def u(in_P: bool) -> int:
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    # x ⪰ y  ⇔  (x ∈ P) ∨ (y ∈ Pᶜ)
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    # x ≻ y  ⇔  (x ∈ P) ∧ (y ∈ Pᶜ)
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    # x ~ y si están en el mismo bloque
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def symbol(i: int, j: int, P: List[bool]) -> str:
    if i == j:
        return "∼"
    if strict_pref(i, j, P):
        return "≻"
    if indifferent(i, j, P):
        return "∼"
    return "⪰" if weak_pref(i, j, P) else "⪯"

def sym_to_num(s: str) -> float:
    return {"≻": 2.0, "⪰": 1.5, "∼": 1.0, "⪯": 0.5}.get(s, 0.0)

def build_relation(names: List[str], P: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(names)
    S = np.empty((n, n), dtype=object)
    Z = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = symbol(i, j, P)
            S[i, j] = s
            Z[i, j] = sym_to_num(s)
    return S, Z

def complete(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, P) or weak_pref(j, i, P)):
                return False
    return True

def transitive(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and not weak_pref(i, k, P):
                    return False
    return True

def bipartite_positions(P: List[bool]) -> Dict[int, Tuple[float, float]]:
    n = len(P)
    top = [i for i in range(n) if P[i]]
    bot = [i for i in range(n) if not P[i]]

    pos: Dict[int, Tuple[float, float]] = {}
    if top:
        xs = np.linspace(0.08, 0.92, len(top))
        for k, i in enumerate(top):
            pos[i] = (float(xs[k]), 0.78)
    if bot:
        xs = np.linspace(0.08, 0.92, len(bot))
        for k, i in enumerate(bot):
            pos[i] = (float(xs[k]), 0.22)
    return pos

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Controles")
    n = st.slider("Tamaño de X", 4, 24, 10, 1)

    default_names = ", ".join([f"x{i+1}" for i in range(n)])
    names_str = st.text_input("Etiquetas (coma)", value=default_names)
    names = [s.strip() for s in names_str.split(",") if s.strip()]
    if len(names) != n:
        st.warning("Etiquetas inválidas. Uso x1,…,xn.")
        names = [f"x{i+1}" for i in range(n)]

    st.divider()
    mode = st.radio("Definir P", ["Manual", "Aleatorio (demo)"], index=0)
    if mode.startswith("Aleatorio"):
        p_share = st.slider("Proporción en P", 0.10, 0.90, 0.50, 0.05)
        seed = st.number_input("Semilla", value=11, step=1)
        rng = np.random.default_rng(int(seed))
        P = list((rng.random(n) < p_share).astype(bool))
    else:
        P = [st.checkbox(f"{names[i]} ∈ P", value=(i % 2 == 0)) for i in range(n)]

    st.divider()
    st.subheader("Visualización")
    show_symbol_table = st.checkbox("Mostrar tabla de símbolos", value=True)

# ----------------- Derivados -----------------
S, Z = build_relation(names, P)
P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("|P|", len(P_set))
k2.metric("|Pᶜ|", len(Pc_set))
k3.metric("Completitud", "OK" if complete(P) else "NO")
k4.metric("Transitividad", "OK" if transitive(P) else "NO")

tab1, tab2, tab3 = st.tabs(["Modelo", "Visualización", "Remate"])

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        r"""
### Formalización
Sea \(X\) un conjunto de alternativas y \(P\subset X\); \(P^c = X\setminus P\).

- Preferencia estricta:  \(x \succ y \iff (x\in P)\land(y\in P^c)\)
- Indiferencia intra-bloque: \(x\sim y\) si ambos están en \(P\) o ambos en \(P^c\)
- Representación ordinal indicadora:
  \[
  u(x)=\mathbf{1}\{x\in P\}\in\{0,1\},\quad x\succeq y \iff u(x)\ge u(y).
  \]

**Lectura clave:** el modelo induce exactamente **dos clases de equivalencia**. No hay ranking fino dentro de cada bloque.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**P =**", P_set if P_set else ["∅"])
    st.write("**Pᶜ =**", Pc_set if Pc_set else ["∅"])
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    colA, colB = st.columns([1.0, 1.05], gap="large")

    with colA:
        st.subheader("Grafo bipartito de dominancia estricta (P → Pᶜ)")
        st.markdown('<div class="muted">Todo ≻ va de P hacia Pᶜ. Dentro de bloques no hay orden: hay indiferencia.</div>', unsafe_allow_html=True)

        pos = bipartite_positions(P)
        top = [i for i in range(n) if P[i]]
        bot = [i for i in range(n) if not P[i]]

        if not top or not bot:
            st.warning("No hay relaciones estrictas (todo en P o todo en Pᶜ).")
        else:
            fig = go.Figure()

            node_ids = list(pos.keys())
            fig.add_trace(
                go.Scatter(
                    x=[pos[i][0] for i in node_ids],
                    y=[pos[i][1] for i in node_ids],
                    mode="markers+text",
                    text=[names[i] for i in node_ids],
                    textposition="top center",
                    hovertext=[f"{names[i]} | {'P' if P[i] else 'Pᶜ'} | u={u(P[i])}" for i in node_ids],
                    hoverinfo="text",
                    marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale="Plasma"),
                    showlegend=False,
                )
            )

            # edges in one trace (clean)
            xs, ys = [], []
            for i in top:
                for j in bot:
                    xs += [pos[i][0], pos[j][0], None]
                    ys += [pos[i][1], pos[j][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False))

            fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Heatmap de la relación (hover = símbolo)")
        st.markdown('<div class="muted">Nada de texto encima del heatmap: el símbolo vive en hover. Así se ve pro.</div>', unsafe_allow_html=True)

        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(go.Heatmap(z=Z, x=names, y=names, text=hover, hoverinfo="text"))
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        if show_symbol_table:
            st.markdown("#### Tabla simbólica")
            st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

with tab3:
    st.subheader("El remate (para que suene a colega, no a fan)")
    st.markdown(
        """
- Esto no es “un ranking”: es un **preorden dicotómico**.
- Lo elegante es su **rigidez**: dos clases de equivalencia y nada intra-bloque.
- Si alguien quiere “más orden”, debe **pagar** con supuestos: lexicografía, umbrales, atributos medibles, etc.
"""
    )

    msg = f"""Formalizé tu construcción como una preferencia dicotómica P/Pᶜ.

La estructura es deliberadamente rígida: induce exactamente dos clases de equivalencia
(P y Pᶜ) y no permite refinamiento ordinal dentro de cada bloque sin introducir supuestos
adicionales (lexicografía, umbrales o atributos).

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area("Mensaje listo para enviar", msg, height=210)
