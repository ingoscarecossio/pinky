\
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------
# Streamlit WOW v3 (seminario)
# - ASCII-only source (portable)
# - Math rendered via LaTeX
# -----------------------------------------
st.set_page_config(page_title="Preferencias dicotomicas — seminario", page_icon="🧠", layout="wide")

st.title("🧠 Preferencias dicotomicas: de nota a estructura (nivel seminario)")
st.caption("Formalizacion canonica, propiedades, visualizacion estructural y comparacion con modelos cercanos.")

# ========= Modelo (dicotomico) =========
def u(is_pinky: bool) -> int:
    return 1 if is_pinky else 0

def weak_pref(i, j, P):
    # x \succeq y  <=> (x in P) OR (y in P^c)
    return P[i] or (not P[j])

def strict_pref(i, j, P):
    # x \succ y <=> (x in P) AND (y in P^c)
    return P[i] and (not P[j])

def indifferent(i, j, P):
    # x ~ y <=> both in P or both in P^c
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def symbol(i, j, P):
    if i == j:
        return "~"
    if strict_pref(i, j, P):
        return "succ"
    if indifferent(i, j, P):
        return "~"
    return "succeq" if weak_pref(i, j, P) else "preceq"

def build_rel(names, P):
    n = len(names)
    R = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            R[i, j] = symbol(i, j, P)
    return R

def rel_num(R):
    m = {"succ": 2.0, "~": 1.0, "succeq": 1.5, "preceq": 0.5}
    return np.vectorize(lambda s: m.get(s, 0.0))(R)

# ========= Axiomas =========
def complete(P):
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, P) or weak_pref(j, i, P)):
                return False
    return True

def transitive(P):
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and (not weak_pref(i, k, P)):
                    return False
    return True

# ========= Sidebar =========
with st.sidebar:
    st.header("⚙️ Experimento")
    n = st.slider("Cardinalidad de X", 4, 24, 10, 1)

    default_names = ", ".join([f"x{i+1}" for i in range(n)])
    names_str = st.text_input("Etiquetas (coma)", value=default_names)
    names = [s.strip() for s in names_str.split(",") if s.strip()]
    if len(names) != n:
        st.warning("Etiquetas inconsistentes. Se usan x1,...,xn.")
        names = [f"x{i+1}" for i in range(n)]

    st.divider()
    st.subheader("Particion P / P^c")
    mode = st.radio("Modo", ["Manual", "Aleatorio (demo)"], index=0)

    if mode.startswith("Aleatorio"):
        p_share = st.slider("Proporcion esperada en P", 0.1, 0.9, 0.5, 0.05)
        seed = st.number_input("Semilla", value=11, step=1)
        rng = np.random.default_rng(int(seed))
        P = list(rng.random(n) < p_share)
    else:
        P = []
        for i, nm in enumerate(names):
            P.append(st.checkbox(f"{nm} in P", value=(i % 2 == 0)))

    st.divider()
    st.subheader("Show")
    autoplay = st.checkbox("Animacion", value=True)
    speed = st.slider("Velocidad", 1, 10, 6, 1)
    max_edges = st.slider("Max flechas (P -> P^c)", 20, 500, 180, 10)
    show_edge_labels = st.checkbox("Etiqueta \\succ en flechas", value=False)

P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

tab1, tab2, tab3 = st.tabs(["Modelo", "Visualizacion", "Comparacion"])

# ========= TAB 1: Modelo =========
with tab1:
    st.subheader("Formalizacion canonica")
    st.markdown(r"""
Sea $X$ el conjunto de alternativas y $P \subset X$ un subconjunto distinguido. Con $P^c = X \setminus P$:

- Preferencia estricta:  $x \succ y \iff (x \in P) \land (y \in P^c)$
- Indiferencia intra-bloque:  $x \sim y$ si ambos estan en $P$ o ambos estan en $P^c$
- Representacion ordinal natural (indicadora):
$$
u(x)=\begin{cases}
1 & x \in P \\
0 & x \in P^c
\end{cases}
\qquad\Rightarrow\qquad x \succeq y \iff u(x)\ge u(y)
$$
""")

    st.info(f"P = {P_set if P_set else ['∅']}    |    P^c = {Pc_set if Pc_set else ['∅']}")

    st.subheader("Propiedades (axiomas)")
    c_ok = complete(P)
    t_ok = transitive(P)
    st.write("Completitud:", "✅ satisfecha" if c_ok else "❌ no satisfecha")
    st.write("Transitividad:", "✅ satisfecha" if t_ok else "❌ no satisfecha")

    st.markdown(r"""
**Lectura estructural (lo que impresiona a alguien experto):**
- Este preorden colapsa exactamente en **dos clases de equivalencia** ($P$ y $P^c$).
- Dentro de cada clase no hay informacion ordinal adicional.
- Cualquier refinamiento del ranking requiere **supuestos extra** (atributos, umbrales, lexicografia, etc.).
""")

    st.subheader("Utilidad indicadora y 'rigidez'")
    util_df = pd.DataFrame({
        "Alternativa": names,
        "Grupo": ["P" if P[i] else "P^c" for i in range(n)],
        "u(x)": [u(P[i]) for i in range(n)]
    }).sort_values(["u(x)", "Alternativa"], ascending=[False, True])
    st.dataframe(util_df, use_container_width=True, hide_index=True)

    fig_u = go.Figure()
    fig_u.add_trace(go.Bar(x=util_df["Alternativa"], y=util_df["u(x)"]))
    fig_u.update_layout(
        height=320, margin=dict(l=10, r=10, t=40, b=10),
        title="u(x) es ordinal: solo preserva orden (no intensidad)",
        yaxis=dict(range=[-0.1, 1.1])
    )
    st.plotly_chart(fig_u, use_container_width=True)

# ========= TAB 2: Visualizacion =========
with tab2:
    st.subheader("Grafo de dominancia (estructura P -> P^c)")
    st.caption("Todas las flechas estrictas van de P hacia P^c. Eso es exactamente la teoria de tus capturas, en una sola imagen.")

    P_idx = [i for i in range(n) if P[i]]
    C_idx = [i for i in range(n) if not P[i]]

    if not P_idx or not C_idx:
        st.warning("No hay pares P -> P^c (si todo esta en P o todo en P^c, la relacion estricta queda vacia).")
    else:
        xP = np.linspace(0.1, 0.9, len(P_idx))
        xC = np.linspace(0.1, 0.9, len(C_idx))
        pos = {}
        for k, i in enumerate(P_idx):
            pos[i] = (xP[k], 0.82)
        for k, i in enumerate(C_idx):
            pos[i] = (xC[k], 0.18)

        edges = [(i, j) for i in P_idx for j in C_idx][:max_edges]
        total = len(edges)
        step = st.slider("Flechas mostradas", 1, max(1, total), min(40, total) if total else 1, 1)

        placeholder = st.empty()

        def render(num_edges: int):
            fig = go.Figure()

            node_ids = list(pos.keys())
            fig.add_trace(go.Scatter(
                x=[pos[i][0] for i in node_ids],
                y=[pos[i][1] for i in node_ids],
                mode="markers+text",
                text=[names[i] for i in node_ids],
                textposition="top center",
                hovertext=[f"{names[i]} | {'P' if P[i] else 'P^c'} | u={u(P[i])}" for i in node_ids],
                hoverinfo="text",
                marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale="Plasma"),
                showlegend=False
            ))

            for (i, j) in edges[:num_edges]:
                fig.add_trace(go.Scatter(
                    x=[pos[i][0], pos[j][0]],
                    y=[pos[i][1], pos[j][1]],
                    mode="lines",
                    line=dict(width=2),
                    showlegend=False,
                    hoverinfo="skip"
                ))
                if show_edge_labels:
                    fig.add_trace(go.Scatter(
                        x=[(pos[i][0] + pos[j][0]) / 2],
                        y=[(pos[i][1] + pos[j][1]) / 2],
                        mode="text",
                        text=[r"$\succ$"],
                        showlegend=False,
                        hoverinfo="skip"
                    ))

            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                title=f"Relaciones estrictas mostradas: {min(num_edges,total)}/{total}"
            )
            return fig

        if autoplay and total > 0:
            for s in range(1, min(step + speed * 3, total) + 1, speed):
                placeholder.plotly_chart(render(s), use_container_width=True)
                time.sleep(max(0.02, 0.16 - 0.012 * speed))
        else:
            placeholder.plotly_chart(render(step), use_container_width=True)

    st.subheader("Matriz de la relacion (tablero logico)")
    R = build_rel(names, P)
    H = rel_num(R)

    # display matrix table (symbols with LaTeX rendering in caption)
    fig_h = go.Figure(data=go.Heatmap(z=H, x=names, y=names))
    for i in range(n):
        for j in range(n):
            # translate tokens to LaTeX for annotation
            tok = R[i, j]
            latex = {"succ": r"$\succ$", "succeq": r"$\succeq$", "preceq": r"$\preceq$", "~": r"$\sim$"}.get(tok, tok)
            fig_h.add_annotation(x=names[j], y=names[i], text=latex, showarrow=False)
    fig_h.update_layout(height=560, margin=dict(l=10, r=10, t=40, b=10), title="Bloques: dos clases de equivalencia + dominancia entre bloques")
    st.plotly_chart(fig_h, use_container_width=True)

# ========= TAB 3: Comparacion =========
with tab3:
    st.subheader("Comparacion (para cerrar como colega, no como estudiante)")

    st.markdown(r"""
### 1) Preferencia dicotomica (este modelo)
- Estructura: dos bloques ($P$ / $P^c$)
- Representacion: utilidad indicadora $u(x)\in\{0,1\}$
- Propiedad: **maxima rigidez** (no hay informacion ordinal intra-bloque)

### 2) Preferencias lexicograficas (jerarquia de criterios)
- Idea: primero se ordena por un criterio principal; empates se rompen con el siguiente, etc.
- Consecuencia: suelen ser **no representables por utilidad continua** (clasico en teoria de decision)
- Ventaja: agregan informacion sin perder consistencia, pero cambian la topologia del problema

### 3) Preferencias con umbral (threshold)
- Idea: indiferencia dentro de una banda; preferencia fuera del umbral
- Consecuencia: requieren un **parametro adicional** (el umbral) y normalmente un atributo medible
- Ventaja: capturan "no me importa" en rangos pequenos

### 4) Ranking completo (orden total refinado)
- Idea: toda alternativa tiene un puesto unico (sin clases grandes de indiferencia)
- Consecuencia: impone mas estructura; es una suposicion fuerte si no hay datos

**Cierre:** Este modelo dicotomico es un **caso limite elegante**: consistente, completo y deliberadamente poco informativo.
""")

    st.subheader("Mensaje final listo para enviar")
    final_msg = f"""Formalice tus notas como una relacion de preferencia dicotomica sobre X, con una particion P / P^c.

- Preferencia estricta: x \\succ y  \\iff  x \\in P,\\; y \\in P^c
- Indiferencia intra-bloque: x \\sim y  si ambos estan en P o ambos en P^c
- Representacion ordinal natural: u(x)=1 si x\\in P, u(x)=0 si x\\in P^c, y x \\succeq y \\iff u(x)\\ge u(y)

Lo interesante no es la definicion sino la rigidez: el preorden induce exactamente dos clases de equivalencia y no admite refinamiento ordinal sin supuestos extra (lexicografia, umbrales o atributos adicionales).

P={P_set if P_set else ['∅']} | P^c={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area("Copia/pega", final_msg, height=230)
