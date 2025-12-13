import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Preferencias dicotomicas — demo seminario', page_icon='🧠', layout='wide', initial_sidebar_state='expanded')

# Minimal styling
st.markdown('''
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
.card { border: 1px solid rgba(250,250,250,0.12); border-radius: 14px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
</style>
''' , unsafe_allow_html=True)

st.title('🧠 Preferencias dicotomicas — de nota a estructura (nivel seminario)')
st.caption('Formalizacion canonica, propiedades, visualizacion estructural y comparacion con modelos cercanos.')

def u(in_P: bool) -> int:
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def rel_token(i: int, j: int, P: List[bool]) -> str:
    if i == j:
        return 'sim'
    if strict_pref(i, j, P):
        return 'succ'
    if indifferent(i, j, P):
        return 'sim'
    return 'succeq' if weak_pref(i, j, P) else 'preceq'

def build_relation_tokens(n: int, P: List[bool]) -> np.ndarray:
    R = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            R[i, j] = rel_token(i, j, P)
    return R

def tokens_to_numeric(R: np.ndarray) -> np.ndarray:
    m = {'succ': 2.0, 'sim': 1.0, 'succeq': 1.5, 'preceq': 0.5}
    return np.vectorize(lambda s: m.get(s, 0.0))(R)

def check_completeness(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, P) or weak_pref(j, i, P)):
                return False
    return True

def check_transitivity(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and (not weak_pref(i, k, P)):
                    return False
    return True

def edges_strict(P: List[bool], max_edges: int) -> List[Tuple[int, int]]:
    n = len(P)
    out = []
    for i in range(n):
        if not P[i]:
            continue
        for j in range(n):
            if P[j]:
                continue
            out.append((i, j))
            if len(out) >= max_edges:
                return out
    return out

def bipartite_positions(P: List[bool]) -> Dict[int, Tuple[float, float]]:
    n = len(P)
    P_idx = [i for i in range(n) if P[i]]
    C_idx = [i for i in range(n) if not P[i]]
    pos: Dict[int, Tuple[float, float]] = {}
    if P_idx:
        xs = np.linspace(0.08, 0.92, len(P_idx))
        for k, i in enumerate(P_idx):
            pos[i] = (float(xs[k]), 0.80)
    if C_idx:
        xs = np.linspace(0.08, 0.92, len(C_idx))
        for k, i in enumerate(C_idx):
            pos[i] = (float(xs[k]), 0.20)
    return pos

def latex_for_token(tok: str) -> str:
    return {'succ': r'$\\succ$', 'succeq': r'$\\succeq$', 'preceq': r'$\\preceq$', 'sim': r'$\\sim$'}.get(tok, tok)

with st.sidebar:
    st.header('⚙️ Configuracion')
    n = st.slider('Cardinalidad de X', 4, 24, 10, 1)
    default_names = ', '.join([f'x{i+1}' for i in range(n)])
    names_str = st.text_input('Etiquetas (coma)', value=default_names)
    names = [s.strip() for s in names_str.split(',') if s.strip()]
    if len(names) != n:
        st.warning('Etiquetas invalidas. Uso x1,...,xn.')
        names = [f'x{i+1}' for i in range(n)]

    st.divider()
    st.subheader('Particion P / P^c')
    mode = st.radio('Modo', ['Manual', 'Aleatorio (demo)'], index=0)
    if mode.startswith('Aleatorio'):
        p_share = st.slider('Proporcion en P', 0.10, 0.90, 0.50, 0.05)
        seed = st.number_input('Semilla', value=11, step=1)
        rng = np.random.default_rng(int(seed))
        P = list((rng.random(n) < p_share).astype(bool))
    else:
        P = []
        for i, nm in enumerate(names):
            P.append(st.checkbox(f'{nm} in P', value=(i % 2 == 0)))

    st.divider()
    st.subheader('Visualizacion')
    max_edges = st.slider('Max flechas (P->P^c)', 20, 800, 220, 10)
    show_edge_labels = st.checkbox('Etiqueta succ en flechas', value=False)

    st.divider()
    st.subheader('Animacion')
    play_steps = st.slider('Pasos a reproducir', 5, 120, 45, 5)
    play_delay = st.slider('Delay (ms)', 20, 250, 90, 10)
    play = st.button('▶️ Play (una vez)', use_container_width=True)

P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1,k2,k3,k4 = st.columns(4)
k1.metric('|P|', len(P_set))
k2.metric('|P^c|', len(Pc_set))
k3.metric('Completitud', 'OK' if check_completeness(P) else 'NO')
k4.metric('Transitividad', 'OK' if check_transitivity(P) else 'NO')

tab1, tab2, tab3 = st.tabs(['Modelo', 'Visualizacion', 'Comparacion'])

with tab1:
    left, right = st.columns([1.1, 0.9], gap='large')
    with left:
        st.markdown(r"""
### Formalizacion canonica
Sea $X$ el conjunto de alternativas y $P \subset X$ un subconjunto distinguido, con $P^c = X \setminus P$.

- Preferencia estricta:  $x \succ y \iff (x \in P)\land(y \in P^c)$  
- Indiferencia intra-bloque:  $x \sim y$ si ambos estan en $P$ o ambos en $P^c$  
- Representacion ordinal natural (indicadora):
$$
u(x)=\begin{cases}
1 & x \in P \\
0 & x \in P^c
\end{cases}
\qquad\Rightarrow\qquad
x \succeq y \iff u(x)\ge u(y)
$$
""" )
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('**Particion actual**')
        st.write('P =', P_set if P_set else ['∅'])
        st.write('P^c =', Pc_set if Pc_set else ['∅'])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(r"""
### Lectura estructural
- Este preorden induce exactamente **dos clases de equivalencia** ($P$ y $P^c$).
- Dentro de cada clase no hay informacion ordinal adicional.
- Cualquier refinamiento requiere supuestos extra (atributos, umbrales, lexicografia, etc.).
""" )
    with right:
        st.markdown('### Utilidad indicadora (u ∈ {0,1})')
        util_df = pd.DataFrame({'Alternativa': names, 'Grupo': ['P' if P[i] else 'P^c' for i in range(n)], 'u(x)': [u(P[i]) for i in range(n)]})
        util_df = util_df.sort_values(['u(x)', 'Alternativa'], ascending=[False, True])
        st.dataframe(util_df, use_container_width=True, hide_index=True)
        fig_u = go.Figure()
        fig_u.add_trace(go.Bar(x=util_df['Alternativa'], y=util_df['u(x)']))
        fig_u.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), title='u(x) preserva el orden (ordinal, no cardinal)', yaxis=dict(range=[-0.1, 1.1]))
        st.plotly_chart(fig_u, use_container_width=True)

with tab2:
    colA, colB = st.columns([1.05, 1.0], gap='large')
    with colA:
        st.markdown('### Grafo de dominancia (P -> P^c)')
        st.caption('Todas las relaciones estrictas van de P hacia P^c. No hay gradacion intra-bloque.')
        P_idx = [i for i in range(n) if P[i]]
        C_idx = [i for i in range(n) if not P[i]]
        if (not P_idx) or (not C_idx):
            st.warning('No hay pares P->P^c (si todo esta en P o todo en P^c, la relacion estricta es vacia).')
        else:
            pos = bipartite_positions(P)
            edges = edges_strict(P, max_edges=max_edges)
            total = len(edges)
            if 'anim_step' not in st.session_state:
                st.session_state.anim_step = min(40, total) if total else 1
            step = st.slider('Flechas mostradas', 1, max(1,total), int(min(st.session_state.anim_step, max(1,total))), 1)
            ph = st.empty()
            def render_graph(m: int):
                fig = go.Figure()
                node_ids = list(pos.keys())
                fig.add_trace(go.Scatter(x=[pos[i][0] for i in node_ids], y=[pos[i][1] for i in node_ids], mode='markers+text', text=[names[i] for i in node_ids], textposition='top center', hovertext=[f"{names[i]} | {'P' if P[i] else 'P^c'} | u={u(P[i])}" for i in node_ids], hoverinfo='text', marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale='Plasma', line=dict(width=1)), showlegend=False))
                for (i,j) in edges[:m]:
                    fig.add_trace(go.Scatter(x=[pos[i][0],pos[j][0]], y=[pos[i][1],pos[j][1]], mode='lines', line=dict(width=2), showlegend=False, hoverinfo='skip'))
                    if show_edge_labels:
                        fig.add_trace(go.Scatter(x=[(pos[i][0]+pos[j][0])/2.0], y=[(pos[i][1]+pos[j][1])/2.0], mode='text', text=[r'$\\succ$'], showlegend=False, hoverinfo='skip'))
                fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), xaxis=dict(visible=False), yaxis=dict(visible=False), title=f'Relaciones estrictas mostradas: {min(m,total)}/{total}')
                return fig
            if play:
                start = max(1, int(step))
                end = min(total, start + int(play_steps))
                for s in range(start, end+1, 2):
                    ph.plotly_chart(render_graph(s), use_container_width=True)
                    time.sleep(max(0.02, float(play_delay)/1000.0))
                st.session_state.anim_step = end
            else:
                ph.plotly_chart(render_graph(step), use_container_width=True)
    with colB:
        st.markdown('### Matriz de la relacion')
        st.caption('Dos bloques de indiferencia + dominancia total entre bloques.')
        R = build_relation_tokens(n, P)
        H = tokens_to_numeric(R)
        fig_h = go.Figure(data=go.Heatmap(z=H, x=names, y=names))
        for i in range(n):
            for j in range(n):
                fig_h.add_annotation(x=names[j], y=names[i], text=latex_for_token(R[i,j]), showarrow=False, font=dict(size=12))
        fig_h.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), title='Relacion: (succ, succeq, preceq, sim)')
        st.plotly_chart(fig_h, use_container_width=True)

with tab3:
    st.markdown(r"""
### Comparacion conceptual
**1) Preferencia dicotomica (este modelo)**  
- Dos bloques ($P$ / $P^c$)  
- Utilidad indicadora $u(x)\in\{0,1\}$  
- Maxima rigidez: sin informacion ordinal intra-bloque

**2) Lexicografica**  
- Jerarquia de criterios; empates se rompen por criterios secundarios  
- Tipicamente no representable por utilidad continua

**3) Umbral (threshold)**  
- Indiferencia en una banda; preferencia fuera del umbral  
- Requiere un parametro adicional (el umbral) y un atributo

**4) Orden total refinado**  
- Ranking completo (sin grandes clases de indiferencia)

**Cierre:** el modelo dicotomico es un caso limite elegante: consistente, completo y deliberadamente poco informativo.
""" )
    st.markdown('### Mensaje final listo para enviar')
    final_msg = f"""Formalice tus notas como una relacion de preferencia dicotomica sobre X, con una particion P / P^c.

- Preferencia estricta: x \\succ y  \\iff  x \\in P,\\; y \\in P^c
- Indiferencia intra-bloque: x \\sim y si ambos estan en P o ambos en P^c
- Representacion ordinal natural: u(x)=1 si x\\in P, u(x)=0 si x\\in P^c, y x \\succeq y \\iff u(x)\\ge u(y)

Lo interesante no es la definicion sino la rigidez: el preorden induce exactamente dos clases de equivalencia y no admite refinamiento ordinal sin supuestos extra.

P={P_set if P_set else ['∅']} | P^c={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area('Copia/pega', final_msg, height=220)
