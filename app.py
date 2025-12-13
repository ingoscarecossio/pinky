import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='Preferencias dicotómicas — demo limpia', page_icon='🧠', layout='wide')

st.markdown('''
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
.card { border: 1px solid rgba(250,250,250,0.12); border-radius: 14px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
.muted { color: rgba(250,250,250,0.72); font-size: 0.92rem; }
</style>
''' , unsafe_allow_html=True)

st.title('🧠 Preferencias dicotómicas (P / Pᶜ) — demo limpia para economista experta')
st.caption('Estructura clara > decoración. Cero ruido visual: solo estructura.')

def u(in_P: bool) -> int:
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def rel_symbol(i: int, j: int, P: List[bool]) -> str:
    if i == j: return '∼'
    if strict_pref(i, j, P): return '≻'
    if indifferent(i, j, P): return '∼'
    return '⪰' if weak_pref(i, j, P) else '⪯'

def rel_numeric(sym: str) -> float:
    return {'≻':2.0,'⪰':1.5,'∼':1.0,'⪯':0.5}.get(sym, 0.0)

def build_relation(names: List[str], P: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(names)
    S = np.empty((n,n), dtype=object)
    Z = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = rel_symbol(i,j,P)
            S[i,j] = s
            Z[i,j] = rel_numeric(s)
    return S, Z

def check_completeness(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i,j,P) or weak_pref(j,i,P)): return False
    return True

def check_transitivity(P: List[bool]) -> bool:
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i,j,P) and weak_pref(j,k,P) and (not weak_pref(i,k,P)): return False
    return True

def strict_edges(P: List[bool], max_edges: int) -> List[Tuple[int,int]]:
    n = len(P)
    out = []
    for i in range(n):
        if not P[i]: continue
        for j in range(n):
            if P[j]: continue
            out.append((i,j))
            if len(out) >= max_edges: return out
    return out

def bipartite_positions(P: List[bool]) -> Dict[int, Tuple[float,float]]:
    n = len(P)
    top = [i for i in range(n) if P[i]]
    bot = [i for i in range(n) if not P[i]]
    pos = {}
    if top:
        xs = np.linspace(0.08, 0.92, len(top))
        for k,i in enumerate(top): pos[i] = (float(xs[k]), 0.78)
    if bot:
        xs = np.linspace(0.08, 0.92, len(bot))
        for k,i in enumerate(bot): pos[i] = (float(xs[k]), 0.22)
    return pos

with st.sidebar:
    st.header('⚙️ Controles')
    n = st.slider('Tamaño de X', 4, 24, 10, 1)
    default_names = ', '.join([f'x{i+1}' for i in range(n)])
    names_str = st.text_input('Nombres (coma)', value=default_names)
    names = [s.strip() for s in names_str.split(',') if s.strip()]
    if len(names) != n:
        st.warning('Nombres inválidos. Uso x1,…,xn.')
        names = [f'x{i+1}' for i in range(n)]

    st.divider()
    mode = st.radio('Definir P', ['Manual', 'Aleatorio (demo)'], index=0)
    if mode.startswith('Aleatorio'):
        p_share = st.slider('Proporción en P', 0.10, 0.90, 0.50, 0.05)
        seed = st.number_input('Semilla', value=11, step=1)
        rng = np.random.default_rng(int(seed))
        P = list((rng.random(n) < p_share).astype(bool))
    else:
        P = [st.checkbox(f"{names[i]} ∈ P", value=(i%2==0)) for i in range(n)]

    st.divider()
    st.subheader('Grafo')
    max_edges = st.slider('Máx flechas P→Pᶜ', 20, 1200, 240, 20)
    show_animation = st.checkbox('Animar (una vez)', value=False)
    play_delay = st.slider('Delay (ms)', 20, 250, 80, 10)

    st.divider()
    st.subheader('Matriz')
    show_symbol_table = st.checkbox('Mostrar tabla de símbolos', value=True)

P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1,k2,k3,k4 = st.columns(4)
k1.metric('|P|', len(P_set))
k2.metric('|Pᶜ|', len(Pc_set))
k3.metric('Completitud', 'OK' if check_completeness(P) else 'NO')
k4.metric('Transitividad', 'OK' if check_transitivity(P) else 'NO')

tab1, tab2, tab3 = st.tabs(['Modelo', 'Visualización', 'Comparación'])

with tab1:
    left, right = st.columns([1.1, 0.9], gap='large')
    with left:
        st.markdown(r'''
### Formalización
Sea \(X\) el conjunto de alternativas y \(P \subset X\) un subconjunto distinguido; \(P^c = X \setminus P\).

- \(x \succ y \iff (x\in P)\land (y\in P^c)\)
- \(x \sim y\) si ambos están en \(P\) o ambos están en \(P^c\)

- Representación ordinal natural (indicadora):
\[
  u(x)=\begin{cases}
    1 & x\in P \\
    0 & x\in P^c
  \end{cases}
  \quad\Rightarrow\quad x\succeq y \iff u(x)\ge u(y)
\]
        ''' )
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write('P =', P_set if P_set else ['∅'])
        st.write('Pᶜ =', Pc_set if Pc_set else ['∅'])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(r'''
### Lectura estructural
- Dos clases de equivalencia: \(P\) y \(P^c\).
- Intra-bloque no hay información ordinal adicional.
- Refinar requiere supuestos extra (atributos, umbrales, lexicografía…).
        ''' )
    with right:
        st.markdown('### Utilidad indicadora')
        util_df = pd.DataFrame({'Alternativa': names, 'Grupo': ['P' if P[i] else 'Pᶜ' for i in range(n)], 'u(x)': [u(P[i]) for i in range(n)]})
        util_df = util_df.sort_values(['u(x)', 'Alternativa'], ascending=[False, True])
        st.dataframe(util_df, use_container_width=True, hide_index=True)

with tab2:
    colA, colB = st.columns([1.0, 1.05], gap='large')
    with colA:
        st.subheader('Grafo bipartito de dominancia estricta (P → Pᶜ)')
        st.markdown('<div class="muted">Sin etiquetas en flechas. El mensaje es la estructura.</div>', unsafe_allow_html=True)
        edges = strict_edges(P, max_edges=max_edges)
        pos = bipartite_positions(P)
        if not edges:
            st.warning('No hay relaciones estrictas (todo en P o todo en Pᶜ).')
        else:
            total = len(edges)
            step = st.slider('Flechas mostradas', 1, total, min(40,total), 1)
            def render_graph(m: int):
                fig = go.Figure()
                node_ids = list(pos.keys())
                fig.add_trace(go.Scatter(x=[pos[i][0] for i in node_ids], y=[pos[i][1] for i in node_ids], mode='markers+text', text=[names[i] for i in node_ids], textposition='top center', hovertext=[f"{names[i]} | {'P' if P[i] else 'Pᶜ'} | u={u(P[i])}" for i in node_ids], hoverinfo='text', marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale='Plasma'), showlegend=False))
                xs, ys = [], []
                for (i,j) in edges[:m]:
                    xs += [pos[i][0], pos[j][0], None]
                    ys += [pos[i][1], pos[j][1], None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(width=2), hoverinfo='skip', showlegend=False))
                fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), xaxis=dict(visible=False), yaxis=dict(visible=False), title=f'Relaciones estrictas mostradas: {m}/{total}')
                return fig
            ph = st.empty()
            if show_animation:
                for s in range(1, min(step,total)+1, 2):
                    ph.plotly_chart(render_graph(s), use_container_width=True)
                    time.sleep(max(0.02, float(play_delay)/1000.0))
            ph.plotly_chart(render_graph(step), use_container_width=True)
    with colB:
        st.subheader('Matriz (color=estructura; hover=símbolo)')
        st.markdown('<div class="muted">El símbolo vive en hover. Así se ve profesional.</div>', unsafe_allow_html=True)
        S, Z = build_relation(names, P)
        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(data=go.Heatmap(z=Z, x=names, y=names, text=hover, hoverinfo='text'))
        fig_h.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), title='Heatmap de la relación (hover para símbolo)')
        st.plotly_chart(fig_h, use_container_width=True)
        if show_symbol_table:
            st.markdown('#### Tabla de símbolos (limpia)')
            st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

with tab3:
    st.subheader('Comparación (para cerrar como colega)')
    st.markdown(r'''
**1) Dicotómica (este modelo)**  
Dos bloques \(P/P^c\). Utilidad indicadora. Máxima rigidez.

**2) Lexicográfica**  
Jerarquía de criterios; típicamente no representable por utilidad continua.

**3) Umbral (threshold)**  
Indiferencia dentro de banda; requiere parámetro adicional.

**4) Orden total refinado**  
Ranking completo; suposición fuerte si no hay datos que lo sostengan.
    ''' )
    st.markdown('### Mensaje listo para enviar')
    msg = f"""Formalizé tus notas como una preferencia dicotómica sobre X con partición P/Pᶜ.

- x ≻ y  ⇔  x∈P, y∈Pᶜ
- x ∼ y dentro de cada bloque
- u(x)∈{0,1} representa el orden: x ⪰ y ⇔ u(x) ≥ u(y)

Lo clave es la rigidez: induce exactamente dos clases de equivalencia y no admite refinamiento ordinal sin supuestos extra.

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area('Copia/pega', msg, height=200)
