import io
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt

# ============================================================
# Pinky WOW v6 — "Modo Presentación" + PDF export (bonito y serio)
# ============================================================

st.set_page_config(page_title="Preferencias dicotómicas — demo premium", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
    .card { border: 1px solid rgba(250,250,250,0.12); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
    .muted { color: rgba(250,250,250,0.72); font-size: 0.95rem; }
    .big { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Preferencias dicotómicas (P / Pᶜ) — demo premium")
st.caption("Para descrestAR: teoría limpia, visuales sobrios, y salida en PDF lista para enviar.")

# ---------------- Model ----------------
def u(in_P: bool) -> int:
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    return (P[i] and P[j]) or ((not P[i]) and (not P[j]))

def rel_symbol(i: int, j: int, P: List[bool]) -> str:
    if i == j:
        return "∼"
    if strict_pref(i, j, P):
        return "≻"
    if indifferent(i, j, P):
        return "∼"
    return "⪰" if weak_pref(i, j, P) else "⪯"

def rel_numeric(sym: str) -> float:
    return {"≻": 2.0, "⪰": 1.5, "∼": 1.0, "⪯": 0.5}.get(sym, 0.0)

def build_relation(names: List[str], P: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(names)
    S = np.empty((n, n), dtype=object)
    Z = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = rel_symbol(i, j, P)
            S[i, j] = s
            Z[i, j] = rel_numeric(s)
    return S, Z

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

def strict_edges(P: List[bool], max_edges: int) -> List[Tuple[int, int]]:
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

# ---------------- Matplotlib figures for PDF ----------------
def fig_bipartite_matplotlib(names: List[str], P: List[bool], max_edges: int = 200):
    edges = strict_edges(P, max_edges=max_edges)
    pos = bipartite_positions(P)
    n = len(P)

    fig = plt.figure(figsize=(7.2, 3.6), dpi=170)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=140)

    for i in range(n):
        ax.text(pos[i][0], pos[i][1] + 0.04, names[i], ha="center", va="bottom", fontsize=8)

    for (i, j) in edges:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], linewidth=1.2)

    ax.set_title("Grafo bipartito de dominancia estricta (P → Pᶜ)", fontsize=11)
    fig.tight_layout()
    return fig

def fig_heatmap_matplotlib(names: List[str], Z: np.ndarray):
    fig = plt.figure(figsize=(7.2, 4.0), dpi=170)
    ax = fig.add_subplot(111)
    im = ax.imshow(Z)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Heatmap de la relación (valores ordinales)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def mpl_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# ---------------- PDF generation ----------------
def build_pdf_bytes(names: List[str], P: List[bool], Z: np.ndarray) -> bytes:
    P_set = [names[i] for i in range(len(P)) if P[i]]
    Pc_set = [names[i] for i in range(len(P)) if not P[i]]

    img1 = mpl_to_png_bytes(fig_bipartite_matplotlib(names, P, max_edges=220))
    img2 = mpl_to_png_bytes(fig_heatmap_matplotlib(names, Z))

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    def header(title: str, subtitle: str):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(0.8*inch, h-0.9*inch, title)
        c.setFont("Helvetica", 10)
        c.drawString(0.8*inch, h-1.15*inch, subtitle)
        c.line(0.8*inch, h-1.25*inch, w-0.8*inch, h-1.25*inch)

    header("Preferencias dicotómicas (P / Pᶜ)", "Nota técnica + visualización estructural")

    y = h - 1.6*inch
    c.setFont("Helvetica", 10)
    text = c.beginText(0.8*inch, y)
    text.setLeading(14)
    text.textLine("Formalización:")
    text.textLine("  x ≻ y  ⇔  (x ∈ P) ∧ (y ∈ Pᶜ)")
    text.textLine("  x ∼ y  dentro de cada bloque (P o Pᶜ)")
    text.textLine("  u(x)=1 si x∈P y u(x)=0 si x∈Pᶜ, y x ⪰ y ⇔ u(x) ≥ u(y)")
    text.textLine("")
    text.textLine(f"P = {P_set if P_set else ['∅']}")
    text.textLine(f"Pᶜ = {Pc_set if Pc_set else ['∅']}")
    text.textLine(f"Axiomas: completitud={'OK' if check_completeness(P) else 'NO'}, transitividad={'OK' if check_transitivity(P) else 'NO'}")
    text.textLine("")
    text.textLine("Lectura: el preorden induce exactamente dos clases de equivalencia y es deliberadamente rígido.")
    c.drawText(text)

    c.drawImage(ImageReader(io.BytesIO(img1)), 0.8*inch, 3.9*inch, width=w-1.6*inch, height=2.3*inch, preserveAspectRatio=True, anchor="n")
    c.drawImage(ImageReader(io.BytesIO(img2)), 0.8*inch, 1.1*inch, width=w-1.6*inch, height=2.6*inch, preserveAspectRatio=True, anchor="n")

    c.showPage()

    header("Cierre y comparación", "Para conversación con economista experta")
    y2 = h - 1.6*inch
    t2 = c.beginText(0.8*inch, y2)
    t2.setLeading(14)
    t2.setFont("Helvetica", 10)
    t2.textLine("Comparación conceptual:")
    t2.textLine("  • Dicotómica: dos bloques, utilidad indicadora, máxima rigidez.")
    t2.textLine("  • Lexicográfica: jerarquía de criterios; agrega estructura sin promediar.")
    t2.textLine("  • Umbral: indiferencia en banda; requiere parámetro extra + atributo.")
    t2.textLine("  • Orden total: ranking completo; suposición fuerte sin datos.")
    t2.textLine("")
    t2.textLine("Mensaje listo:")
    t2.textLine("  “Formalizé tu construcción como una preferencia dicotómica P/Pᶜ.")
    t2.textLine("   Lo interesante es la rigidez: dos clases de equivalencia y nada intra-bloque;")
    t2.textLine("   cualquier refinamiento exige supuestos extra (lexicografía, umbrales o atributos).”")
    c.drawText(t2)

    c.save()
    return buf.getvalue()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controles")
    n = st.slider("Tamaño de X", 4, 24, 10, 1)

    default_names = ", ".join([f"x{i+1}" for i in range(n)])
    names_str = st.text_input("Nombres (coma)", value=default_names)
    names = [s.strip() for s in names_str.split(",") if s.strip()]
    if len(names) != n:
        st.warning("Nombres inválidos. Uso x1,…,xn.")
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
    st.subheader("Visual")
    max_edges = st.slider("Máx flechas P→Pᶜ", 20, 1200, 240, 20)
    animate_once = st.checkbox("Animar (una vez)", value=False)
    delay_ms = st.slider("Delay (ms)", 20, 250, 70, 10)

    st.divider()
    st.subheader("Export")
    export_pdf = st.button("📄 Generar PDF (nota técnica)", use_container_width=True)

P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]
S, Z = build_relation(names, P)

k1, k2, k3, k4 = st.columns(4)
k1.metric("|P|", len(P_set))
k2.metric("|Pᶜ|", len(Pc_set))
k3.metric("Completitud", "OK" if check_completeness(P) else "NO")
k4.metric("Transitividad", "OK" if check_transitivity(P) else "NO")

# ---------------- Presentation mode ----------------
mode_col1, mode_col2 = st.columns([0.62, 0.38])
with mode_col1:
    st.subheader("🎤 Modo Presentación")
    st.markdown('<div class="muted">Te guía en 4 pasos y suelta el remate intelectual al final.</div>', unsafe_allow_html=True)
with mode_col2:
    if "slide" not in st.session_state:
        st.session_state.slide = 1
    prev, nxt = st.columns(2)
    if prev.button("⬅️ Atrás", use_container_width=True):
        st.session_state.slide = max(1, st.session_state.slide - 1)
    if nxt.button("Siguiente ➡️", use_container_width=True):
        st.session_state.slide = min(4, st.session_state.slide + 1)

slide = st.session_state.slide

tab_pres, tab_viz, tab_comp = st.tabs(["Presentación", "Visualización", "Comparación"])

with tab_pres:
    if slide == 1:
        st.markdown('<div class="card big">', unsafe_allow_html=True)
        st.markdown("### Paso 1 — Definición (sin romanticismo)")
        st.markdown(r"""
La captura define una **preferencia dicotómica**: separa \(X\) en dos bloques \(P\) y \(P^c\).  
La preferencia estricta es:
\[
x \succ y \iff (x\in P)\land(y\in P^c).
\]
""")
        st.markdown("</div>", unsafe_allow_html=True)
    elif slide == 2:
        st.markdown('<div class="card big">', unsafe_allow_html=True)
        st.markdown("### Paso 2 — Consecuencia: dos clases de equivalencia")
        st.markdown(r"""
Dentro de cada bloque no hay ranking:
- Si \(x,y\in P\) entonces \(x\sim y\)
- Si \(x,y\in P^c\) entonces \(x\sim y\)

Lo único informativo es el salto \(P 	o P^c\).
""")
        st.markdown("</div>", unsafe_allow_html=True)
    elif slide == 3:
        st.markdown('<div class="card big">', unsafe_allow_html=True)
        st.markdown("### Paso 3 — Representación por utilidad indicadora")
        st.markdown(r"""
Hay una representación ordinal natural:
\[
u(x)=\mathbb{1}\{x\in P\}\in\{0,1\},\qquad x\succeq y \iff u(x)\ge u(y).
\]
Esto no mide intensidad. Solo orden.
""")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card big">', unsafe_allow_html=True)
        st.markdown("### Paso 4 — El remate (lo que suena a colega)")
        st.markdown(r"""
Lo elegante es la **rigidez**: el modelo colapsa en dos clases y **no admite refinamiento** sin supuestos extra.
Si quieres más información, debes introducir estructura: lexicografía, umbrales, atributos, etc.
""")
        st.markdown("**Cierre listo para decir/enviar:**")
        st.code(
            "Formalizé tu construcción como una preferencia dicotómica P/Pᶜ.
"
            "Lo interesante es la rigidez: dos clases de equivalencia y nada intra-bloque;
"
            "cualquier refinamiento exige supuestos extra (lexicografía, umbrales o atributos)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if export_pdf:
        pdf_bytes = build_pdf_bytes(names, P, Z)
        st.download_button(
            "⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name="preferencias_dicotomicas_nota_tecnica.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with tab_viz:
    colA, colB = st.columns([1.0, 1.05], gap="large")
    with colA:
        st.subheader("Grafo bipartito P → Pᶜ (limpio)")
        edges = strict_edges(P, max_edges=max_edges)
        pos = bipartite_positions(P)
        if not edges:
            st.warning("No hay relaciones estrictas (todo en P o todo en Pᶜ).")
        else:
            total = len(edges)
            step = st.slider("Flechas mostradas", 1, total, min(40, total), 1)

            def render_graph(m: int) -> go.Figure:
                fig = go.Figure()
                node_ids = list(pos.keys())
                fig.add_trace(go.Scatter(
                    x=[pos[i][0] for i in node_ids],
                    y=[pos[i][1] for i in node_ids],
                    mode="markers+text",
                    text=[names[i] for i in node_ids],
                    textposition="top center",
                    hovertext=[f"{names[i]} | {'P' if P[i] else 'Pᶜ'} | u={u(P[i])}" for i in node_ids],
                    hoverinfo="text",
                    marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale="Plasma"),
                    showlegend=False
                ))
                xs, ys = [], []
                for (i, j) in edges[:m]:
                    xs += [pos[i][0], pos[j][0], None]
                    ys += [pos[i][1], pos[j][1], None]
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                ))
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                                  xaxis=dict(visible=False), yaxis=dict(visible=False),
                                  title=f"Relaciones estrictas mostradas: {m}/{total}")
                return fig

            ph = st.empty()
            if animate_once:
                for s in range(1, min(step, total) + 1, 2):
                    ph.plotly_chart(render_graph(s), use_container_width=True)
                    time.sleep(max(0.02, float(delay_ms) / 1000.0))
            ph.plotly_chart(render_graph(step), use_container_width=True)

    with colB:
        st.subheader("Heatmap (color=estructura; hover=símbolo)")
        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(data=go.Heatmap(z=Z, x=names, y=names, text=hover, hoverinfo="text"))
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                            title="Heatmap de la relación (hover para símbolo)")
        st.plotly_chart(fig_h, use_container_width=True)

        st.markdown("#### Tabla de símbolos (limpia)")
        st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

with tab_comp:
    st.subheader("Comparación (para conversación con economista experta)")
    st.markdown(r"""
**Dicotómica (este modelo)**: dos bloques, utilidad indicadora, máxima rigidez.

**Lexicográfica**: jerarquía de criterios; agrega estructura sin “promediar” criterios.

**Umbral**: indiferencia en banda; requiere parámetro extra + atributo medible.

**Orden total refinado**: ranking completo; suposición fuerte si no hay datos.
""")
    st.markdown("### Mensaje listo para enviar")
    msg = f"""Formalizé tus notas como una preferencia dicotómica sobre X con partición P/Pᶜ.

- x ≻ y  ⇔  x∈P, y∈Pᶜ
- x ∼ y dentro de cada bloque
- u(x)∈{{0,1}} representa el orden: x ⪰ y ⇔ u(x) ≥ u(y)

Lo clave es la rigidez: induce exactamente dos clases de equivalencia y no admite refinamiento ordinal sin supuestos extra (lexicografía, umbrales o atributos).

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area("Copia/pega", msg, height=210)

if export_pdf:
    pdf_bytes = build_pdf_bytes(names, P, Z)
    st.download_button(
        "⬇️ Descargar PDF (nota técnica)",
        data=pdf_bytes,
        file_name="preferencias_dicotomicas_nota_tecnica.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
