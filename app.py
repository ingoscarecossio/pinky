from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Preferencias dicotómicas P / Pᶜ — versión técnica + regalo
# (sin matplotlib, sin reportlab; 100% Streamlit Cloud friendly)
# ============================================================

st.set_page_config(page_title="Preferencias dicotómicas — Leidy edition", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .card { border: 1px solid rgba(250,250,250,0.12); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
      .muted { color: rgba(250,250,250,0.72); font-size: 0.95rem; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .h { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Model -------------------------------

def u(in_P: bool) -> int:
    """Indicadora: u(x)=1 si x∈P; 0 si x∈P^c."""
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    """x ⪰ y  ⇔  (x∈P) ∨ (y∈P^c)  ⇔ u(x) ≥ u(y)."""
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    """x ≻ y ⇔ (x∈P) ∧ (y∈P^c)."""
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    """x ~ y si ambos están en el mismo bloque."""
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
    # Solo para colorear (no es cardinal).
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

def check_antisymmetry(P: List[bool]) -> bool:
    # En general NO se cumple: dentro de P, i⪰j y j⪰i pero i≠j.
    # Esto confirma que es preorden, no orden parcial.
    n = len(P)
    for i in range(n):
        for j in range(n):
            if i != j and weak_pref(i, j, P) and weak_pref(j, i, P):
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

# ------------------------------- Econ layer -------------------------------

@dataclass
class ScreeningRule:
    """Regla de aceptación tipo economía pública / evaluación de proyectos."""
    theta_min: float  # umbral
    noise: float      # "medición imperfecta"

def screening_classification(theta: np.ndarray, rule: ScreeningRule, rng: np.random.Generator) -> np.ndarray:
    # Medición con ruido: \hat{theta} = theta + eps
    eps = rng.normal(0.0, rule.noise, size=theta.shape)
    theta_hat = theta + eps
    return theta_hat >= rule.theta_min

def confusion_counts(true_P: np.ndarray, pred_P: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum(true_P & pred_P))
    tn = int(np.sum((~true_P) & (~pred_P)))
    fp = int(np.sum((~true_P) & pred_P))
    fn = int(np.sum(true_P & (~pred_P)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

# ------------------------------- Header -------------------------------

st.title("🧠 Preferencias dicotómicas (P / Pᶜ) — Leidy edition")
st.caption("Un regalo técnico: teoría de preferencias + lectura económica (screening / aceptación binaria).")

with st.sidebar:
    st.header("⚙️ Controles")

    leidy_name = st.text_input("Nombre", value="Leidy")
    n = st.slider("Tamaño de X", 4, 24, 10, 1)

    default_names = ", ".join([f"x{i+1}" for i in range(n)])
    names_str = st.text_input("Etiquetas (coma)", value=default_names)
    names = [s.strip() for s in names_str.split(",") if s.strip()]
    if len(names) != n:
        st.warning("Etiquetas inválidas. Uso x1,…,xn.")
        names = [f"x{i+1}" for i in range(n)]

    st.divider()
    mode = st.radio("Definir P", ["Manual (Pinky)", "Econ: Screening (umbral)"], index=0)

    rng_seed = st.number_input("Semilla (reproducible)", value=11, step=1)
    rng = np.random.default_rng(int(rng_seed))

    if mode.startswith("Manual"):
        P = [st.checkbox(f"{names[i]} ∈ P", value=(i % 2 == 0)) for i in range(n)]
        theta = None
        rule = None
        P_true = None
        P_hat = None
    else:
        st.subheader("📈 Screening (economía)")
        st.caption("Interpreta P como ‘aprobado’ según un criterio θ con medición imperfecta.")
        theta_min = st.slider("Umbral θ_min", -2.0, 2.0, 0.0, 0.05)
        noise = st.slider("Ruido de medición (σ)", 0.0, 1.0, 0.25, 0.05)
        rule = ScreeningRule(theta_min=theta_min, noise=noise)

        # “Calidad/retorno” latente de cada alternativa (ciencia: variable latente)
        theta = rng.normal(0.0, 1.0, size=n)
        # “Verdad”: aprobado si theta >= theta_min sin ruido
        P_true = theta >= theta_min
        # Observado: con ruido
        P_hat = screening_classification(theta, rule, rng)

        # Para el modelo de preferencia usamos P_hat (decisión observada / institucional)
        P = list(P_hat.astype(bool))

    st.divider()
    st.subheader("Visual")
    show_symbol_table = st.checkbox("Mostrar tabla de símbolos", value=True)

# ------------------------------- Derived -------------------------------

S, Z = build_relation(names, P)
P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("|P|", len(P_set))
k2.metric("|Pᶜ|", len(Pc_set))
k3.metric("Completitud", "OK" if check_completeness(P) else "NO")
k4.metric("Transitividad", "OK" if check_transitivity(P) else "NO")

tab1, tab2, tab3, tab4 = st.tabs(["Demostración", "Estructura", "Economía", "Mensaje"])

# ------------------------------- Demostración -------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {leidy_name}, esto es lo que tu demo está diciendo (sin adornos)")
    st.markdown(
        """
**1) Partición del conjunto de alternativas**  
Sea \(X\) el conjunto de alternativas. Se define un subconjunto \(P \subset X\) y su complemento \(P^c = X\\setminus P\).

**2) Preferencia estricta**  
\[
x \succ y \iff (x\in P)\land(y\in P^c).
\]

**3) Indiferencia**  
\[
x\sim y \iff (x,y\in P)\ \ \text{o}\ \ (x,y\in P^c).
\]

**4) Preferencia débil y representación por utilidad indicadora**  
Definiendo
\[
u(x)=\begin{cases}
1 & x\in P \\\\
0 & x\in P^c
\end{cases}
\]
se tiene
\[
x\succeq y \iff u(x)\ge u(y).
\]
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Lo que se concluye (propiedad estructural)")
    st.markdown(
        """
- Esto **no** produce un ranking fino: colapsa \(X\) en **dos clases de equivalencia**: \(P\) y \(P^c\).  
- Es **completo y transitivo** ⇒ **preorden** (no necesariamente antisymétrico).  
- La “falta” de antisimetría no es un bug: es exactamente la idea de indiferencia dentro de bloques.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Estructura -------------------------------
with tab2:
    colA, colB = st.columns([1.0, 1.05], gap="large")

    with colA:
        st.subheader("Grafo bipartito de dominancia estricta (P → Pᶜ)")
        st.markdown('<div class="muted">Cada flecha representa “aprobado” ≻ “no aprobado”. Dentro de bloques hay indiferencia.</div>', unsafe_allow_html=True)

        pos = bipartite_positions(P)
        top = [i for i in range(n) if P[i]]
        bot = [i for i in range(n) if not P[i]]

        if not top or not bot:
            st.warning("No hay relaciones estrictas (todo quedó en P o todo en Pᶜ).")
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

            # edges as one trace (clean)
            xs, ys = [], []
            for i in top:
                for j in bot:
                    xs += [pos[i][0], pos[j][0], None]
                    ys += [pos[i][1], pos[j][1], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=2), hoverinfo="skip", showlegend=False))

            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                title="Dominancia estricta inducida por la partición P/Pᶜ",
            )
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.subheader("Matriz de la relación (color = estructura; hover = símbolo)")
        st.markdown('<div class="muted">Nada de texto encima: el símbolo aparece en hover (legible y serio).</div>', unsafe_allow_html=True)

        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(go.Heatmap(z=Z, x=names, y=names, text=hover, hoverinfo="text"))
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        if show_symbol_table:
            st.markdown("#### Tabla simbólica (para auditoría)")
            st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Axiomas y tipo de objeto")
    st.write(
        {
            "Completitud": check_completeness(P),
            "Transitividad": check_transitivity(P),
            "Antisimetría": check_antisymmetry(P),  # esperable False salvo n=1
            "Tipo": "Preorden completo (dos clases de equivalencia)"
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Economía -------------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Lectura económica (sin poesía, con sustancia)")
    st.markdown(
        """
Esta preferencia dicotómica es exactamente lo que aparece en varios contextos:

- **Screening / aceptación**: “pasa el criterio mínimo” vs “no pasa” (regulación, evaluación de proyectos).
- **Approval voting**: el agente no ordena; aprueba un subconjunto \(P\).
- **Satisficing** (Simon): regla por umbral en vez de maximización fina.
- **Economía pública**: clasificación de elegibilidad (programas focalizados) con errores de medición.

La gracia formal: la utilidad indicadora \(u(x)\in\{0,1\}\) es ordinal y **no** pretende medir intensidad.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if mode.startswith("Econ") and theta is not None and P_true is not None and P_hat is not None and rule is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mini-experimento: screening con variable latente θ y medición imperfecta")
        df = pd.DataFrame(
            {
                "Alternativa": names,
                "θ (latente)": np.round(theta, 3),
                "Verdad (θ≥θ_min)": P_true,
                "Decisión observada (P)": P_hat,
                "u(x)": [u(bool(v)) for v in P_hat],
            }
        ).sort_values(["u(x)", "θ (latente)"], ascending=[False, False])
        st.dataframe(df, use_container_width=True, hide_index=True)

        cc = confusion_counts(P_true, P_hat)
        st.write(
            {
                "Umbral θ_min": rule.theta_min,
                "σ (ruido)": rule.noise,
                "Confusión": cc,
                "Tasa FP": (cc["FP"] / max(1, (cc["FP"] + cc["TN"]))),
                "Tasa FN": (cc["FN"] / max(1, (cc["FN"] + cc["TP"]))),
            }
        )

        st.markdown(
            """
**Lectura:** la política induce un subconjunto aprobado \(P\), pero con medición imperfecta aparecen FP/FN.
La preferencia dicotómica sigue siendo consistente; lo que cambia es el *mecanismo* que define \(P\).
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Mensaje -------------------------------
with tab4:
    st.subheader("Mensaje listo (técnico, corto, con sonrisa)")
    msg = f"""Leidy,

Formalizé tu construcción como una preferencia dicotómica sobre X mediante una partición P/Pᶜ.
La preferencia estricta queda: x ≻ y ⇔ (x∈P) ∧ (y∈Pᶜ), con indiferencia dentro de cada bloque.

La lectura económica es directa: esto es una regla de aceptación (screening/approval) representable por utilidad indicadora u(x)∈{{0,1}}.
Lo interesante no es “rankear”; es la rigidez: dos clases de equivalencia y cero orden intra-bloque.
Cualquier refinamiento exige supuestos extra (umbrales adicionales, atributos medibles o estructura lexicográfica).

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area("Copia/pega", msg, height=260)

    st.markdown('<div class="muted">Nota: el app evita LaTeX dentro de gráficos (plotly) para que todo sea legible y profesional.</div>', unsafe_allow_html=True)
