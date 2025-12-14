from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Preferencias dicotómicas P / Pᶜ — Leidy edition (precisa)
# Streamlit Cloud friendly: sin matplotlib / sin reportlab
# ============================================================

st.set_page_config(page_title="Preferencias dicotómicas — Leidy edition", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .card { border: 1px solid rgba(250,250,250,0.12); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
      .muted { color: rgba(250,250,250,0.72); font-size: 0.95rem; }
      .h { font-weight: 650; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Modelo formal -------------------------------

def u(in_P: bool) -> int:
    """Utilidad indicadora: u(x)=1 si x∈P, u(x)=0 si x∈P^c."""
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    """
    Preferencia débil inducida por u(x)∈{0,1}:
    x ⪰ y  ⇔  u(x) ≥ u(y).
    Con u indicadora, esto es equivalente a: (x∈P) ∨ (y∈P^c).
    """
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    """Preferencia estricta: x ≻ y ⇔ (x∈P) ∧ (y∈P^c)."""
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    """Indiferencia: x ~ y si están en el mismo bloque (P o P^c)."""
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
    # Solo para visual (no cardinal).
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
    """Completitud: ∀x,y, x⪰y o y⪰x."""
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, P) or weak_pref(j, i, P)):
                return False
    return True

def check_transitivity(P: List[bool]) -> bool:
    """Transitividad de ⪰: (x⪰y y y⪰z) ⇒ x⪰z."""
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and (not weak_pref(i, k, P)):
                    return False
    return True

def check_antisymmetry(P: List[bool]) -> bool:
    """
    Antisimetría: (x⪰y y y⪰x) ⇒ x=y.
    Aquí típicamente NO se cumple (hay indiferencia con x≠y), así que es preorden, no orden parcial.
    """
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

# ------------------------------- Capa económica (precisa) -------------------------------

@dataclass
class ScreeningRule:
    """
    Regla de elegibilidad (screening): decisión binaria basada en una señal observada.
    theta: atributo latente (p.ej., calidad / retorno social).
    theta_hat: medición con error.
    """
    theta_min: float  # umbral de elegibilidad
    sigma: float      # desviación estándar del error de medición

def screening(theta: np.ndarray, rule: ScreeningRule, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      true_eligible: 1{theta >= theta_min} (verdad "latente")
      decided_eligible: 1{theta_hat >= theta_min} con theta_hat = theta + eps
    """
    eps = rng.normal(0.0, rule.sigma, size=theta.shape)
    theta_hat = theta + eps
    true_eligible = theta >= rule.theta_min
    decided_eligible = theta_hat >= rule.theta_min
    return true_eligible, decided_eligible

def confusion_counts(true_yes: np.ndarray, decided_yes: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum(true_yes & decided_yes))
    tn = int(np.sum((~true_yes) & (~decided_yes)))
    fp = int(np.sum((~true_yes) & decided_yes))  # Type I error (false positive) w.r.t. eligibility
    fn = int(np.sum(true_yes & (~decided_yes)))  # Type II error (false negative)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def policy_loss(cc: Dict[str, int], c_fp: float, c_fn: float) -> float:
    """
    Pérdida simple (económica): costo por error de asignación.
    No es bienestar completo; es una métrica operativa de targeting.
    """
    return c_fp * cc["FP"] + c_fn * cc["FN"]

# ------------------------------- UI -------------------------------

st.title("🧠 Preferencias dicotómicas (P / Pᶜ) — Leidy edition")
st.caption("Regalo técnico: demostración formal + lectura económica (screening / elegibilidad).")

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
    mode = st.radio("Cómo definir P", ["Manual (Pinky)", "Econ: elegibilidad por umbral (screening)"], index=0)

    seed = st.number_input("Semilla (reproducible)", value=11, step=1)
    rng = np.random.default_rng(int(seed))

    theta = None
    true_eligible = None
    decided_eligible = None
    rule = None

    if mode.startswith("Manual"):
        P = [st.checkbox(f"{names[i]} ∈ P", value=(i % 2 == 0)) for i in range(n)]
    else:
        st.subheader("📈 Screening / elegibilidad")
        theta_min = st.slider("Umbral θ_min", -2.0, 2.0, 0.0, 0.05)
        sigma = st.slider("Error de medición σ", 0.0, 1.0, 0.25, 0.05)
        rule = ScreeningRule(theta_min=theta_min, sigma=sigma)

        # atributo latente
        theta = rng.normal(0.0, 1.0, size=n)
        true_eligible, decided_eligible = screening(theta, rule, rng)

        # En el modelo de preferencia usamos la decisión observada/institucional: P ≡ elegibles decididos
        P = list(decided_eligible.astype(bool))

    st.divider()
    show_symbol_table = st.checkbox("Mostrar tabla de símbolos", value=True)

# ------------------------------- Derivados -------------------------------

S, Z = build_relation(names, P)
P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("|P|", len(P_set))
k2.metric("|Pᶜ|", len(Pc_set))
k3.metric("Completitud (⪰)", "OK" if check_completeness(P) else "NO")
k4.metric("Transitividad (⪰)", "OK" if check_transitivity(P) else "NO")

tab1, tab2, tab3, tab4 = st.tabs(["Demostración (formal)", "Estructura (visual)", "Economía (precisa)", "Mensaje para enviar"])

# ------------------------------- Demostración (formal, con latex correcto) -------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {leidy_name}, tu demo formalizada (con precisión)")
    st.markdown("**1) Partición del conjunto de alternativas**")
    st.latex(r"\text{Sea } X \text{ el conjunto de alternativas. Defina } P\subset X \text{ y } P^c = X\setminus P.")

    st.markdown("**2) Preferencia estricta**")
    st.latex(r"x \succ y \iff (x\in P)\land (y\in P^c).")

    st.markdown("**3) Indiferencia**")
    st.latex(r"x \sim y \iff \big((x\in P)\land(y\in P)\big)\ \ \lor\ \ \big((x\in P^c)\land(y\in P^c)\big).")

    st.markdown("**4) Preferencia débil y representación por utilidad indicadora**")
    st.latex(r"u(x)=\begin{cases}1 & \text{si } x\in P\\0 & \text{si } x\in P^c\end{cases}")
    st.latex(r"x \succeq y \iff u(x)\ge u(y).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Consecuencia estructural (sin exagerar)")
    st.markdown(
        """
- El modelo induce **exactamente dos clases de equivalencia**: \(P\) y \(P^c\).  
- La relación \(\succeq\) es **completa** y **transitiva** ⇒ es un **preorden completo**.  
- Usualmente **no** es antisimétrica (porque hay indiferencia con \(x\neq y\)), por eso no es “orden” en sentido fuerte.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Visual -------------------------------
with tab2:
    colA, colB = st.columns([1.0, 1.05], gap="large")

    with colA:
        st.subheader("Grafo bipartito de dominancia estricta (P → Pᶜ)")
        st.markdown('<div class="muted">Las flechas codifican \(x\succ y\). Dentro de bloques: indiferencia.</div>', unsafe_allow_html=True)

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
        st.subheader("Matriz de la relación (hover = símbolo)")
        st.markdown('<div class="muted">Nada de fórmulas “encima”: el símbolo aparece en hover.</div>', unsafe_allow_html=True)

        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(go.Heatmap(z=Z, x=names, y=names, text=hover, hoverinfo="text"))
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        if show_symbol_table:
            st.markdown("#### Tabla simbólica (auditoría)")
            st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Axiomas y tipo de objeto (diagnóstico formal)")
    st.write(
        {
            "Completitud (⪰)": check_completeness(P),
            "Transitividad (⪰)": check_transitivity(P),
            "Antisimetría": check_antisymmetry(P),
            "Conclusión": "Preorden completo con dos clases de equivalencia"
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Economía (precisa) -------------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Lectura económica (precisa, sin analogías baratas)")
    st.markdown(
        """
Interpretación estándar:

- **Screening / regla de elegibilidad**: \(P\) es el conjunto “aprobado/elegible”; \(P^c\) es “no elegible”.  
- **Señal con error**: la decisión puede basarse en una medición imperfecta \(\hat{\theta}=\theta+\varepsilon\).  
- **Errores de targeting**:  
  - **FP (false positive / Type I)**: asignas elegibilidad a quien no cumple el criterio latente.  
  - **FN (false negative / Type II)**: excluyes a quien sí cumple el criterio latente.  

Punto clave: la preferencia dicotómica no “mide intensidad”; modela una **decisión binaria** coherente con una regla de asignación.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if theta is not None and true_eligible is not None and decided_eligible is not None and rule is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mini-experimento reproducible: elegibilidad con medición imperfecta")

        df = pd.DataFrame(
            {
                "Alternativa": names,
                "θ (latente)": np.round(theta, 3),
                "Elegible (verdad: θ≥θ_min)": true_eligible,
                "Elegible (decisión: θ̂≥θ_min)": decided_eligible,
                "u(x) = 1{decisión}": [u(bool(v)) for v in decided_eligible],
            }
        ).sort_values(["u(x) = 1{decisión}", "θ (latente)"], ascending=[False, False])

        st.dataframe(df, use_container_width=True, hide_index=True)

        cc = confusion_counts(true_eligible, decided_eligible)

        c1, c2, c3 = st.columns([1, 1, 2])
        c_fp = c1.number_input("Costo por FP (c_FP)", value=1.0, step=0.5)
        c_fn = c2.number_input("Costo por FN (c_FN)", value=2.0, step=0.5)

        loss = policy_loss(cc, float(c_fp), float(c_fn))
        c3.metric("Pérdida operativa (c_FP·FP + c_FN·FN)", f"{loss:.2f}")

        st.write(
            {
                "θ_min": rule.theta_min,
                "σ (error de medición)": rule.sigma,
                "Confusión": cc,
                "FPR (FP/(FP+TN))": cc["FP"] / max(1, (cc["FP"] + cc["TN"])),
                "FNR (FN/(FN+TP))": cc["FN"] / max(1, (cc["FN"] + cc["TP"])),
            }
        )

        st.markdown(
            """
**Interpretación:** la regla induce \(P\) como conjunto elegible observado.  
La preferencia dicotómica sigue siendo consistente; lo que se discute económicamente es el **mecanismo de medición** y el **trade-off** entre FP y FN (costos de asignación).
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Mensaje -------------------------------
with tab4:
    st.subheader("Mensaje listo (técnico, breve, sin pretensión)")

    msg = f"""{leidy_name},

Formalizé tu construcción como una preferencia dicotómica sobre X mediante una partición P/Pᶜ:
x ≻ y ⇔ (x∈P) ∧ (y∈Pᶜ), con indiferencia dentro de cada bloque.
La preferencia débil se representa con utilidad indicadora u(x)∈{{0,1}} y x ⪰ y ⇔ u(x) ≥ u(y).

Lectura económica: esto es una regla de elegibilidad (screening/approval). No intenta “rankear” intensidad;
define consistencia ordinal entre aprobados y no aprobados. El punto interesante es la rigidez:
dos clases de equivalencia y cero orden intra-bloque; cualquier refinamiento exige supuestos extra
(atributos adicionales, umbrales, o estructura lexicográfica).

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.
"""
    st.text_area("Copia/pega", msg, height=260)

    st.markdown('<div class="muted">Nota técnica: el LaTeX se renderiza con st.latex() para evitar texto plano.</div>', unsafe_allow_html=True)
