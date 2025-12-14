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
      .highlight-box { 
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1)); 
        border-left: 4px solid rgba(99,102,241,0.8);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 12px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Modelo formal -------------------------------

def u(in_P: bool) -> int:
    """Utilidad indicadora: u(x)=1 si x∈P, u(x)=0 si x∈Pᶜ."""
    return 1 if in_P else 0

def weak_pref(i: int, j: int, P: List[bool]) -> bool:
    """
    Preferencia débil inducida por u(x)∈{0,1}:
    x ⪰ y  ⇔  u(x) ≥ u(y)
           ⇔  ¬(u(x) < u(y))
           ⇔  ¬(x∈Pᶜ ∧ y∈P)
           ⇔  (x∈P) ∨ (y∈Pᶜ)   [De Morgan]
    """
    return P[i] or (not P[j])

def strict_pref(i: int, j: int, P: List[bool]) -> bool:
    """Preferencia estricta: x ≻ y ⇔ (x∈P) ∧ (y∈Pᶜ)."""
    return P[i] and (not P[j])

def indifferent(i: int, j: int, P: List[bool]) -> bool:
    """Indiferencia: x ~ y si están en el mismo bloque (P o Pᶜ)."""
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
    """Completitud: ∀x,y, x⪰y ∨ y⪰x."""
    n = len(P)
    for i in range(n):
        for j in range(n):
            if not (weak_pref(i, j, P) or weak_pref(j, i, P)):
                return False
    return True

def check_transitivity(P: List[bool]) -> bool:
    """Transitividad de ⪰: (x⪰y ∧ y⪰z) ⇒ x⪰z."""
    n = len(P)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weak_pref(i, j, P) and weak_pref(j, k, P) and (not weak_pref(i, k, P)):
                    return False
    return True

def has_strict_order(P: List[bool]) -> bool:
    """
    Verifica si la relación es antisimétrica (orden parcial).
    Para preferencias dicotómicas con |P|≥2 y |Pᶜ|≥2, esto NO se cumple
    (hay indiferencia entre elementos distintos dentro de cada bloque).
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
    """
    Matriz de confusión para clasificación binaria.
    - FP (False Positive): error de inclusión (asignar a quien no cumple)
    - FN (False Negative): error de exclusión (omitir a quien sí cumple)
    """
    tp = int(np.sum(true_yes & decided_yes))
    tn = int(np.sum((~true_yes) & (~decided_yes)))
    fp = int(np.sum((~true_yes) & decided_yes))
    fn = int(np.sum(true_yes & (~decided_yes)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def policy_loss(cc: Dict[str, int], c_fp: float, c_fn: float) -> float:
    """
    Pérdida operativa: costo por error de asignación.
    No es bienestar completo; es métrica de targeting.
    """
    return c_fp * cc["FP"] + c_fn * cc["FN"]

def safe_rate(numerator: int, denominator: int) -> float | None:
    """Retorna tasa o None si denominador es cero."""
    return numerator / denominator if denominator > 0 else None

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
    show_insights = st.checkbox("Mostrar insight teórico adicional", value=False)

# ------------------------------- Derivados -------------------------------

S, Z = build_relation(names, P)
P_set = [names[i] for i in range(n) if P[i]]
Pc_set = [names[i] for i in range(n) if not P[i]]

k1, k2, k3, k4 = st.columns(4)
k1.metric("|P|", len(P_set))
k2.metric("|Pᶜ|", len(Pc_set))
k3.metric("Completitud (⪰)", "✓" if check_completeness(P) else "✗")
k4.metric("Transitividad (⪰)", "✓" if check_transitivity(P) else "✗")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📐 Demostración formal", 
    "🎨 Estructura visual", 
    "💼 Economía (precisa)", 
    "✉️ Mensaje para enviar",
    "🔬 Extra: Teoría"
])

# ------------------------------- Demostración formal -------------------------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {leidy_name}, tu demo formalizada (con precisión)")
    
    st.markdown("**1) Partición del conjunto de alternativas**")
    st.latex(r"\text{Sea } X \text{ el conjunto de alternativas. Defina } P\subset X \text{ y } P^c = X\setminus P.")

    st.markdown("**2) Preferencia estricta**")
    st.latex(r"x \succ y \iff (x\in P)\wedge (y\in P^c).")

    st.markdown("**3) Indiferencia**")
    st.latex(r"x \sim y \iff \big((x\in P)\wedge(y\in P)\big) \vee \big((x\in P^c)\wedge(y\in P^c)\big).")

    st.markdown("**4) Preferencia débil y representación por utilidad indicadora**")
    st.latex(r"u(x)=\begin{cases}1 & \text{si } x\in P\\0 & \text{si } x\in P^c\end{cases}")
    st.latex(r"x \succeq y \iff u(x)\geq u(y).")
    
    st.markdown("**5) Equivalencia lógica (derivación)**")
    st.latex(r"x \succeq y \iff u(x)\geq u(y) \iff \neg(x\in P^c \wedge y\in P) \iff (x\in P) \vee (y\in P^c).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Consecuencia estructural")
    st.markdown(
        r"""
- El modelo induce **exactamente dos clases de equivalencia**: $P$ y $P^c$.  
- La relación $\succeq$ es **completa** y **transitiva** $\Rightarrow$ es un **preorden completo**.  
- Usualmente **no** es antisimétrica (hay indiferencia con $x\neq y$), por eso no es orden parcial estricto.
- La representación numérica $u: X \to \{0,1\}$ es una **función de utilidad ordinal**.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Visual -------------------------------
with tab2:
    colA, colB = st.columns([1.0, 1.05], gap="large")

    with colA:
        st.subheader("Grafo bipartito de dominancia estricta (P → Pᶜ)")
        st.markdown(r'<div class="muted">Las flechas codifican $x\succ y$. Dentro de bloques: indiferencia.</div>', unsafe_allow_html=True)

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
                    marker=dict(size=22, color=[1 if P[i] else 0 for i in node_ids], colorscale="Viridis"),
                    showlegend=False,
                )
            )

            xs, ys = [], []
            for i in top:
                for j in bot:
                    xs += [pos[i][0], pos[j][0], None]
                    ys += [pos[i][1], pos[j][1], None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", 
                line=dict(width=1.5, color="rgba(99,102,241,0.3)"), 
                hoverinfo="skip", showlegend=False
            ))

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
        st.markdown('<div class="muted">El símbolo aparece en hover sobre cada celda.</div>', unsafe_allow_html=True)

        hover = [[f"{names[i]} vs {names[j]}: {S[i,j]}" for j in range(n)] for i in range(n)]
        fig_h = go.Figure(go.Heatmap(
            z=Z, x=names, y=names, text=hover, hoverinfo="text",
            colorscale="Viridis"
        ))
        fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        if show_symbol_table:
            st.markdown("#### Tabla simbólica (auditoría)")
            st.dataframe(pd.DataFrame(S, index=names, columns=names), use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Axiomas y tipo de objeto (diagnóstico formal)")
    
    completeness = check_completeness(P)
    transitivity = check_transitivity(P)
    antisymmetry = has_strict_order(P)
    
    st.write({
        "Completitud (⪰)": "✓ Cumple" if completeness else "✗ No cumple",
        "Transitividad (⪰)": "✓ Cumple" if transitivity else "✗ No cumple",
        "Antisimetría": "✓ Es orden parcial" if antisymmetry else "✗ Hay indiferencias no triviales",
        "Tipo de estructura": "Preorden completo con dos clases de equivalencia"
    })
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Economía -------------------------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Intento de lectura económica (hice mi tarea 📚)")
    st.markdown(
        r"""
Interpretación desde teoría económica:

- **Screening / regla de elegibilidad**: $P$ es el conjunto "aprobado/elegible"; $P^c$ es "no elegible".  
- **Señal con error**: la decisión puede basarse en una medición imperfecta $\hat{\theta}=\theta+\varepsilon$.  
- **Errores de targeting**:  
  - **FP (false positive)**: error de inclusión — asignas elegibilidad a quien no cumple el criterio latente.  
  - **FN (false negative)**: error de exclusión — excluyes a quien sí cumple el criterio latente.  

**Nota conceptual:** la preferencia dicotómica no "mide intensidad"; modela una **decisión binaria** coherente con una regla de asignación. La utilidad $u(x)\in\{0,1\}$ es puramente ordinal, no cardinal.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if theta is not None and true_eligible is not None and decided_eligible is not None and rule is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Mini-experimento reproducible: elegibilidad con medición imperfecta")

        df = pd.DataFrame({
            "Alternativa": names,
            "θ (latente)": np.round(theta, 3),
            "Elegible (verdad: θ≥θ_min)": true_eligible,
            "Elegible (decisión: θ̂≥θ_min)": decided_eligible,
            "u(x)": [u(bool(v)) for v in decided_eligible],
        }).sort_values(["u(x)", "θ (latente)"], ascending=[False, False])

        st.dataframe(df, use_container_width=True, hide_index=True)

        cc = confusion_counts(true_eligible, decided_eligible)

        c1, c2, c3 = st.columns([1, 1, 2])
        c_fp = c1.number_input("Costo por FP (c_FP)", value=1.0, step=0.5)
        c_fn = c2.number_input("Costo por FN (c_FN)", value=2.0, step=0.5)

        loss = policy_loss(cc, float(c_fp), float(c_fn))
        c3.metric("Pérdida operativa (c_FP·FP + c_FN·FN)", f"{loss:.2f}")

        fpr = safe_rate(cc["FP"], cc["FP"] + cc["TN"])
        fnr = safe_rate(cc["FN"], cc["FN"] + cc["TP"])
        
        st.write({
            "θ_min (umbral)": rule.theta_min,
            "σ (error de medición)": rule.sigma,
            "Confusión": cc,
            "FPR (FP/(FP+TN))": f"{fpr:.3f}" if fpr is not None else "N/A (sin negativos verdaderos)",
            "FNR (FN/(FN+TP))": f"{fnr:.3f}" if fnr is not None else "N/A (sin positivos verdaderos)",
        })

        st.markdown(
            r"""
**Interpretación:** la regla induce $P$ como conjunto elegible observado.  
La preferencia dicotómica sigue siendo consistente; lo que se discute económicamente es el **mecanismo de medición** y el **trade-off** entre FP y FN (costos de asignación).

El parámetro $\sigma$ controla el nivel de ruido: a mayor $\sigma$, mayor probabilidad de clasificación errónea.
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Mensaje -------------------------------
with tab4:
    st.subheader("No es mucho pero es trabajo honesto 💪")

    msg = f"""{leidy_name},

Formalizé tu construcción como una preferencia dicotómica sobre X mediante una partición P/Pᶜ:
x ≻ y ⇔ (x∈P) ∧ (y∈Pᶜ), con indiferencia dentro de cada bloque.
La preferencia débil se representa con utilidad indicadora u(x)∈{{0,1}} y x ⪰ y ⇔ u(x) ≥ u(y).

Mi lectura: esto es una regla de elegibilidad (screening/approval). No pretende "rankear" intensidad;
define consistencia ordinal entre aprobados y no aprobados. El punto interesante es la rigidez:
dos clases de equivalencia y cero orden intra-bloque; cualquier refinamiento exige supuestos extra
(atributos adicionales, umbrales, o estructura lexicográfica).

P={P_set if P_set else ['∅']} | Pᶜ={Pc_set if Pc_set else ['∅']}.

La estructura es un preorden completo (completo + transitivo, pero no antisimétrico).
La representación numérica u: X → {{0,1}} es minimal pero suficiente para capturar el orden.
"""
    st.text_area("Copia/pega", msg, height=300)

    st.markdown('<div class="muted">Nota técnica: las ecuaciones LaTeX se renderizan correctamente con st.latex().</div>', unsafe_allow_html=True)

# ------------------------------- Extra: Teoría -------------------------------
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Insight teórico adicional: representabilidad y extensiones")
    
    if show_insights:
        st.markdown(
            r"""
#### 1. Teorema de representación (Debreu, 1954)
Una relación de preferencia $\succeq$ sobre $X$ admite representación por función de utilidad $u: X \to \mathbb{R}$ 
tal que $x \succeq y \iff u(x) \geq u(y)$ si y solo si:
- $\succeq$ es completa
- $\succeq$ es transitiva  
- $X$ es numerable o $\succeq$ es continua (en espacios topológicos)

**En nuestro caso:** $X$ es finito, $\succeq$ es completa y transitiva $\Rightarrow$ siempre existe representación numérica. 
La función $u(x) \in \{0,1\}$ es la **más simple** (solo 2 valores), pero podríamos usar cualquier $u: P \to \{a\}, P^c \to \{b\}$ con $a > b$.

---

#### 2. Unicidad de la representación
La función $u$ es única **salvo transformaciones monótonas crecientes**. Es decir, si $u$ representa $\succeq$, 
entonces $v = f \circ u$ también representa $\succeq$ si $f$ es estrictamente creciente.

Ejemplo: $u(x) \in \{0, 1\}$ y $v(x) = 100 \cdot u(x) \in \{0, 100\}$ representan la misma preferencia.

---

#### 3. Extensión a más clases
¿Qué pasa si queremos más de 2 clases de equivalencia? Necesitamos:
- Una partición de $X$ en $k$ bloques: $X = C_1 \sqcup C_2 \sqcup \ldots \sqcup C_k$
- Un orden total sobre los bloques: $C_1 \succ C_2 \succ \ldots \succ C_k$
- Función de utilidad: $u(x) = i$ si $x \in C_i$

Esto sigue siendo un **preorden completo**, pero con $k$ clases de equivalencia en lugar de 2.

---

#### 4. Conexión con teoría de elección social
En teoría de votación/elección social (Arrow, Sen), las preferencias dicotómicas aparecen como:
- **Approval voting**: cada votante "aprueba" o "rechaza" candidatos (partición binaria)
- **Quota rules**: una alternativa es elegida si supera un umbral de aprobación

La agregación de preferencias dicotómicas individuales en una decisión colectiva es menos problemática 
que la agregación de rankings completos (evita paradojas como ciclos de Condorcet en muchos casos).

---

#### 5. Limitación fundamental
La preferencia dicotómica **no** puede distinguir intensidades dentro de cada bloque. Por ejemplo:
- Si $x_1, x_2 \in P$, el modelo dice $x_1 \sim x_2$ (indiferencia)
- Pero en la realidad, podrías preferir $x_1$ sobre $x_2$ (preferencia débil pero no estricta dentro de $P$)

Para capturar esto, necesitas:
- **Refinamiento de la partición** (más clases)
- **Atributos multidimensionales** (preferencias lexicográficas)
- **Estructura probabilística** (loterías sobre alternativas)
"""
        )
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown(
            r"""
**Conclusión filosófica:** las preferencias dicotómicas son un modelo **minimal pero coherente**. 
Sacrifican riqueza expresiva (intensidad intra-bloque) a cambio de simplicidad analítica y robustez axiomática.
Son ideales para modelar decisiones binarias institucionales (elegibilidad, aprobación, cumplimiento de umbral)
donde la granularidad fina no es necesaria o no es observable.
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Activa 'Mostrar insight teórico adicional' en el sidebar para ver el análisis profundo.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------- Footer -------------------------------
st.divider()
st.caption("Hecho con 🧠, ☕ y mucho respeto matemático para Leidy. No me quemes porfavor!")
