from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterable

import itertools
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# Preferencias dicotómicas P / Pᶜ — Leidy edition (precisa)
# Streamlit Cloud friendly: sin matplotlib / sin reportlab
# + Calculadora viva: ingresa preferencias y se infiere P/Pᶜ
# + Diagnóstico: incompletitud + núcleo mínimo inconsistente (MUS)
# + Extensión a k clases (preorden con k clases)
# + Economía: screening + umbral óptimo por pérdida
# + Export: nota técnica MD/HTML
# ============================================================

st.set_page_config(
    page_title="Preferencias dicotómicas — Leidy edition",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .card { border: 1px solid rgba(250,250,250,0.12); border-radius: 16px; padding: 14px 16px; background: rgba(255,255,255,0.02); }
      .muted { color: rgba(250,250,250,0.72); font-size: 0.95rem; }
      .h { font-weight: 650; }
      .highlight-box { 
        background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.10)); 
        border-left: 4px solid rgba(99,102,241,0.8);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 12px 0;
      }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
      .small { color: rgba(250,250,250,0.72); font-size: 0.90rem; }
      .tag { display:inline-block; padding: 3px 8px; border: 1px solid rgba(250,250,250,0.15); border-radius: 999px; font-size: 0.85rem; color: rgba(250,250,250,0.75); }
      code { font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------- Modelo formal (2 clases) -------------------------------

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

def loss_curve_optimal_threshold(
    theta: np.ndarray,
    sigma: float,
    c_fp: float,
    c_fn: float,
    rng: np.random.Generator,
    grid: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    """
    Calcula pérdida promedio por umbral (simulación directa sobre la muestra theta):
      - true_eligible depende del umbral
      - decided_eligible usa theta_hat = theta + eps (misma sigma)
    Devuelve: (theta_star, df con columnas: theta_min, loss, FP, FN, FPR, FNR)
    """
    # fijar un eps para reproducibilidad del barrido (evita ruido por umbral)
    eps = rng.normal(0.0, sigma, size=theta.shape)
    theta_hat = theta + eps

    rows = []
    for tmin in grid:
        true_yes = theta >= tmin
        decided_yes = theta_hat >= tmin
        cc = confusion_counts(true_yes, decided_yes)
        L = policy_loss(cc, c_fp, c_fn)
        fpr = safe_rate(cc["FP"], cc["FP"] + cc["TN"])
        fnr = safe_rate(cc["FN"], cc["FN"] + cc["TP"])
        rows.append({
            "theta_min": float(tmin),
            "loss": float(L),
            "FP": cc["FP"],
            "FN": cc["FN"],
            "FPR": float(fpr) if fpr is not None else np.nan,
            "FNR": float(fnr) if fnr is not None else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("theta_min")
    theta_star = float(df.loc[df["loss"].idxmin(), "theta_min"])
    return theta_star, df


# ------------------------------- Calculadora viva: preferencias -> P/Pᶜ -------------------------------

@dataclass(frozen=True)
class PreferenceStatement:
    x: str
    relation: str  # "≻" o "∼"
    y: str

def _normalize_list(raw: str) -> List[str]:
    toks = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    # mantener orden, quitar duplicados
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def summarize_pref_input(prefs: List[PreferenceStatement]) -> str:
    if not prefs:
        return "∅"
    parts = [f"{p.x} {p.relation} {p.y}" for p in prefs]
    return "; ".join(parts)

def infer_partition_from_preferences(
    alternatives: List[str],
    prefs: List[PreferenceStatement],
    default_unassigned: bool = False,  # convención: no asignados -> Pᶜ (False)
) -> Tuple[bool, Dict[str, bool] | None, str, Dict[str, str]]:
    """
    Racionaliza (si es posible) comparaciones como preferencia dicotómica.

    Reglas:
      - x ≻ y  => x ∈ P y y ∈ Pᶜ
      - x ∼ y  => x y y en el mismo bloque

    Retorna:
      ok,
      assignment (a -> bool),
      mensaje,
      audit (a -> razón de asignación/forzamiento o "default")
    """
    assignment: Dict[str, bool] = {}
    audit: Dict[str, str] = {}

    def force(a: str, val: bool, reason: str) -> bool:
        if a in assignment and assignment[a] != val:
            return False
        if a not in assignment:
            assignment[a] = val
            audit[a] = reason
        return True

    # 1) Procesar estrictas
    for p in prefs:
        if p.relation == "≻":
            if not force(p.x, True, f"Forzado a P por: {p.x} ≻ {p.y}"):
                return False, None, f"Inconsistencia: {p.x} queda forzado simultáneamente a P y a Pᶜ.", audit
            if not force(p.y, False, f"Forzado a Pᶜ por: {p.x} ≻ {p.y}"):
                return False, None, f"Inconsistencia: {p.y} queda forzado simultáneamente a P y a Pᶜ.", audit

    # 2) Propagar indiferencias
    changed = True
    while changed:
        changed = False
        for p in prefs:
            if p.relation != "∼":
                continue
            x, y = p.x, p.y
            if x in assignment and y not in assignment:
                assignment[y] = assignment[x]
                audit[y] = f"Propagado por indiferencia: {x} ∼ {y}"
                changed = True
            elif y in assignment and x not in assignment:
                assignment[x] = assignment[y]
                audit[x] = f"Propagado por indiferencia: {x} ∼ {y}"
                changed = True
            elif x in assignment and y in assignment:
                if assignment[x] != assignment[y]:
                    return False, None, f"Inconsistencia: declaraste {x} ∼ {y}, pero quedaron forzados a bloques distintos.", audit

    # 3) Completar no asignados (convención)
    for a in alternatives:
        if a not in assignment:
            assignment[a] = bool(default_unassigned)
            audit[a] = "Asignación por convención (no identificado por tus comparaciones)"

    return True, assignment, "OK: existe una partición P/Pᶜ que racionaliza exactamente las comparaciones ingresadas.", audit

def incompleteness_metrics(alternatives: List[str], prefs: List[PreferenceStatement]) -> Dict[str, float | int | List[Tuple[str, str]]]:
    """
    Mide cobertura de comparaciones:
      - total pares no ordenados: n(n-1)/2
      - observados: pares (x,y) sobre los que se declaró ≻ o ∼ (ignorando dirección)
      - faltantes: lista de pares no observados (puede ser larga; se recorta para UI)
    """
    n = len(alternatives)
    total_pairs = n * (n - 1) // 2

    observed = set()
    for p in prefs:
        if p.x == p.y:
            continue
        observed.add(_pair_key(p.x, p.y))

    all_pairs = set(_pair_key(alternatives[i], alternatives[j]) for i in range(n) for j in range(i + 1, n))
    missing = sorted(list(all_pairs - observed))

    coverage = (len(observed) / total_pairs) if total_pairs > 0 else 1.0

    return {
        "n": n,
        "total_pairs": total_pairs,
        "observed_pairs": len(observed),
        "coverage": float(coverage),
        "missing_pairs": missing,
    }

def is_feasible_dichotomy(alternatives: List[str], prefs: List[PreferenceStatement], default_unassigned: bool) -> bool:
    ok, _, _, _ = infer_partition_from_preferences(alternatives, prefs, default_unassigned=default_unassigned)
    return ok

def find_min_inconsistent_subset(
    alternatives: List[str],
    prefs: List[PreferenceStatement],
    default_unassigned: bool,
    max_steps: int = 4000,
) -> List[PreferenceStatement]:
    """
    Encuentra un subconjunto inconsistente mínimo (MUS) por algoritmo "deletion-based".
    - Si prefs ya es consistente, devuelve [].
    - Si es inconsistente, devuelve subconjunto que sigue siendo inconsistente y donde
      remover cualquier elemento lo vuelve consistente (mínimo por eliminación, no necesariamente único).

    Nota: m <= ~24 recomendado. Para UI lo acotamos.
    """
    if is_feasible_dichotomy(alternatives, prefs, default_unassigned):
        return []

    mus = list(prefs)
    steps = 0
    changed = True
    while changed and steps < max_steps:
        changed = False
        for i in range(len(mus)):
            steps += 1
            test = mus[:i] + mus[i + 1:]
            if not test:
                continue
            if not is_feasible_dichotomy(alternatives, test, default_unassigned):
                # si sigue inconsistente sin este elemento, lo eliminamos
                mus = test
                changed = True
                break
    return mus

def parse_preferences_text(raw: str, valid_alts: List[str]) -> Tuple[List[PreferenceStatement], List[str]]:
    """
    Parseo robusto:
      líneas tipo: x1 ≻ x2  o  x1 ~ x2  o  x1 ∼ x2
      separadores: espacios
    Valida:
      - símbolos aceptados: ≻, ~, ∼
      - alternativas en el set
      - x != y
    """
    prefs: List[PreferenceStatement] = []
    errors: List[str] = []
    alts_set = set(valid_alts)

    if not raw.strip():
        return prefs, errors

    for ln, line in enumerate(raw.splitlines(), start=1):
        s = line.strip()
        if not s:
            continue
        # normalizar tilde de indiferencia
        s = s.replace("~", "∼")
        # intentar separar por espacios
        parts = s.split()
        if len(parts) != 3:
            errors.append(f"L{ln}: formato inválido. Usa: x ≻ y  o  x ∼ y")
            continue
        x, rel, y = parts
        if rel not in {"≻", "∼"}:
            errors.append(f"L{ln}: relación inválida '{rel}'. Usa ≻ o ∼")
            continue
        if x not in alts_set:
            errors.append(f"L{ln}: alternativa '{x}' no está en X.")
            continue
        if y not in alts_set:
            errors.append(f"L{ln}: alternativa '{y}' no está en X.")
            continue
        if x == y:
            errors.append(f"L{ln}: x=y no aporta (se ignora).")
            continue
        prefs.append(PreferenceStatement(x=x, relation=rel, y=y))

    return prefs, errors


# ------------------------------- k-clases: inferencia simple y consistente -------------------------------

@dataclass(frozen=True)
class KPreference:
    x: str
    relation: str  # ">" for strict, "=" for indiff
    y: str

def infer_k_classes(
    alternatives: List[str],
    prefs: List[PreferenceStatement],
    k: int,
) -> Tuple[bool, Optional[Dict[str, int]], str]:
    """
    Extensión k-clases (ordinal):
      - x ≻ y  => class(x) > class(y)
      - x ∼ y  => class(x) = class(y)

    Resuelve por:
      1) componentes de igualdad
      2) DAG entre componentes por strict
      3) longest-path levels (mínimos) y compresión si levels <= k

    Devuelve asignación (alt -> level 0..k-1) donde mayor = "mejor".
    """
    if k < 2:
        return False, None, "k debe ser ≥ 2."

    # 1) Union-find para igualdades
    parent = {a: a for a in alternatives}
    rank = {a: 0 for a in alternatives}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for p in prefs:
        if p.relation == "∼":
            union(p.x, p.y)

    # componentes
    comp_members: Dict[str, List[str]] = {}
    for a in alternatives:
        r = find(a)
        comp_members.setdefault(r, []).append(a)
    comps = list(comp_members.keys())

    # 2) DAG de componentes por strict
    edges = set()
    for p in prefs:
        if p.relation == "≻":
            cx, cy = find(p.x), find(p.y)
            if cx == cy:
                return False, None, f"Inconsistencia: declaraste {p.x} ≻ {p.y} pero también quedaron iguales por ∼."
            edges.add((cx, cy))  # cx > cy

    # 3) detectar ciclos (toposort)
    succ: Dict[str, List[str]] = {c: [] for c in comps}
    indeg: Dict[str, int] = {c: 0 for c in comps}
    for a, b in edges:
        succ[a].append(b)
        indeg[b] += 1

    q = [c for c in comps if indeg[c] == 0]
    topo = []
    while q:
        c = q.pop()
        topo.append(c)
        for nb in succ[c]:
            indeg[nb] -= 1
            if indeg[nb] == 0:
                q.append(nb)

    if len(topo) != len(comps):
        return False, None, "Inconsistencia: las restricciones estrictas inducen un ciclo (no hay orden ordinal que las satisfaga)."

    # longest-path levels (mínimos)
    level = {c: 0 for c in comps}
    # como edges representan ">" (mejor a peor), imponemos level[a] >= level[b] + 1
    for a in topo:
        for b in succ[a]:
            level[b] = max(level[b], level[a] + 1)

    # normalizar para que mayor nivel sea "mejor": actualmente peor tiene niveles más altos.
    # Invertimos para que "mejor" sea mayor:
    maxL = max(level.values()) if level else 0
    inv = {c: maxL - level[c] for c in comps}  # mayor = mejor

    used_levels = sorted(set(inv.values()))
    # reindexar a 0..m-1
    map_to_dense = {lv: i for i, lv in enumerate(used_levels)}
    dense = {c: map_to_dense[inv[c]] for c in comps}

    m = max(dense.values()) + 1 if dense else 1
    if m > k:
        return False, None, f"No cabe en k={k}: las restricciones requieren al menos {m} clases distintas."

    # asignación por alternativa
    assign = {}
    for c, members in comp_members.items():
        for a in members:
            assign[a] = dense[c]

    return True, assign, f"OK: restricciones racionalizables en {m} clases (≤ k={k})."


# ------------------------------- Export (MD/HTML) -------------------------------

def make_report_markdown(
    leidy_name: str,
    X: List[str],
    prefs: List[PreferenceStatement],
    ok2: bool,
    assign2: Optional[Dict[str, bool]],
    incom: Dict[str, float | int | List[Tuple[str, str]]],
    mus: List[PreferenceStatement],
    k_ok: bool,
    k_assign: Optional[Dict[str, int]],
    k: int,
) -> str:
    lines = []
    lines.append(f"# Nota técnica — Preferencias (Leidy edition)")
    lines.append("")
    lines.append(f"**Destinataria:** {leidy_name}")
    lines.append("")
    lines.append("## 1) Entrada")
    lines.append(f"- |X| = {len(X)}")
    lines.append(f"- Comparaciones declaradas (m) = {len(prefs)}")
    lines.append(f"- Cobertura pares = {incom['observed_pairs']}/{incom['total_pairs']} = {incom['coverage']:.3f}")
    lines.append("")
    lines.append("### Comparaciones")
    if prefs:
        for p in prefs:
            lines.append(f"- {p.x} {p.relation} {p.y}")
    else:
        lines.append("- ∅")
    lines.append("")
    lines.append("## 2) Diagnóstico dicotómico (P/Pᶜ)")
    if ok2 and assign2 is not None:
        Pset = [a for a, v in assign2.items() if v]
        Pcset = [a for a, v in assign2.items() if not v]
        lines.append("- **Resultado:** Consistente (existe partición P/Pᶜ).")
        lines.append(f"- P = {Pset if Pset else ['∅']}")
        lines.append(f"- Pᶜ = {Pcset if Pcset else ['∅']}")
        lines.append("")
        lines.append("Interpretación: estructura compatible con regla binaria tipo *approval/elegibility*; no identifica intensidad intra-bloque.")
    else:
        lines.append("- **Resultado:** Inconsistente (no existe partición P/Pᶜ que satisfaga simultáneamente las comparaciones).")
        if mus:
            lines.append("")
            lines.append("### Núcleo mínimo inconsistente (MUS)")
            for p in mus:
                lines.append(f"- {p.x} {p.relation} {p.y}")
        lines.append("")
        lines.append("Interpretación: hay contradicción lógica entre al menos un conjunto pequeño de comparaciones.")
    lines.append("")
    lines.append("## 3) Extensión a k clases")
    lines.append(f"- k solicitado = {k}")
    if k_ok and k_assign is not None:
        df = pd.DataFrame({"x": list(k_assign.keys()), "clase": list(k_assign.values())}).sort_values(["clase", "x"])
        lines.append("- **Resultado:** Consistente (preorden con clases ordinales).")
        # resumen por clase
        for cl in sorted(df["clase"].unique()):
            items = df[df["clase"] == cl]["x"].tolist()
            lines.append(f"  - Clase {cl}: {items}")
    else:
        lines.append("- **Resultado:** No consistente en k clases (o requiere más clases).")
    lines.append("")
    lines.append("## 4) Hasta aqui entendì")
    lines.append("")
    lines.append("Si quieres, el siguiente paso natural sería:")
    lines.append("")
    lines.append("- Clase completa de economica, o")
    lines.append("- lecturas relacionadas, o")
    lines.append("- Quemar en la hoguera.")
    lines.append("")
    return "\n".join(lines)

def markdown_to_simple_html(md: str) -> str:
    # HTML minimalista (sin librerías externas)
    # - convierte headers simples y listas
    html = ["<html><head><meta charset='utf-8'><style>",
            "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto; padding:24px; max-width:980px; margin:auto;}",
            "h1{font-size:26px;} h2{font-size:20px; margin-top:22px;} h3{font-size:16px; margin-top:18px;}",
            "code{background:#f2f2f2; padding:2px 5px; border-radius:6px;}",
            "li{margin:6px 0;} .muted{color:#555;}",
            "</style></head><body>"]
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("# "):
            html.append(f"<h1>{s[2:]}</h1>")
        elif s.startswith("## "):
            html.append(f"<h2>{s[3:]}</h2>")
        elif s.startswith("### "):
            html.append(f"<h3>{s[4:]}</h3>")
        elif s.startswith("- "):
            # abrir lista si no está abierta
            if not html or not html[-1].startswith("<ul"):
                html.append("<ul>")
            html.append(f"<li>{s[2:]}</li>")
        elif s.strip() == "":
            # cerrar lista si aplica
            if html and html[-1] == "</li>":
                pass
            # si el anterior fue lista y la siguiente no, cerramos
            if html and "<ul>" in html[-1:]:
                pass
            html.append("<br/>")
        else:
            # cerrar ul si veníamos de lista
            if html and html[-1].startswith("<li"):
                # buscamos si hay un <ul> abierto al final
                pass
            html.append(f"<p>{s}</p>")
    # cerrar ul si quedó abierta
    if "<ul>" in html:
        # cierre tosco: si hay <ul> sin cerrar, cerramos al final
        # (suficiente para descarga)
        html.append("</ul>")
    html.append("</body></html>")
    return "\n".join(html)


# ------------------------------- UI -------------------------------

st.title("🧠 Preferencias dicotómicas (P / Pᶜ)")
st.caption("Informe técnico: demostración formal + lectura económica (screening / elegibilidad) + calculadora (preferencias → P/Pᶜ) + diagnóstico.")

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


tab1, tab2, tab3, tab_calc, tab_k, tab_rep, tab4, tab5 = st.tabs([
    "📐 Demostración formal",
    "🎨 Estructura visual",
    "💼 Lectura Economica",
    "🧮 Calculadora (preferencias → P/Pᶜ)",
    "🔢 Extensión k-clases",
    "📄 Diagnóstico + Export",
    "✉️ Mensaje",
    "🔬 Extra: Teoría",
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
    st.markdown("### Lectura económica (precisa, sin humo)")
    st.markdown(
        r"""
- **Screening / regla de elegibilidad**: $P$ es "aprobado/elegible"; $P^c$ es "no elegible".  
- **Señal con error**: decisión basada en $\hat{\theta}=\theta+\varepsilon$.  
- **Errores de targeting**:  
  - **FP (false positive)**: error de inclusión.  
  - **FN (false negative)**: error de exclusión.  

**Nota conceptual:** la preferencia dicotómica no mide intensidad. Es un orden ordinal inducido por una regla binaria.
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # experimento base (el de tu sidebar)
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
            "FPR (FP/(FP+TN))": f"{fpr:.3f}" if fpr is not None else "N/A",
            "FNR (FN/(FN+TP))": f"{fnr:.3f}" if fnr is not None else "N/A",
        })

        st.markdown("</div>", unsafe_allow_html=True)

        # Umbral óptimo
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Umbral óptimo (minimiza pérdida operativa)")

        grid = np.linspace(-2.5, 2.5, 101)
        theta_star, df_loss = loss_curve_optimal_threshold(
            theta=theta,
            sigma=rule.sigma,
            c_fp=float(c_fp),
            c_fn=float(c_fn),
            rng=np.random.default_rng(int(seed) + 99),
            grid=grid
        )
        st.metric("θ* (argmin pérdida)", f"{theta_star:.3f}")

        figL = go.Figure()
        figL.add_trace(go.Scatter(x=df_loss["theta_min"], y=df_loss["loss"], mode="lines", name="Pérdida"))
        figL.add_vline(x=theta_star, line_width=2, line_dash="dot")
        figL.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="θ_min",
            yaxis_title="Pérdida (c_FP·FP + c_FN·FN)"
        )
        st.plotly_chart(figL, use_container_width=True)

        st.markdown(
            r"""
**Interpretación:** el umbral óptimo depende de (i) la dispersión del ruido σ y (ii) el trade-off de costos (c_FP, c_FN).  
Esto es targeting operativo, no bienestar social completo.
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- Calculadora viva -------------------------------
with tab_calc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🧮 Calculadora: tú declaras preferencias, yo infiero P/Pᶜ")
    st.markdown(
        """
**Entrada primitiva:** comparaciones declaradas (x ≻ y, x ∼ y).  
**Salida:** si existe una partición P/Pᶜ que racionaliza esas comparaciones; si no, se muestra el conflicto (MUS).  
**Además:** mide incompletitud (qué tanto información falta).
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.subheader("1) Alternativas")
        raw_alts = st.text_area(
            "Lista de alternativas X (coma o salto de línea)",
            value=", ".join(names),
            height=100
        )
        alts = _normalize_list(raw_alts)
        if len(alts) < 2:
            st.warning("Ingresa al menos 2 alternativas.")
            st.stop()

        st.subheader("2) Entrada por texto (pegable)")
        st.markdown('<div class="small">Formato: <span class="mono">x1 ≻ x2</span> o <span class="mono">x1 ∼ x2</span> (una por línea). Se acepta <span class="mono">~</span> como sinónimo de <span class="mono">∼</span>.</div>', unsafe_allow_html=True)
        raw_text = st.text_area(
            "Pega tus comparaciones aquí",
            value="",
            height=120,
            placeholder="x1 ≻ x2\nx1 ∼ x3\nx4 ≻ x2"
        )
        prefs_text, errors = parse_preferences_text(raw_text, alts)
        if errors:
            st.error("Errores en texto:\n- " + "\n- ".join(errors))

        st.subheader("3) Entrada por UI (opcional)")
        m = st.number_input("Número de comparaciones (UI)", min_value=0, max_value=24, value=min(6, max(0, len(alts))), step=1)

        prefs_ui: List[PreferenceStatement] = []
        for i in range(int(m)):
            c1, c2, c3 = st.columns([1.0, 0.55, 1.0])
            with c1:
                x = st.selectbox(f"x{i}", alts, key=f"calc_x_{i}")
            with c2:
                rel = st.selectbox(f"rel{i}", ["≻", "∼"], key=f"calc_rel_{i}")
            with c3:
                y = st.selectbox(f"y{i}", alts, index=min(1, len(alts)-1), key=f"calc_y_{i}")

            if x != y:
                prefs_ui.append(PreferenceStatement(x=x, relation=rel, y=y))

        prefs = prefs_text + prefs_ui

        st.markdown('<div class="small">Tip: no necesitas completar todo. Con unas pocas comparaciones ya puedes inferir (o refutar) P/Pᶜ.</div>', unsafe_allow_html=True)

        default_unassigned = st.checkbox("Asignar no declarados a P (si quedan libres)", value=False)
        run = st.button("Inferir P / Pᶜ desde preferencias", type="primary")

    with right:
        st.subheader("Resultado")
        if run:
            # incompletitud primero (siempre)
            inc = incompleteness_metrics(alts, prefs)

            ok, assign, msg, audit = infer_partition_from_preferences(alts, prefs, default_unassigned=default_unassigned)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Diagnóstico de incompletitud")
            st.write({
                "n": inc["n"],
                "pares totales": inc["total_pairs"],
                "pares observados": inc["observed_pairs"],
                "cobertura": f"{inc['coverage']:.3f}",
            })
            missing = inc["missing_pairs"]
            if isinstance(missing, list) and missing:
                st.markdown("**Ejemplo de pares NO observados (muestra):**")
                st.write(missing[:min(12, len(missing))])
            st.markdown("</div>", unsafe_allow_html=True)

            if not ok:
                st.error(msg)

                # MUS (núcleo mínimo)
                mus = find_min_inconsistent_subset(alts, prefs, default_unassigned=default_unassigned)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Núcleo mínimo inconsistente (MUS)")
                if mus:
                    st.markdown("Este subconjunto ya es inconsistente; si quitas cualquiera de sus elementos, deja de serlo (mínimo por eliminación).")
                    st.code("\n".join([f"{p.x} {p.relation} {p.y}" for p in mus]), language="text")
                else:
                    st.markdown("No se pudo aislar MUS (raro). Revisa entradas.")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("**Entrada (audit):**")
                st.code(summarize_pref_input(prefs), language="text")

            else:
                P_calc = [a for a, v in assign.items() if v] if assign else []
                Pc_calc = [a for a, v in assign.items() if not v] if assign else []

                st.success(msg)
                st.write({"P": P_calc if P_calc else ["∅"], "Pᶜ": Pc_calc if Pc_calc else ["∅"]})

                # identificabilidad
                forced = [a for a, reason in audit.items() if "convención" not in reason]
                unident = [a for a, reason in audit.items() if "convención" in reason]

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Identificación (qué quedó realmente inferido)")
                st.write({
                    "forzados/identificados": len(forced),
                    "no identificados (por convención)": len(unident),
                })
                if forced:
                    st.markdown("**Ejemplos forzados (con razón):**")
                    for a in forced[:min(8, len(forced))]:
                        st.markdown(f"- `{a}` → {('P' if assign[a] else 'Pᶜ')}  ·  {audit[a]}")
                if unident:
                    st.markdown("**No identificados (sin información suficiente):**")
                    st.write(unident[:min(12, len(unident))])
                st.markdown("</div>", unsafe_allow_html=True)

                # Construir relación inducida y mostrar estructura
                P_bool = [assign[a] for a in alts]
                S2, Z2 = build_relation(alts, P_bool)

                cA, cB, cC, cD = st.columns(4)
                cA.metric("|P|", len(P_calc))
                cB.metric("|Pᶜ|", len(Pc_calc))
                cC.metric("Completitud (⪰)", "✓" if check_completeness(P_bool) else "✗")
                cD.metric("Transitividad (⪰)", "✓" if check_transitivity(P_bool) else "✗")

                st.markdown("#### Matriz simbólica inducida")
                st.dataframe(pd.DataFrame(S2, index=alts, columns=alts), use_container_width=True)

                st.markdown("#### Visual rápido (heatmap ordinal solo para lectura)")
                hover2 = [[f"{alts[i]} vs {alts[j]}: {S2[i,j]}" for j in range(len(alts))] for i in range(len(alts))]
                fig2 = go.Figure(go.Heatmap(
                    z=Z2, x=alts, y=alts, text=hover2, hoverinfo="text", colorscale="Viridis"
                ))
                fig2.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.markdown(
                    """
**Lectura económica correcta (una línea):**  
Si existe P/Pᶜ, lo que declaraste es consistente con una regla binaria tipo *approval / eligibility*.  
La intensidad intra-bloque no está identificada: requiere supuestos extra.
"""
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- k-clases -------------------------------
with tab_k:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔢 Extensión a k clases (preorden ordinal con k niveles)")
    st.markdown(
        """
Aquí no “inventamos” cardinalidad: seguimos ordinal.
- x ≻ y  ⇒ clase(x) > clase(y)
- x ∼ y  ⇒ clase(x) = clase(y)

Si las restricciones requieren más clases que k, se reporta (no se fuerza).
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Reusar alternativas y preferencias del tab_calc en un estado simple
    st.markdown(
        '<div class="small">Tip: pega aquí las mismas comparaciones de la calculadora viva para ver si caben en k clases.</div>',
        unsafe_allow_html=True
    )
    raw_alts_k = st.text_area("Alternativas X", value=", ".join(names), height=90, key="k_alts")
    Xk = _normalize_list(raw_alts_k)
    if len(Xk) < 2:
        st.warning("Ingresa al menos 2 alternativas.")
        st.stop()

    raw_prefs_k = st.text_area(
        "Comparaciones (una por línea)",
        value="",
        height=140,
        placeholder="x1 ≻ x2\nx1 ∼ x3\nx4 ≻ x2",
        key="k_prefs"
    )
    prefs_k, errs_k = parse_preferences_text(raw_prefs_k, Xk)
    if errs_k:
        st.error("Errores:\n- " + "\n- ".join(errs_k))

    k = st.slider("k", 2, 8, 3, 1, key="k_slider")

    run_k = st.button("Inferir clases ordinales (k)", type="primary", key="run_k")

    if run_k:
        if errs_k:
            st.warning("Corrige los errores de formato/alternativas antes de inferir.")
        else:
            k_ok, k_assign, k_msg = infer_k_classes(Xk, prefs_k, k)
            if not k_ok or k_assign is None:
                st.error(k_msg)
            else:
                st.success(k_msg)
                dfk = pd.DataFrame({"Alternativa": list(k_assign.keys()), "Clase": list(k_assign.values())})
                dfk = dfk.sort_values(["Clase", "Alternativa"], ascending=[False, True])

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Asignación (mayor clase = “mejor”)")
                st.dataframe(dfk, use_container_width=True, hide_index=True)

                # Resumen por clases
                st.markdown("### Resumen por clase")
                for cl in sorted(dfk["Clase"].unique(), reverse=True):
                    items = dfk[dfk["Clase"] == cl]["Alternativa"].tolist()
                    st.markdown(f"- **Clase {cl}**: {items}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Visual: heatmap ordinal (k niveles)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Visual ordinal (k-clases)")
                alt_order = dfk["Alternativa"].tolist()
                cls = np.array([k_assign[a] for a in alt_order], dtype=int)

                # Construir símbolo ordinal simple: si clase mayor → ≻, si igual → ∼, si menor → ⪯
                n2 = len(alt_order)
                S_k = np.empty((n2, n2), dtype=object)
                Z_k = np.zeros((n2, n2), dtype=float)

                def sym_k(ci: int, cj: int) -> str:
                    if ci > cj:
                        return "≻"
                    if ci == cj:
                        return "∼"
                    return "⪯"

                for i in range(n2):
                    for j in range(n2):
                        s = sym_k(cls[i], cls[j])
                        S_k[i, j] = s
                        Z_k[i, j] = {"≻": 2.0, "∼": 1.0, "⪯": 0.5}[s]

                hoverk = [[f"{alt_order[i]} vs {alt_order[j]}: {S_k[i,j]}" for j in range(n2)] for i in range(n2)]
                figk = go.Figure(go.Heatmap(
                    z=Z_k, x=alt_order, y=alt_order, text=hoverk, hoverinfo="text", colorscale="Viridis"
                ))
                figk.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(figk, use_container_width=True)

                st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- Diagnóstico + Export -------------------------------
with tab_rep:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Diagnóstico + Export (nota técnica)")
    st.markdown(
        """
Este tab toma tus comparaciones (pegables) y produce:
- Diagnóstico dicotómico (P/Pᶜ)
- Incompletitud (cobertura de pares)
- MUS si hay inconsistencia
- Extensión a k clases (si procede)
- Export MD y HTML (descargables)
"""
    )
    st.markdown("</div>", unsafe_allow_html=True)

    colL, colR = st.columns([1.05, 1.0], gap="large")

    with colL:
        st.subheader("Entrada")
        raw_X = st.text_area("Alternativas X", value=", ".join(names), height=90, key="rep_X")
        X = _normalize_list(raw_X)
        if len(X) < 2:
            st.warning("Ingresa al menos 2 alternativas.")
            st.stop()

        st.markdown(
            '<div class="small">Comparaciones (una por línea): <span class="mono">x1 ≻ x2</span> o <span class="mono">x1 ∼ x2</span></div>',
            unsafe_allow_html=True
        )
        raw_prefs = st.text_area(
            "Comparaciones",
            value="",
            height=160,
            placeholder="x1 ≻ x2\nx1 ∼ x3\nx4 ≻ x2",
            key="rep_prefs"
        )
        prefs, errs = parse_preferences_text(raw_prefs, X)
        if errs:
            st.error("Errores:\n- " + "\n- ".join(errs))

        default_unassigned = st.checkbox(
            "Convención: no identificados van a P (si quedan libres)",
            value=False,
            key="rep_default_unassigned"
        )

        k_rep = st.slider("k para extensión (si aplica)", 2, 8, 3, 1, key="rep_k")
        run_rep = st.button("Generar diagnóstico + export", type="primary", key="run_rep")

    with colR:
        st.subheader("Salida")
        if run_rep:
            if errs:
                st.warning("Corrige errores antes de generar el reporte.")
            else:
                inc = incompleteness_metrics(X, prefs)

                ok2, assign2, msg2, audit2 = infer_partition_from_preferences(
                    X, prefs, default_unassigned=default_unassigned
                )

                mus = []
                if not ok2:
                    mus = find_min_inconsistent_subset(X, prefs, default_unassigned=default_unassigned)

                k_ok, k_assign, k_msg = infer_k_classes(X, prefs, k_rep)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Resumen diagnóstico")
                st.write({
                    "Dicotómico (P/Pᶜ)": "Consistente" if ok2 else "Inconsistente",
                    "Cobertura pares": f"{inc['observed_pairs']}/{inc['total_pairs']} = {inc['coverage']:.3f}",
                    "MUS": len(mus),
                    f"k-clases (k={k_rep})": "OK" if k_ok else "No",
                })
                st.markdown("</div>", unsafe_allow_html=True)

                if ok2 and assign2 is not None:
                    Pset = [a for a, v in assign2.items() if v]
                    Pcset = [a for a, v in assign2.items() if not v]

                    st.success(msg2)
                    st.write({"P": Pset if Pset else ["∅"], "Pᶜ": Pcset if Pcset else ["∅"]})

                    # Identificación
                    forced = [a for a, reason in audit2.items() if "convención" not in reason]
                    unident = [a for a, reason in audit2.items() if "convención" in reason]
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Identificación (qué quedó realmente inferido)")
                    st.write({
                        "forzados/identificados": len(forced),
                        "no identificados (por convención)": len(unident),
                    })
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(msg2)
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### MUS (núcleo mínimo inconsistente)")
                    if mus:
                        st.code("\n".join([f"{p.x} {p.relation} {p.y}" for p in mus]), language="text")
                    else:
                        st.markdown("No se pudo aislar MUS.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # k-clases output
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Extensión k-clases")
                if k_ok and k_assign is not None:
                    st.success(k_msg)
                    dfk = pd.DataFrame({"Alternativa": list(k_assign.keys()), "Clase": list(k_assign.values())})
                    dfk = dfk.sort_values(["Clase", "Alternativa"], ascending=[False, True])
                    st.dataframe(dfk, use_container_width=True, hide_index=True)
                else:
                    st.error(k_msg)
                st.markdown("</div>", unsafe_allow_html=True)

                # Export
                md = make_report_markdown(
                    leidy_name=leidy_name,
                    X=X,
                    prefs=prefs,
                    ok2=ok2,
                    assign2=assign2,
                    incom=inc,
                    mus=mus,
                    k_ok=k_ok,
                    k_assign=k_assign,
                    k=k_rep,
                )
                html = markdown_to_simple_html(md)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### Descargas")
                st.download_button("⬇️ Descargar MD", data=md, file_name="nota_tecnica_preferencias.md", mime="text/markdown")
                st.download_button("⬇️ Descargar HTML", data=html, file_name="nota_tecnica_preferencias.html", mime="text/html")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.markdown(
                    """

"""
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- Mensaje para enviar -------------------------------
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
    st.text_area("Copia/pega", msg, height=330)

    st.markdown('<div class="muted">Nota: las ecuaciones LaTeX se renderizan correctamente con st.latex().</div>', unsafe_allow_html=True)


# ------------------------------- Extra: Teoría -------------------------------
with tab5:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔬 Insight teórico adicional: representabilidad y extensiones")

    if show_insights:
        st.markdown(
            r"""
#### 1) Representación (Debreu / orden finito)
En conjuntos finitos, si $\succeq$ es completa y transitiva, existe $u:X\to\mathbb{R}$ tal que:
$$x\succeq y \iff u(x)\ge u(y).$$
En el caso dicotómico, basta una utilidad indicadora $u\in\{0,1\}$.

#### 2) Unicidad (transformaciones monótonas)
Si $u$ representa $\succeq$, entonces $v=f\circ u$ también representa $\succeq$ para cualquier $f$ estrictamente creciente.

#### 3) Extensión a k clases
Una partición $X=\bigsqcup_{i=0}^{k-1} C_i$ induce un preorden con k clases:
- si $i>j$, entonces $C_i \succ C_j$
- si $x,y\in C_i$, entonces $x\sim y$

#### 4) Limitación estructural
La dicotomía NO identifica orden dentro de cada bloque.
Si quieres refinamiento, tienes que pagar con supuestos (más clases, atributos, o reglas lexicográficas).
"""
        )

        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown(
            r"""
**Conclusión:** esto es un modelo minimalista, pero axiomáticamente limpio.
Sirve perfecto cuando el proceso real es binario (aprobación/elegibilidad).
"""
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("👆 Activa 'Mostrar insight teórico adicional' en el sidebar para ver el análisis profundo.")

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------- Footer -------------------------------
st.divider()
st.caption("Hecho con 🧠, ☕ y mucho respeto matemático para Leidy. No me quemes porfavor!")
st.markdown(
    """
<div class="small">
Nota: Ahora si me quemas.
</div>
""",
    unsafe_allow_html=True,
)
