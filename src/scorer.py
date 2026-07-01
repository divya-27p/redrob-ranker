"""Combines a candidate feature dictionary (from feature_extractor.py) into
a single 0-1 ranking score.

Six weighted subscores - experience, technical, career evidence, company
fit, location/logistics, behavior - are blended into a base score. A
capped multiplicative trap penalty is then applied so a single bad signal
tempers a strong score without ever hard-zeroing a candidate.

No external packages. Pure standard-library arithmetic.
"""

from typing import Any, Dict, Tuple

from .utils import clamp

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: Dict[str, float] = {
    "experience": 0.20,
    "technical": 0.25,
    "career_evidence": 0.20,
    "company_fit": 0.10,
    "location": 0.10,
    "behavior": 0.15,
}

# Each trap flag knocks this fraction off the final score (multiplicative,
# so flags compound but can never fully zero out a candidate on their own).
TRAP_PENALTY_PER_FLAG: float = 0.18
MAX_TRAP_PENALTY: float = 0.65


# ---------------------------------------------------------------------------
# 1. Experience subscore
# ---------------------------------------------------------------------------

def _experience_band_score(years: float) -> float:
    """Piecewise reward centered on the 5-9 year ideal band."""
    if 5 <= years <= 9:
        return 1.0
    if 4 <= years <= 10:
        return 0.8
    if 3 <= years < 4 or 10 < years <= 12:
        return 0.5
    if years < 3:
        # steep penalty below 3, floor near 0 for very junior profiles
        return clamp(0.15 + years * 0.1)
    # years > 12: gentle decay for very senior over-qualification
    return clamp(0.5 - (years - 12) * 0.05, lo=0.05, hi=0.5)


def _experience_subscore(features: Dict[str, Any]) -> float:
    band_score = _experience_band_score(float(features.get("years_experience", 0.0)))
    stability = clamp(float(features.get("career_stability_score", 0.0)))
    promotion = clamp(float(features.get("promotion_score", 0.0)))

    # band fit dominates; stability and promotion nudge the score up or down
    raw = band_score * 0.7 + stability * 0.2 + promotion * 0.1
    return clamp(raw)


# ---------------------------------------------------------------------------
# 2. Technical subscore
# ---------------------------------------------------------------------------

def _saturating_ratio(count: int, cap: int) -> float:
    """Diminishing-returns ratio: reaches 1.0 at `cap` hits, never rewards
    piling on more of the same keyword beyond that (anti keyword-stuffing)."""
    if cap <= 0:
        return 0.0
    return clamp(min(count, cap) / cap)


def _technical_subscore(features: Dict[str, Any]) -> float:
    ai = _saturating_ratio(int(features.get("ai_skill_count", 0)), 4)
    ml = _saturating_ratio(int(features.get("ml_skill_count", 0)), 4)
    retrieval = _saturating_ratio(int(features.get("search_retrieval_skill_count", 0)), 3)
    ranking = _saturating_ratio(int(features.get("ranking_skill_count", 0)), 2)
    backend = _saturating_ratio(int(features.get("backend_skill_count", 0)), 3)
    cloud = _saturating_ratio(int(features.get("cloud_skill_count", 0)), 2)
    infra = _saturating_ratio(int(features.get("infrastructure_skill_count", 0)), 2)
    vector_db = _saturating_ratio(int(features.get("vector_db_skill_count", 0)), 2)
    evaluation = _saturating_ratio(int(features.get("evaluation_skill_count", 0)), 2)
    advanced = _saturating_ratio(int(features.get("advanced_skill_count", 0)), 4)

    # career-evidenced tech usage counts extra: it's proven, not just listed
    career_tech = _saturating_ratio(len(features.get("tech_hits_career", []) or []), 5)

    skills_component = (
        ai * 0.20
        + ml * 0.12
        + retrieval * 0.15
        + ranking * 0.10
        + backend * 0.08
        + cloud * 0.05
        + infra * 0.05
        + vector_db * 0.10
        + evaluation * 0.10
        + advanced * 0.05
    )

    raw = clamp(skills_component) * 0.6 + career_tech * 0.4
    return clamp(raw)


# ---------------------------------------------------------------------------
# 3. Career evidence subscore
# ---------------------------------------------------------------------------

def _career_evidence_subscore(features: Dict[str, Any]) -> float:
    evidence_hits = features.get("career_evidence_hits", []) or []
    evidence_ratio = _saturating_ratio(len(evidence_hits), 6)

    total_ai_roles = int(features.get("total_ai_roles", 0))
    ai_role_ratio = _saturating_ratio(total_ai_roles, 3)

    stability = clamp(float(features.get("career_stability_score", 0.0)))
    promotion = clamp(float(features.get("promotion_score", 0.0)))

    raw = evidence_ratio * 0.55 + ai_role_ratio * 0.20 + stability * 0.15 + promotion * 0.10
    return clamp(raw)


# ---------------------------------------------------------------------------
# 4. Company fit subscore
# ---------------------------------------------------------------------------

def _company_fit_subscore(features: Dict[str, Any]) -> float:
    positive_flags = [
        bool(features.get("is_startup", False)),
        bool(features.get("is_ai_company", False)),
        bool(features.get("is_saas_company", False)),
        bool(features.get("is_software_company", False)),
        bool(features.get("is_marketplace_company", False)),
        bool(features.get("is_product_company", False)),
    ]
    positive_hit_ratio = _saturating_ratio(sum(positive_flags), 2)

    product_roles = int(features.get("total_product_roles", 0))
    product_role_ratio = _saturating_ratio(product_roles, 3)

    consulting_hits = features.get("consulting_hits", []) or []
    strong_ml_evidence = (
        len(features.get("career_evidence_hits", []) or []) >= 2
        and len(features.get("tech_hits_career", []) or []) >= 2
    )

    base = clamp(positive_hit_ratio * 0.6 + product_role_ratio * 0.4)
    if not positive_flags[0] and not any(positive_flags[1:]) and product_roles == 0:
        base = 0.5  # neutral / unknown company type, not penalized

    if consulting_hits:
        # soft penalty only, never a hard reject; offset if strong ML evidence exists
        base = base * (0.55 if not strong_ml_evidence else 0.8)

    return clamp(base)


# ---------------------------------------------------------------------------
# 5. Location subscore
# ---------------------------------------------------------------------------

def _location_subscore(features: Dict[str, Any]) -> float:
    if features.get("in_priority_location", False):
        return 1.0
    if features.get("in_india", False):
        return 0.85
    if features.get("willing_to_relocate", False):
        return 0.6
    return 0.25


# ---------------------------------------------------------------------------
# 6. Behavior subscore
# ---------------------------------------------------------------------------

def _behavior_subscore(features: Dict[str, Any]) -> float:
    parts = []

    parts.append(clamp(float(features.get("profile_completeness_score", 0.0))))
    parts.append(1.0 if features.get("open_to_work", False) else 0.3)

    last_active_days = features.get("last_active_days", None)
    if last_active_days is None:
        parts.append(0.3)
    else:
        parts.append(clamp(1.0 - float(last_active_days) / 90.0, lo=0.0, hi=1.0))

    parts.append(clamp(float(features.get("recruiter_response_rate", 0.0))))

    resp_hours = float(features.get("avg_response_time_hours", 999.0))
    parts.append(clamp(1.0 - resp_hours / 72.0, lo=0.0, hi=1.0))

    notice = float(features.get("notice_period_days", 999.0))
    if notice <= 30:
        parts.append(1.0)
    else:
        parts.append(clamp(1.0 - (notice - 30) / 90.0, lo=0.0, hi=1.0))

    parts.append(clamp(float(features.get("github_activity_score", 0.0))))
    parts.append(clamp(float(features.get("saved_by_recruiters_30d", 0.0)) / 10.0))
    parts.append(clamp(float(features.get("interview_completion_rate", 0.0))))
    parts.append(clamp(float(features.get("offer_acceptance_rate", 0.0))))

    verification_score = sum([
        bool(features.get("verified_email", False)),
        bool(features.get("verified_phone", False)),
        bool(features.get("linkedin_connected", False)),
    ]) / 3.0
    parts.append(verification_score)

    return clamp(sum(parts) / len(parts))


# ---------------------------------------------------------------------------
# Trap penalty
# ---------------------------------------------------------------------------

def _trap_penalty(features: Dict[str, Any]) -> float:
    num_flags = len(features.get("trap_flags", []) or [])
    return clamp(num_flags * TRAP_PENALTY_PER_FLAG, lo=0.0, hi=MAX_TRAP_PENALTY)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_candidate(features: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Compute the final ranking score and its component subscores.

    Args:
        features: feature dictionary produced by feature_extractor.extract_features

    Returns:
        (final_score, subscores) where final_score is a float in [0, 1] and
        subscores is a dict of the six weighted component scores, each also
        in [0, 1].
    """
    subscores: Dict[str, float] = {
        "experience": _experience_subscore(features),
        "technical": _technical_subscore(features),
        "career_evidence": _career_evidence_subscore(features),
        "company_fit": _company_fit_subscore(features),
        "location": _location_subscore(features),
        "behavior": _behavior_subscore(features),
    }

    base_score = sum(subscores[key] * WEIGHTS[key] for key in WEIGHTS)

    trap_penalty = _trap_penalty(features)
    final_score = clamp(base_score * (1.0 - trap_penalty))

    return final_score, subscores