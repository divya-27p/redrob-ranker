"""Generates factual, rank-aware, 1-2 sentence reasoning for a ranked
candidate.

Every fact mentioned is read directly from the feature dictionary produced
by feature_extractor.py - nothing is invented. Output is plain text
(no embedded newlines) so it is always safe to write into a CSV cell.
"""

from typing import Any, Dict, List, Optional

MAX_LENGTH = 350

TRAP_FLAG_LABELS: Dict[str, str] = {
    "ai_keywords_without_career_evidence": "an AI-heavy skills list without matching career evidence",
    "high_endorsements_weak_career_evidence": "heavily endorsed AI skills but weak supporting career history",
    "non_technical_current_title": "a non-technical current title",
    "expected_salary_min_greater_than_max": "an inconsistent expected salary range",
    "impossible_years_of_experience": "an inconsistent years-of-experience value",
    "long_notice_period": "a long notice period",
    "inactive_profile": "an inactive profile",
    "not_open_to_work_and_low_engagement": "low engagement and not flagged as open to work",
    "ai_claim_unrelated_career": "profile mentions AI but career history doesn't back it up",
    "career_contradicts_summary": "career history doesn't fully match the stated summary",
}


def _fmt_years(years: float) -> str:
    return f"{years:g}"


def _strength_experience(features: Dict[str, Any]) -> Optional[str]:
    years = float(features.get("years_experience", 0.0) or 0.0)
    if 5 <= years <= 9:
        return f"{_fmt_years(years)} years of experience in the ideal range"
    if 4 <= years <= 10:
        return f"{_fmt_years(years)} years of experience close to the ideal range"
    return None


def _strength_role(features: Dict[str, Any]) -> Optional[str]:
    title = str(features.get("current_title", "") or "").strip()
    company = str(features.get("current_company", "") or "").strip()
    if title and company:
        return f"currently {title} at {company}"
    if title:
        return f"currently {title}"
    if company:
        return f"currently at {company}"
    return None


def _strength_technical(features: Dict[str, Any]) -> Optional[str]:
    tech_career = features.get("tech_hits_career", []) or []
    if tech_career:
        shown = ", ".join(tech_career[:3])
        return f"career-evidenced skills in {shown}"
    tech_all = features.get("tech_hits_all", []) or []
    if tech_all:
        shown = ", ".join(tech_all[:3])
        return f"listed skills in {shown}"
    return None


def _strength_career_evidence(features: Dict[str, Any]) -> Optional[str]:
    hits = features.get("career_evidence_hits", []) or []
    if hits:
        shown = ", ".join(hits[:2])
        return f"career history mentioning {shown}"
    return None


def _strength_company_fit(features: Dict[str, Any]) -> Optional[str]:
    product_hits = features.get("product_hits", []) or []
    consulting_hits = features.get("consulting_hits", []) or []
    if product_hits and not consulting_hits:
        return "background at product/software companies"
    return None


def _strength_location(features: Dict[str, Any]) -> Optional[str]:
    location = str(features.get("location", "") or "").strip()
    if features.get("in_priority_location", False) and location:
        return f"based in {location}"
    if features.get("in_india", False) and location:
        return f"based in {location}"
    if features.get("willing_to_relocate", False):
        return "open to relocation"
    return None


def _strength_behavioral(features: Dict[str, Any]) -> Optional[str]:
    last_active_days = features.get("last_active_days", None)
    if (
        features.get("open_to_work", False)
        and last_active_days is not None
        and float(last_active_days) <= 30
    ):
        return "actively engaged and open to work"
    return None


def _collect_strengths(features: Dict[str, Any]) -> List[str]:
    builders = [
        _strength_role,
        _strength_experience,
        _strength_technical,
        _strength_career_evidence,
        _strength_company_fit,
        _strength_location,
        _strength_behavioral,
    ]
    strengths: List[str] = []
    for build in builders:
        value = build(features)
        if value:
            strengths.append(value)
    return strengths


def _concern_experience(features: Dict[str, Any]) -> Optional[str]:
    years = float(features.get("years_experience", 0.0) or 0.0)
    if years < 3:
        return f"only {_fmt_years(years)} years of experience, below the target range"
    if years > 12:
        return f"{_fmt_years(years)} years of experience, above the ideal range"
    return None


def _concern_consulting(features: Dict[str, Any]) -> Optional[str]:
    consulting_hits = features.get("consulting_hits", []) or []
    evidence_hits = features.get("career_evidence_hits", []) or []
    if consulting_hits and len(evidence_hits) < 2:
        return f"a services-firm background ({consulting_hits[0]}) with limited product ML evidence"
    return None


def _concern_weak_evidence(features: Dict[str, Any]) -> Optional[str]:
    tech_career = features.get("tech_hits_career", []) or []
    evidence_hits = features.get("career_evidence_hits", []) or []
    if not tech_career and not evidence_hits:
        return "no clear career-history evidence of relevant ML/search/retrieval work"
    return None


def _concern_location(features: Dict[str, Any]) -> Optional[str]:
    if not features.get("in_india", False) and not features.get("willing_to_relocate", False):
        return "located outside India with no stated relocation flexibility"
    return None


def _concern_notice(features: Dict[str, Any]) -> Optional[str]:
    notice = features.get("notice_period_days", None)
    if notice is not None and float(notice) > 90:
        return f"a long notice period ({float(notice):g} days)"
    return None


def _concern_inactive(features: Dict[str, Any]) -> Optional[str]:
    last_active_days = features.get("last_active_days", None)
    if last_active_days is not None and float(last_active_days) > 90:
        return f"inactive for {float(last_active_days):.0f}+ days"
    return None


def _concern_not_open_to_work(features: Dict[str, Any]) -> Optional[str]:
    if not features.get("open_to_work", False):
        return "not currently flagged as open to work"
    return None


def _concern_non_technical_title(features: Dict[str, Any]) -> Optional[str]:
    if "non_technical_current_title" in (features.get("trap_flags", []) or []):
        return "a non-technical current title"
    return None


def _concern_trap_flags(features: Dict[str, Any]) -> List[str]:
    labels = []
    for flag in features.get("trap_flags", []) or []:
        label = TRAP_FLAG_LABELS.get(flag)
        if label and label not in labels:
            labels.append(label)
    return labels


def _collect_concerns(features: Dict[str, Any]) -> List[str]:
    builders = [
        _concern_experience,
        _concern_consulting,
        _concern_weak_evidence,
        _concern_location,
        _concern_notice,
        _concern_inactive,
        _concern_not_open_to_work,
    ]
    concerns: List[str] = []
    for build in builders:
        value = build(features)
        if value and value not in concerns:
            concerns.append(value)
    for label in _concern_trap_flags(features):
        if label not in concerns:
            concerns.append(label)
    return concerns


def _sanitize(text: str) -> str:
    """Ensure the reasoning is a single CSV-safe line."""
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) > MAX_LENGTH:
        cleaned = cleaned[: MAX_LENGTH - 1].rstrip() + "."
    return cleaned


def _compose(lead: str, primary: List[str], secondary: List[str], primary_conjunction: str) -> str:
    if primary:
        sentence = f"{lead}, {primary_conjunction} {'; '.join(primary[:2])}."
    else:
        sentence = f"{lead}."
    if secondary:
        sentence += f" {secondary[0][0].upper()}{secondary[0][1:]}."
    return sentence


def generate_reasoning(features: Dict[str, Any], rank: int) -> str:
    """Build a 1-2 sentence, fact-only reasoning string for a ranked
    candidate, with tone calibrated to their rank tier."""
    strengths = _collect_strengths(features)
    concerns = _collect_concerns(features)

    title = str(features.get("current_title", "") or "").strip()
    company = str(features.get("current_company", "") or "").strip()
    if title and company:
        subject = f"{title} at {company}"
    elif title:
        subject = title
    elif company:
        subject = f"a role at {company}"
    else:
        subject = "this candidate"

    if rank <= 20:
        lead = f"Strong fit: {subject}"
        body = strengths[:2] if strengths else []
        sentence = _compose(lead, body, concerns, "with")
        if not body and not concerns:
            sentence = f"{lead}, based on overall composite score."
    elif rank >= 80:
        lead = f"Cautious fit: {subject}"
        body = concerns[:2] if concerns else []
        sentence = _compose(lead, body, strengths, "given")
        if not body and not strengths:
            sentence = f"{lead}, limited standout evidence in the available signals."
    else:
        lead = f"{subject}".capitalize() if not title and not company else subject
        primary = strengths[:1]
        secondary = concerns[:1]
        if primary:
            sentence = f"{lead} shows {primary[0]}."
        else:
            sentence = f"{lead} shows reasonable overall alignment with the role."
        if secondary:
            sentence += f" Worth noting: {secondary[0]}."

    return _sanitize(sentence)