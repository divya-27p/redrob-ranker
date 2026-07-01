"""Extracts high-quality recruiter features from a single candidate record.

Pure string/number processing - no network calls, no ML models. Every
field access is defensive so a missing/malformed field never crashes the
pipeline; safe defaults are always returned. The returned dict is a
superset of what scorer.py and reasoner.py consume, so it stays a drop-in
replacement for the original extractor.

Performance notes (this revision):
    - Keyword lists that used to trigger *many* separate find_matches()
      calls against the *same* text (skill categories, company-type
      hints, location variants, education-degree hints) are merged into a
      single combined keyword list per text blob, scanned once, with a
      keyword -> category lookup table used to fan the single match list
      back out into per-category counts/flags. This turns e.g. 10 skill
      scans into 1, 7 location scans into 1, and 6 company scans into 1,
      without changing any computed value.
    - The unused `full_text` blob (never referenced downstream) has been
      removed - it cost a normalize_text() call per candidate for nothing.
    - `career_contradicts_summary` trap detection now computes
      find_matches(summary, TECH_KEYWORDS) once instead of twice.
    - The `{k.lower() for k in TECH_KEYWORDS}` set used for the
      high-endorsement trap check was being rebuilt on every single
      candidate; it is now a module-level constant.
    - Per-role scanning merges the product/software/saas/marketplace +
      consulting company-type check into a single find_matches() call per
      role instead of two-to-three.
    - All keyword-category lookup tables are built once at import time,
      not per candidate.
"""

from typing import Any, Dict, List, Tuple

from .utils import (
    as_bool,
    as_float,
    clamp,
    days_since,
    find_matches,
    normalize_text,
    parse_date,
    safe_get,
)

# ---------------------------------------------------------------------------
# Keyword / reference lists
# ---------------------------------------------------------------------------

CAREER_EVIDENCE_KEYWORDS = [
    "production", "deployed", "shipped", "real users", "scale",
    "retrieval", "ranking", "recommendation", "search", "embeddings",
    "vector search", "evaluation", "a/b testing", "ab testing",
    "experimentation", "feature store", "ml pipeline", "mlops",
    "llm", "fine tuning", "fine-tuning", "rag", "bm25", "faiss",
    "elasticsearch", "opensearch", "milvus", "pinecone", "weaviate",
    "qdrant",
]

TECH_KEYWORDS = [
    "python", "embeddings", "vector search", "retrieval", "ranking",
    "recommendation system", "recommendation", "search", "nlp",
    "ml systems", "machine learning systems", "mlops", "evaluation",
    "a/b testing", "ab testing", "bm25", "faiss", "elasticsearch",
    "opensearch", "qdrant", "milvus", "pinecone", "weaviate", "llm",
    "fine-tuning", "finetuning", "lora", "rag",
]

TECH_KEYWORDS_LOWER_SET = frozenset(k.lower() for k in TECH_KEYWORDS)

AI_SKILLS = [
    "machine learning", "deep learning", "nlp", "computer vision",
    "llm", "generative ai", "transformers", "fine-tuning", "lora",
    "rag", "prompt engineering", "reinforcement learning",
]

ML_SKILLS = [
    "pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm",
    "keras", "ml pipeline", "mlops", "feature engineering",
    "model training", "model deployment",
]

SEARCH_RETRIEVAL_SKILLS = [
    "search", "retrieval", "ranking", "recommendation", "bm25",
    "elasticsearch", "opensearch", "information retrieval",
]

BACKEND_SKILLS = [
    "java", "go", "golang", "node.js", "django", "flask", "fastapi",
    "spring boot", "microservices", "rest api", "grpc",
]

DATA_ENGINEERING_SKILLS = [
    "spark", "airflow", "kafka", "etl", "hadoop", "data pipeline",
    "dbt", "snowflake", "data warehouse",
]

CLOUD_SKILLS = ["aws", "gcp", "azure", "cloud", "s3", "ec2", "lambda"]

INFRASTRUCTURE_SKILLS = [
    "docker", "kubernetes", "terraform", "ci/cd", "infrastructure",
    "devops",
]

VECTOR_DB_SKILLS = ["faiss", "milvus", "pinecone", "weaviate", "qdrant"]

RANKING_SKILLS = ["ranking", "learning to rank", "recommendation", "search relevance"]

EVALUATION_SKILLS = ["evaluation", "a/b testing", "ab testing", "experimentation", "offline evaluation"]

ADVANCED_SKILL_MARKERS = ["expert", "advanced", "senior"]

CONSULTING_COMPANIES = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree",
]

STARTUP_HINTS = ["startup", "early stage", "seed stage", "series a", "series b"]
PRODUCT_COMPANY_HINTS = ["product"]
SOFTWARE_COMPANY_HINTS = ["software"]
AI_COMPANY_HINTS = ["ai", "artificial intelligence", "machine learning"]
SAAS_COMPANY_HINTS = ["saas"]
MARKETPLACE_COMPANY_HINTS = ["marketplace"]

PRIORITY_LOCATIONS = {
    "pune": ["pune"],
    "noida": ["noida"],
    "delhi_ncr": ["delhi", "ncr", "new delhi"],
    "hyderabad": ["hyderabad"],
    "mumbai": ["mumbai"],
    "bangalore": ["bangalore", "bengaluru"],
    "gurgaon": ["gurgaon", "gurugram"],
}

NON_TECH_TITLES = [
    "marketing manager", "hr manager", "human resources manager",
    "accountant", "operations manager", "customer support",
    "civil engineer", "mechanical engineer", "sales manager",
    "recruiter", "office manager",
]

DEGREE_TIERS = {
    "phd": 4,
    "doctorate": 4,
    "masters": 3,
    "master": 3,
    "m.tech": 3,
    "m.s.": 3,
    "ms": 3,
    "mba": 3,
    "bachelors": 2,
    "bachelor": 2,
    "b.tech": 2,
    "b.e.": 2,
    "be": 2,
    "bs": 2,
    "diploma": 1,
}

AI_DEGREE_KEYWORDS = [
    "artificial intelligence", "machine learning", "data science",
    "computer vision", "nlp",
]

CS_DEGREE_KEYWORDS = [
    "computer science", "computer engineering", "software engineering",
    "information technology", "cse", "it",
]

SENIORITY_LADDER = ["intern", "associate", "engineer", "senior", "staff", "principal", "lead", "director", "vp", "head"]


# ---------------------------------------------------------------------------
# Precomputed keyword -> category(ies) lookup tables (built once at import
# time). Each merges what used to be N separate find_matches() calls
# against the *same* text into a single scan.
# ---------------------------------------------------------------------------

def _build_category_index(category_to_keywords: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    """Given {category_name: [keywords...]}, return (all_unique_keywords,
    keyword -> [category_names]) so a single find_matches() pass can be
    fanned back out into per-category results."""
    keyword_to_categories: Dict[str, List[str]] = {}
    for category, keywords in category_to_keywords.items():
        for kw in keywords:
            keyword_to_categories.setdefault(kw, []).append(category)
    return list(keyword_to_categories.keys()), keyword_to_categories


_SKILL_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "ai": AI_SKILLS,
    "ml": ML_SKILLS,
    "search_retrieval": SEARCH_RETRIEVAL_SKILLS,
    "backend": BACKEND_SKILLS,
    "data_engineering": DATA_ENGINEERING_SKILLS,
    "cloud": CLOUD_SKILLS,
    "infrastructure": INFRASTRUCTURE_SKILLS,
    "vector_db": VECTOR_DB_SKILLS,
    "ranking": RANKING_SKILLS,
    "evaluation": EVALUATION_SKILLS,
}
_ALL_SKILL_KEYWORDS, _SKILL_KEYWORD_TO_CATEGORIES = _build_category_index(_SKILL_CATEGORY_KEYWORDS)

_COMPANY_HINT_CATEGORIES: Dict[str, List[str]] = {
    "startup": STARTUP_HINTS,
    "product": PRODUCT_COMPANY_HINTS,
    "software": SOFTWARE_COMPANY_HINTS,
    "ai": AI_COMPANY_HINTS,
    "saas": SAAS_COMPANY_HINTS,
    "marketplace": MARKETPLACE_COMPANY_HINTS,
    "consulting": CONSULTING_COMPANIES,
}
_ALL_COMPANY_KEYWORDS, _COMPANY_KEYWORD_TO_CATEGORIES = _build_category_index(_COMPANY_HINT_CATEGORIES)

_LOCATION_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    key: variants for key, variants in PRIORITY_LOCATIONS.items()
}
_ALL_LOCATION_KEYWORDS, _LOCATION_KEYWORD_TO_CATEGORIES = _build_category_index(_LOCATION_CATEGORY_KEYWORDS)

_DEGREE_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "ai_degree": AI_DEGREE_KEYWORDS,
    "cs_degree": CS_DEGREE_KEYWORDS,
}
_ALL_DEGREE_KEYWORDS, _DEGREE_KEYWORD_TO_CATEGORIES = _build_category_index(_DEGREE_CATEGORY_KEYWORDS)

_ROLE_TEXT_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "product_type": PRODUCT_COMPANY_HINTS + SOFTWARE_COMPANY_HINTS + SAAS_COMPANY_HINTS + MARKETPLACE_COMPANY_HINTS,
    "consulting": CONSULTING_COMPANIES,
    "ai_company": AI_COMPANY_HINTS,
}
_ALL_ROLE_TEXT_KEYWORDS, _ROLE_TEXT_KEYWORD_TO_CATEGORIES = _build_category_index(_ROLE_TEXT_CATEGORY_KEYWORDS)


def _matched_categories(text: str, all_keywords: List[str], keyword_to_categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Run one find_matches() pass over `text` against the merged keyword
    list, then fan the results back out per category. Returns
    {category_name: [matched_keyword, ...]} for every category present in
    `keyword_to_categories` (empty list if nothing matched)."""
    result: Dict[str, List[str]] = {}
    for kw in find_matches(text, all_keywords):
        for category in keyword_to_categories[kw]:
            result.setdefault(category, []).append(kw)
    return result


def _seniority_rank(title: str) -> int:
    title_l = title.lower()
    rank = 0
    for i, word in enumerate(SENIORITY_LADDER):
        if word in title_l:
            rank = max(rank, i)
    return rank


def _career_stability_score(career_history: List[Dict[str, Any]]) -> float:
    if not career_history:
        return 0.0
    durations = [as_float(safe_get(ch, "duration_months", 0.0)) for ch in career_history]
    durations = [d for d in durations if d > 0]
    if not durations:
        return 0.0
    avg_duration = sum(durations) / len(durations)
    short_stints = sum(1 for d in durations if d < 9)
    stability = clamp(avg_duration / 36.0) - clamp(short_stints / max(len(durations), 1)) * 0.3
    return clamp(stability)


def _promotion_score(career_history: List[Dict[str, Any]]) -> float:
    if not career_history or len(career_history) < 2:
        return 0.0
    ranks = [_seniority_rank(str(safe_get(ch, "title", ""))) for ch in career_history]
    if not ranks or all(r == 0 for r in ranks):
        return 0.0
    increases = sum(1 for i in range(1, len(ranks)) if ranks[i - 1] > ranks[i])
    # career_history is typically most-recent-first; an increase walking
    # backward in time (older -> newer seniority growth) is a promotion signal
    return clamp(increases / max(len(ranks) - 1, 1))


def _highest_degree_and_tier(education: List[Dict[str, Any]]):
    highest_degree = ""
    highest_tier = 0
    for edu in education:
        if not isinstance(edu, dict):
            continue
        degree = str(safe_get(edu, "degree", "") or "")
        degree_l = degree.lower()
        for key, tier in DEGREE_TIERS.items():
            if tier > highest_tier and key in degree_l:
                highest_tier = tier
                highest_degree = degree
    return highest_degree, highest_tier


def extract_features(candidate: Dict[str, Any]) -> Dict[str, Any]:
    profile = candidate.get("profile") or {}
    career_history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []
    signals = candidate.get("redrob_signals") or {}
    education = candidate.get("education") or []

    if not isinstance(profile, dict):
        profile = {}
    if not isinstance(career_history, list):
        career_history = []
    if not isinstance(skills, list):
        skills = []
    if not isinstance(signals, dict):
        signals = {}
    if not isinstance(education, list):
        education = []

    career_history = [ch for ch in career_history if isinstance(ch, dict)]
    skills = [s for s in skills if isinstance(s, dict)]

    # =====================================================================
    # 1. BASIC PROFILE
    # =====================================================================
    candidate_id = safe_get(profile, "candidate_id", safe_get(candidate, "candidate_id", "UNKNOWN"))
    headline = str(safe_get(profile, "headline", "") or "")
    summary = str(safe_get(profile, "summary", "") or "")
    current_title = str(safe_get(profile, "current_title", "") or "")
    current_company = str(safe_get(profile, "current_company", "") or "")
    current_industry = str(safe_get(profile, "current_industry", "") or "")
    country = str(safe_get(profile, "country", "") or "")
    location = str(safe_get(profile, "location", "") or "")
    years_experience = as_float(safe_get(profile, "years_of_experience", 0.0))

    # =====================================================================
    # text blobs for keyword scanning (built once, scanned once per topic)
    # =====================================================================
    skill_names = [str(safe_get(s, "name", "") or "") for s in skills]
    career_descriptions = [str(safe_get(ch, "description", "") or "") for ch in career_history]
    career_titles = [str(safe_get(ch, "title", "") or "") for ch in career_history]
    career_companies_text_list = [str(safe_get(ch, "company", "") or "") for ch in career_history]
    career_industries_text_list = [str(safe_get(ch, "industry", "") or "") for ch in career_history]

    profile_text = normalize_text(headline, summary, current_title, *skill_names)
    career_text = normalize_text(*career_descriptions, *career_titles)

    tech_hits_profile = set(find_matches(profile_text, TECH_KEYWORDS))
    tech_hits_career = set(find_matches(career_text, TECH_KEYWORDS))
    tech_hits_all = tech_hits_profile | tech_hits_career

    career_evidence_hits = set(find_matches(career_text, CAREER_EVIDENCE_KEYWORDS))
    career_evidence_count = len(career_evidence_hits)

    # =====================================================================
    # 2. CAREER FEATURES
    # =====================================================================
    total_roles = len(career_history)

    current_role_duration = (
        as_float(safe_get(career_history[0], "duration_months", 0.0)) if career_history else 0.0
    )

    total_product_roles = 0
    total_consulting_roles = 0
    total_ai_roles = 0
    for ch in career_history:
        role_text = normalize_text(
            str(safe_get(ch, "company", "") or ""),
            str(safe_get(ch, "industry", "") or ""),
        )
        role_matches = _matched_categories(role_text, _ALL_ROLE_TEXT_KEYWORDS, _ROLE_TEXT_KEYWORD_TO_CATEGORIES)

        if role_matches.get("product_type"):
            total_product_roles += 1
        if role_matches.get("consulting"):
            total_consulting_roles += 1

        if role_matches.get("ai_company"):
            total_ai_roles += 1
        else:
            role_desc_text = normalize_text(str(safe_get(ch, "description", "") or ""))
            if find_matches(role_desc_text, CAREER_EVIDENCE_KEYWORDS):
                total_ai_roles += 1

    career_stability_score = _career_stability_score(career_history)
    promotion_score = _promotion_score(career_history)

    # =====================================================================
    # 3. SKILL FEATURES (single merged scan across all skill categories)
    # =====================================================================
    skill_text = normalize_text(*skill_names)
    skill_category_matches = _matched_categories(skill_text, _ALL_SKILL_KEYWORDS, _SKILL_KEYWORD_TO_CATEGORIES)

    ai_skill_count = len(skill_category_matches.get("ai", ()))
    ml_skill_count = len(skill_category_matches.get("ml", ()))
    search_retrieval_skill_count = len(skill_category_matches.get("search_retrieval", ()))
    backend_skill_count = len(skill_category_matches.get("backend", ()))
    data_engineering_skill_count = len(skill_category_matches.get("data_engineering", ()))
    cloud_skill_count = len(skill_category_matches.get("cloud", ()))
    infrastructure_skill_count = len(skill_category_matches.get("infrastructure", ()))
    vector_db_skill_count = len(skill_category_matches.get("vector_db", ()))
    ranking_skill_count = len(skill_category_matches.get("ranking", ()))
    evaluation_skill_count = len(skill_category_matches.get("evaluation", ()))

    skill_count = len(skills)
    advanced_skill_count = sum(
        1 for s in skills
        if str(safe_get(s, "proficiency", "") or "").lower() in ADVANCED_SKILL_MARKERS
    )
    ai_keyword_count = len(tech_hits_all)

    # =====================================================================
    # 4. EDUCATION FEATURES (single merged scan)
    # =====================================================================
    highest_degree, highest_tier = _highest_degree_and_tier(education)
    education_text = normalize_text(
        *[str(safe_get(e, "degree", "") or "") for e in education if isinstance(e, dict)],
        *[str(safe_get(e, "field", "") or "") for e in education if isinstance(e, dict)],
    )
    degree_matches = _matched_categories(education_text, _ALL_DEGREE_KEYWORDS, _DEGREE_KEYWORD_TO_CATEGORIES)
    ai_related_degree = bool(degree_matches.get("ai_degree"))
    cs_related_degree = bool(degree_matches.get("cs_degree"))

    # =====================================================================
    # 5. BEHAVIOR FEATURES
    # =====================================================================
    profile_completeness_score = as_float(safe_get(signals, "profile_completeness_score", 0.0))
    open_to_work = as_bool(safe_get(signals, "open_to_work_flag", False))
    last_active_dt = parse_date(safe_get(signals, "last_active_date", None))
    last_active_days = days_since(last_active_dt)
    recruiter_response_rate = as_float(safe_get(signals, "recruiter_response_rate", 0.0))
    avg_response_time_hours = as_float(safe_get(signals, "avg_response_time_hours", 999.0))
    notice_period_days = as_float(safe_get(signals, "notice_period_days", 999.0))
    github_activity_score = as_float(safe_get(signals, "github_activity_score", 0.0))
    saved_by_recruiters_30d = as_float(safe_get(signals, "saved_by_recruiters_30d", 0.0))
    interview_completion_rate = as_float(safe_get(signals, "interview_completion_rate", 0.0))
    offer_acceptance_rate = as_float(safe_get(signals, "offer_acceptance_rate", 0.0))
    verified_email = as_bool(safe_get(signals, "verified_email", False))
    verified_phone = as_bool(safe_get(signals, "verified_phone", False))
    linkedin_connected = as_bool(safe_get(signals, "linkedin_connected", False))
    willing_to_relocate = as_bool(safe_get(signals, "willing_to_relocate", False))

    # =====================================================================
    # 6. LOCATION FEATURES (single merged scan across all priority cities)
    # =====================================================================
    loc_text = normalize_text(location, country)
    location_matches = _matched_categories(loc_text, _ALL_LOCATION_KEYWORDS, _LOCATION_KEYWORD_TO_CATEGORIES)

    location_flags: Dict[str, bool] = {}
    in_priority_location = False
    for key in PRIORITY_LOCATIONS:
        hit = bool(location_matches.get(key))
        location_flags[f"in_{key}"] = hit
        in_priority_location = in_priority_location or hit
    in_india = "india" in loc_text or in_priority_location

    # =====================================================================
    # 7. COMPANY FEATURES (single merged scan)
    # =====================================================================
    all_companies_text = normalize_text(
        current_company, current_industry,
        *career_companies_text_list,
        *career_industries_text_list,
    )
    company_matches = _matched_categories(all_companies_text, _ALL_COMPANY_KEYWORDS, _COMPANY_KEYWORD_TO_CATEGORIES)

    consulting_hits = sorted(set(company_matches.get("consulting", ())))
    is_startup = bool(company_matches.get("startup"))
    is_product_company = bool(company_matches.get("product"))
    is_software_company = bool(company_matches.get("software"))
    is_ai_company = bool(company_matches.get("ai"))
    is_saas_company = bool(company_matches.get("saas"))
    is_marketplace_company = bool(company_matches.get("marketplace"))
    product_hits = sorted({
        hint for hint, flag in (
            ("startup", is_startup),
            ("product", is_product_company),
            ("software", is_software_company),
            ("ai", is_ai_company),
            ("saas", is_saas_company),
            ("marketplace", is_marketplace_company),
        ) if flag
    })

    # =====================================================================
    # 8. TRAP DETECTION
    # =====================================================================
    trap_flags: List[str] = []

    high_endorsement_ai_skills = [
        s for s in skills
        if str(safe_get(s, "name", "") or "").lower() in TECH_KEYWORDS_LOWER_SET
        and as_float(safe_get(s, "endorsements", 0)) >= 20
    ]

    if len(tech_hits_profile) >= 4 and career_evidence_count == 0 and len(tech_hits_career) == 0:
        trap_flags.append("ai_keywords_without_career_evidence")

    if len(high_endorsement_ai_skills) >= 3 and career_evidence_count == 0:
        trap_flags.append("high_endorsements_weak_career_evidence")

    if find_matches(current_title.lower(), NON_TECH_TITLES):
        trap_flags.append("non_technical_current_title")

    expected_salary_min = safe_get(profile, "expected_salary_min", None)
    expected_salary_max = safe_get(profile, "expected_salary_max", None)
    if expected_salary_min is not None and expected_salary_max is not None:
        try:
            if as_float(expected_salary_min) > as_float(expected_salary_max):
                trap_flags.append("expected_salary_min_greater_than_max")
        except Exception:
            pass

    if years_experience < 0 or years_experience > 50:
        trap_flags.append("impossible_years_of_experience")

    if notice_period_days > 90:
        trap_flags.append("long_notice_period")

    if last_active_days is not None and last_active_days > 180:
        trap_flags.append("inactive_profile")

    if not open_to_work and (recruiter_response_rate < 0.2 or last_active_days is None or last_active_days > 90):
        trap_flags.append("not_open_to_work_and_low_engagement")

    if "ai" in profile_text and len(tech_hits_career) == 0 and career_evidence_count == 0:
        trap_flags.append("ai_claim_unrelated_career")

    summary_tech_hits = find_matches(summary.lower(), TECH_KEYWORDS) if summary else []
    if summary and career_text and not find_matches(career_text, summary_tech_hits or ["__none__"]):
        if summary_tech_hits and not tech_hits_career and not career_evidence_hits:
            trap_flags.append("career_contradicts_summary")

    # =====================================================================
    # FINAL FEATURE DICTIONARY (superset, safe for scorer.py / reasoner.py)
    # =====================================================================
    features: Dict[str, Any] = {
        # basic profile
        "candidate_id": candidate_id,
        "headline": headline,
        "summary": summary,
        "current_title": current_title,
        "current_company": current_company,
        "current_industry": current_industry,
        "country": country,
        "location": location,
        "years_experience": years_experience,

        # career features
        "total_roles": total_roles,
        "current_role_duration": current_role_duration,
        "total_product_roles": total_product_roles,
        "total_consulting_roles": total_consulting_roles,
        "total_ai_roles": total_ai_roles,
        "career_stability_score": career_stability_score,
        "promotion_score": promotion_score,
        "career_company_count": total_roles,
        "top_company_for_reasoning": (
            safe_get(career_history[0], "company", current_company) if career_history else current_company
        ),

        # keyword / evidence hits (consumed by scorer.py / reasoner.py)
        "tech_hits_all": sorted(tech_hits_all),
        "tech_hits_career": sorted(tech_hits_career),
        "career_evidence_hits": sorted(career_evidence_hits),

        # skill features
        "skill_names": skill_names,
        "skill_count": skill_count,
        "advanced_skill_count": advanced_skill_count,
        "ai_keyword_count": ai_keyword_count,
        "ai_skill_count": ai_skill_count,
        "ml_skill_count": ml_skill_count,
        "search_retrieval_skill_count": search_retrieval_skill_count,
        "backend_skill_count": backend_skill_count,
        "data_engineering_skill_count": data_engineering_skill_count,
        "cloud_skill_count": cloud_skill_count,
        "infrastructure_skill_count": infrastructure_skill_count,
        "vector_db_skill_count": vector_db_skill_count,
        "ranking_skill_count": ranking_skill_count,
        "evaluation_skill_count": evaluation_skill_count,

        # education features
        "highest_degree": highest_degree,
        "highest_tier": highest_tier,
        "ai_related_degree": ai_related_degree,
        "cs_related_degree": cs_related_degree,

        # behavior features
        "profile_completeness_score": profile_completeness_score,
        "open_to_work": open_to_work,
        "last_active_days": last_active_days,
        "recruiter_response_rate": recruiter_response_rate,
        "avg_response_time_hours": avg_response_time_hours,
        "notice_period_days": notice_period_days,
        "github_activity_score": github_activity_score,
        "saved_by_recruiters_30d": saved_by_recruiters_30d,
        "interview_completion_rate": interview_completion_rate,
        "offer_acceptance_rate": offer_acceptance_rate,
        "verified_email": verified_email,
        "verified_phone": verified_phone,
        "linkedin_connected": linkedin_connected,
        "willing_to_relocate": willing_to_relocate,

        # location features
        "in_india": in_india,
        "in_priority_location": in_priority_location,

        # company features
        "consulting_hits": consulting_hits,
        "product_hits": product_hits,
        "is_startup": is_startup,
        "is_product_company": is_product_company,
        "is_software_company": is_software_company,
        "is_ai_company": is_ai_company,
        "is_saas_company": is_saas_company,
        "is_marketplace_company": is_marketplace_company,

        # trap detection
        "trap_flags": trap_flags,
    }

    features.update(location_flags)

    return features