"""ClinicalTrials.gov client (API v2).

Docs: https://clinicaltrials.gov/data-api/api
"""

from __future__ import annotations

from typing import Any

import httpx

from biomed_agent.schemas import SourceDocument
from biomed_agent.sources.http import get_with_retry

API = "https://clinicaltrials.gov/api/v2/studies"

_FIELDS = ",".join(
    [
        "NCTId",
        "BriefTitle",
        "OfficialTitle",
        "OverallStatus",
        "Phase",
        "StudyType",
        "BriefSummary",
        "Condition",
        "InterventionName",
        "EnrollmentCount",
        "StartDate",
        "PrimaryCompletionDate",
        "PrimaryOutcomeMeasure",
        "LeadSponsorName",
        "HasResults",
    ]
)


class ClinicalTrialsClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    async def search(self, query: str, max_results: int = 8) -> list[SourceDocument]:
        resp = await get_with_retry(
            self._http,
            API,
            params={
                "query.term": query,
                "pageSize": max_results,
                "fields": _FIELDS,
                "format": "json",
            },
        )
        return [doc for s in resp.json().get("studies", []) if (doc := study_to_document(s))]


def _get(d: dict[str, Any], *path: str, default: Any = None) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def study_to_document(study: dict[str, Any]) -> SourceDocument | None:
    p = study.get("protocolSection", {})
    nct = _get(p, "identificationModule", "nctId")
    if not nct:
        return None

    title = _get(p, "identificationModule", "briefTitle", default="")
    status = _get(p, "statusModule", "overallStatus", default="UNKNOWN")
    start = _get(p, "statusModule", "startDateStruct", "date", default="")
    phases = _get(p, "designModule", "phases", default=[]) or []
    study_type = _get(p, "designModule", "studyType", default="")
    enrollment = _get(p, "designModule", "enrollmentInfo", "count")
    conditions = _get(p, "conditionsModule", "conditions", default=[]) or []
    interventions = [
        i.get("name", "") for i in _get(p, "armsInterventionsModule", "interventions", default=[])
    ]
    outcomes = [
        o.get("measure", "") for o in _get(p, "outcomesModule", "primaryOutcomes", default=[])
    ]
    sponsor = _get(p, "sponsorCollaboratorsModule", "leadSponsor", "name")
    summary = _get(p, "descriptionModule", "briefSummary", default="")

    # Flatten the structured record into text so it can be chunked and embedded
    # the same way as an abstract.
    lines = [
        f"Title: {title}",
        f"Status: {status}",
        f"Study type: {study_type}" if study_type else "",
        f"Phase: {', '.join(phases)}" if phases else "",
        f"Enrollment: {enrollment}" if enrollment is not None else "",
        f"Conditions: {', '.join(conditions)}" if conditions else "",
        f"Interventions: {', '.join(i for i in interventions if i)}" if interventions else "",
        f"Primary outcomes: {'; '.join(o for o in outcomes[:5] if o)}" if outcomes else "",
        f"Summary: {summary}" if summary else "",
    ]

    return SourceDocument(
        source_id=nct,
        source_type="trial",
        title=title,
        text="\n".join(line for line in lines if line),
        url=f"https://clinicaltrials.gov/study/{nct}",
        year=int(start[:4]) if start[:4].isdigit() else None,
        venue=sponsor,
        metadata={
            k: v
            for k, v in {
                "status": status,
                "phase": ", ".join(phases),
                "has_results": str(study.get("hasResults", "")).lower(),
            }.items()
            if v
        },
    )
