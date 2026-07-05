"""
PII redaction for Document Portal using Microsoft Presidio.

Strips the following entity types before any text reaches the LLM:
    PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD,
    IBAN_CODE, IP_ADDRESS, LOCATION (configurable via PII_ENTITIES env var)

Replacement strategy: <ENTITY_TYPE> placeholder tokens so the LLM still
receives coherent sentences (e.g. "Contact <PERSON> at <EMAIL_ADDRESS>").

The redactor is inserted as a single step in the ingestion pipeline
(data_ingestion.py → ChatIngestor._split) and also called on the user's
question in the chat/query endpoint before it hits the RAG chain.

Environment variables:
    PII_ENABLED   — "true" (default) | "false"  — master switch
    PII_ENTITIES  — comma-separated Presidio entity names to redact
                    defaults to the list above

Usage:
    from pii.redactor import get_redactor

    redactor = get_redactor()
    clean_text, count = redactor.redact(raw_text)
    # count == number of PII entities removed (useful for audit log)
"""

from __future__ import annotations

import os
from typing import Optional

from logger import GLOBAL_LOGGER as log

# ---------------------------------------------------------------------------
# Default entity set
# ---------------------------------------------------------------------------

_DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "LOCATION",
]


# ---------------------------------------------------------------------------
# Lazy Presidio import — graceful degradation if package not installed
# ---------------------------------------------------------------------------

def _load_presidio():
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
        from presidio_anonymizer.entities import OperatorConfig  # type: ignore
        return AnalyzerEngine, AnonymizerEngine, OperatorConfig
    except ImportError:
        log.warning(
            "presidio-analyzer / presidio-anonymizer not installed — "
            "PII redaction disabled"
        )
        return None, None, None


# ---------------------------------------------------------------------------
# PIIRedactor
# ---------------------------------------------------------------------------

class PIIRedactor:
    """
    Wraps Presidio AnalyzerEngine + AnonymizerEngine.

    Degrades gracefully: if Presidio is not installed or PII_ENABLED=false,
    redact() is a no-op that returns the original text and count=0.
    """

    def __init__(self, entities: Optional[list[str]] = None):
        self._enabled = os.getenv("PII_ENABLED", "true").lower() == "true"

        if not self._enabled:
            log.info("PII redaction disabled via PII_ENABLED=false")
            self._analyzer = None
            self._anonymizer = None
            self._operator_config = None
            self._entities = []
            return

        # Parse entity list from env or use default
        env_entities = os.getenv("PII_ENTITIES", "")
        if env_entities:
            self._entities = [e.strip() for e in env_entities.split(",") if e.strip()]
        else:
            self._entities = entities or _DEFAULT_ENTITIES

        AnalyzerEngine, AnonymizerEngine, OperatorConfig = _load_presidio()

        if AnalyzerEngine is None:
            self._enabled = False
            self._analyzer = self._anonymizer = self._operator_config = None
            return

        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()

        # Replace each PII token with a readable placeholder e.g. <PERSON>
        self._operator_config = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in self._entities
        }

        log.info(
            "PIIRedactor initialised",
            entities=self._entities,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def redact(self, text: str) -> tuple[str, int]:
        """
        Analyse text for PII and replace all findings with placeholder tokens.

        Returns:
            (clean_text, entity_count)

        If redaction is disabled or Presidio is unavailable, returns
        (original_text, 0) — the pipeline continues unaffected.
        """
        if not self._enabled or not self._analyzer or not text:
            return text, 0

        try:
            results = self._analyzer.analyze(
                text=text,
                entities=self._entities,
                language="en",
            )

            if not results:
                return text, 0

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self._operator_config,
            )

            count = len(results)
            log.info("PII redacted", entity_count=count)
            return anonymized.text, count

        except Exception as exc:
            # Never block the pipeline — log and pass through
            log.error("PII redaction failed — passing original text", error=str(exc))
            return text, 0

    def redact_documents(self, docs: list) -> tuple[list, int]:
        """
        Redact PII from a list of LangChain Document objects (in-place mutation
        of page_content).

        Returns (docs, total_entity_count).
        """
        total = 0
        for doc in docs:
            clean, count = self.redact(doc.page_content)
            doc.page_content = clean
            total += count
        return docs, total


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_redactor: Optional[PIIRedactor] = None


def get_redactor() -> PIIRedactor:
    global _redactor
    if _redactor is None:
        _redactor = PIIRedactor()
    return _redactor
