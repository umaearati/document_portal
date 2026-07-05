"""
AWS Lambda handler for Document Portal.

Wraps the /health and /analyze endpoints as Lambda functions
behind API Gateway. This lets you run lightweight endpoints
serverlessly without keeping ECS running 24/7.

Architecture:
    API Gateway → Lambda → Document Portal logic

Deployed endpoints:
    GET  /health    → health_handler
    POST /analyze   → analyze_handler

Environment variables (set in Lambda console or via Terraform):
    OPENAI_API_KEY
    GOOGLE_API_KEY
    GROQ_API_KEY
    LLM_PROVIDER    (default: openai)
    ENV             (default: production)

Deployment:
    cd infra/lambda
    pip install -r ../../requirements.txt -t package/
    cp handler.py package/
    cd package && zip -r ../lambda_package.zip .
    # Upload lambda_package.zip in AWS Lambda console

Or use the GitHub Actions workflow at .github/workflows/lambda.yaml
"""

from __future__ import annotations

import json
import os
import sys
import base64
import tempfile

# Make sure project root is on the path when running in Lambda
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _response(status_code: int, body: dict) -> dict:
    """Standard API Gateway response format."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


# ── /health ──────────────────────────────────────────────────────────────────

def health_handler(event: dict, context) -> dict:
    """
    GET /health
    Simple liveness check — no dependencies, instant response.
    """
    return _response(200, {"status": "ok", "service": "document-portal-lambda"})


# ── /analyze ─────────────────────────────────────────────────────────────────

def analyze_handler(event: dict, context) -> dict:
    """
    POST /analyze
    Accepts a base64-encoded PDF in the request body and returns
    extracted metadata + summary.

    Request body (JSON):
        {
            "file_content": "<base64-encoded PDF bytes>",
            "filename": "contract.pdf"
        }
    """
    try:
        body = json.loads(event.get("body") or "{}")
        file_content_b64 = body.get("file_content")
        filename = body.get("filename", "upload.pdf")

        if not file_content_b64:
            return _response(400, {"error": "file_content (base64) is required"})

        # Decode and write to a temp file
        file_bytes = base64.b64decode(file_content_b64)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # Reuse existing Document Portal logic
        from src.document_ingestion.data_ingestion import DocHandler
        from src.document_analyzer.data_analysis import DocumentAnalyzer

        dh = DocHandler(data_dir=tempfile.gettempdir())

        # Read the PDF directly from the temp path
        text = dh.read_pdf(tmp_path)

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)

        return _response(200, result)

    except Exception as exc:
        return _response(500, {"error": str(exc)})


# ── Router — single Lambda, multiple routes ───────────────────────────────────

def lambda_handler(event: dict, context) -> dict:
    """
    Main Lambda entry point.
    Routes based on HTTP method + path from API Gateway event.
    """
    method = event.get("httpMethod", "GET").upper()
    path   = event.get("path", "/")

    if path == "/health" and method == "GET":
        return health_handler(event, context)

    if path == "/analyze" and method == "POST":
        return analyze_handler(event, context)

    return _response(404, {"error": f"No handler for {method} {path}"})
