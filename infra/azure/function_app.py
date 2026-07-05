"""
Azure Functions handler for Document Portal.

Wraps /health and /analyze as Azure HTTP-triggered functions.

Setup:
    1. Install Azure Functions Core Tools:
       npm install -g azure-functions-core-tools@4

    2. Login:
       az login
       az account set --subscription YOUR_SUBSCRIPTION_ID

    3. Create Function App (free tier):
       az group create --name document-portal-rg --location uksouth
       az storage account create --name docportalstore --resource-group document-portal-rg --sku Standard_LRS
       az functionapp create \
         --name document-portal-func \
         --resource-group document-portal-rg \
         --consumption-plan-location uksouth \
         --runtime python \
         --runtime-version 3.10 \
         --functions-version 4 \
         --storage-account docportalstore

    4. Deploy:
       cd infra/azure
       func azure functionapp publish document-portal-func

    5. Set env vars:
       az functionapp config appsettings set \
         --name document-portal-func \
         --resource-group document-portal-rg \
         --settings OPENAI_API_KEY=sk-... LLM_PROVIDER=openai ENV=production
"""

from __future__ import annotations

import json
import logging
import os
import sys
import base64
import tempfile

import azure.functions as func  # type: ignore

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ── /health ───────────────────────────────────────────────────────────────────

@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Liveness check."""
    return func.HttpResponse(
        json.dumps({"status": "ok", "service": "document-portal-azure"}),
        mimetype="application/json",
        status_code=200,
    )


# ── /analyze ──────────────────────────────────────────────────────────────────

@app.route(route="analyze", methods=["POST"])
def analyze(req: func.HttpRequest) -> func.HttpResponse:
    """
    Accepts a base64-encoded PDF and returns extracted metadata.

    Request body (JSON):
        {
            "file_content": "<base64-encoded PDF bytes>",
            "filename": "contract.pdf"
        }
    """
    try:
        body = req.get_json()
        file_content_b64 = body.get("file_content")

        if not file_content_b64:
            return func.HttpResponse(
                json.dumps({"error": "file_content (base64) is required"}),
                mimetype="application/json",
                status_code=400,
            )

        file_bytes = base64.b64decode(file_content_b64)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        from src.document_ingestion.data_ingestion import DocHandler
        from src.document_analyzer.data_analysis import DocumentAnalyzer

        dh = DocHandler(data_dir=tempfile.gettempdir())
        text = dh.read_pdf(tmp_path)

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)

        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
        )

    except Exception as exc:
        logging.error("Azure Function analyze failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": str(exc)}),
            mimetype="application/json",
            status_code=500,
        )
