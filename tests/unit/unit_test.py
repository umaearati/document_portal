# tests/test_unit_cases.py

import pytest
from fastapi.testclient import TestClient
from api.main import app   # or your FastAPI entrypoint

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Document Portal" in response.text
    
    
    # fastAPI routing
    # App initialisation
    # Endpoint logic
    
    
    
#     “Is 100% coverage always good?”

# Not necessarily. Coverage measures execution, not correctness. It’s possible to have high coverage with weak assertions. Strong tests validate behaviour, edge cases, and failure scenarios.


# So Why 90%+ Is Called “Very Strong”?

# Because in real systems:

# It’s hard to reach 100%
# Some lines are defensive
# Some code paths are rarely triggered
# Some third-party integrations are mocked

# 90%+ usually means:
# Almost all important logic is tested
# Edge cases covered
# Error handling tested