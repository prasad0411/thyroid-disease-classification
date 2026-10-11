"""GraphQL endpoint: model info, an explanation, and validation errors."""
from fastapi.testclient import TestClient

from api.predict import app

client = TestClient(app)


def gql(query: str, variables: dict | None = None) -> dict:
    res = client.post("/graphql", json={"query": query, "variables": variables or {}})
    assert res.status_code == 200
    return res.json()


def test_model_info_matches_rest():
    body = gql("{ modelInfo { model version datasetSize testMetrics { accuracy f1Macro } } }")
    info = body["data"]["modelInfo"]
    rest = client.get("/model-info").json()
    assert info["model"] == rest["model"] and info["version"] == rest["version"]
    assert info["datasetSize"] == rest["dataset_size"]
    assert info["testMetrics"]["accuracy"] == rest["test_metrics"]["accuracy"]


def test_explain_returns_only_requested_fields():
    q = "query($p: PatientInput!) { explain(patient: $p) { prediction contributions { feature shap } } }"
    out = gql(q, {"p": {"TSH": 15, "T4": 60}})["data"]["explain"]
    assert out["prediction"] == "hypothyroid"
    assert set(out) == {"prediction", "contributions"}
    assert {"feature", "shap"} == set(out["contributions"][0])


def test_invalid_input_is_a_graphql_error():
    q = "query($p: PatientInput!) { explain(patient: $p) { prediction } }"
    body = gql(q, {"p": {"TSH": -1}})
    assert body["data"] is None
    assert "Invalid input: TSH" in body["errors"][0]["message"]
