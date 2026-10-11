"""REST API: health, model card, predictions, explanations, input validation and CORS."""
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware

from api.predict import app

client = TestClient(app)
HYPO = {"TSH": 15, "T4": 60}


def allowed_origins() -> list[str]:
    for m in app.user_middleware:
        if m.cls is CORSMiddleware:
            return list(m.kwargs.get("allow_origins", []))
    return []


def test_health():
    assert client.get("/health").status_code == 200


def test_model_card():
    info = client.get("/model-info").json()
    assert info["model"] and info["version"]
    assert info["dataset_size"] == 150000
    assert 0.8 < info["test_metrics"]["accuracy"] < 0.9, "the honest, leak free accuracy is about 85%"


def test_predict_hypothyroid_case():
    res = client.post("/predict", json=HYPO)
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == "hypothyroid"
    if isinstance(body.get("probabilities"), dict):
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_out_of_range_input_is_rejected():
    res = client.post("/predict", json={"TSH": -1})
    assert res.status_code == 422
    assert "TSH" in res.text


def test_wrong_type_is_rejected():
    assert client.post("/predict", json={"TSH": "high"}).status_code == 422


def test_explain_returns_contributions_for_model_features():
    body = client.post("/explain", json=HYPO).json()
    assert body["prediction"] == "hypothyroid"
    assert body["contributions"], "an explanation must list feature contributions"
    features = set(client.get("/model-info").json().get("features") or [])
    if features:
        assert {c["feature"] for c in body["contributions"]} <= features


def test_cors_allows_the_frontend_origin():
    origins = [o for o in allowed_origins() if o != "*"]
    assert origins, "CORS must be locked to explicit origins"
    res = client.options("/predict", headers={"Origin": origins[0], "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "content-type"})
    assert res.headers.get("access-control-allow-origin") == origins[0]


def test_cors_blocks_other_origins():
    res = client.options("/predict", headers={"Origin": "https://not-allowed.example", "Access-Control-Request-Method": "POST"})
    assert res.headers.get("access-control-allow-origin") is None
