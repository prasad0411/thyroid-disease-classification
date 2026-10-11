"""
GraphQL API, served at /graphql next to the REST endpoints.

REST stays the simple path for predictions. GraphQL lets a client ask for exactly the fields it
needs in one round trip, for example the model card and an explanation together, and the React
app reads model information through Apollo Client, which caches it.
"""
from __future__ import annotations

import strawberry
from pydantic import ValidationError
from strawberry.fastapi import GraphQLRouter


@strawberry.type
class Metrics:
    accuracy: float | None
    f1_macro: float | None
    log_loss: float | None


@strawberry.type
class ModelInfo:
    model: str
    version: str
    dataset_size: int
    majority_baseline: float | None
    test_metrics: Metrics
    features: list[str]


@strawberry.type
class ClassProbability:
    label: str
    probability: float


@strawberry.type
class Contribution:
    feature: str
    value: float
    shap: float


@strawberry.type
class Explanation:
    prediction: str
    confidence: float
    probabilities: list[ClassProbability]
    base_value: float
    method: str
    contributions: list[Contribution]
    model_version: str


@strawberry.input
class PatientInput:
    TSH: float = 2.5
    T3: float = 1.8
    T4: float = 105.0
    T4U: float = 1.0
    age: float = 45.0
    sex: int = 0
    on_thyroxine: int = 0
    on_antithyroid: int = 0
    sick: int = 0
    query_hypothyroid: int = 0
    query_hyperthyroid: int = 0


def build_router(bundle, explain_one, rest_patient_model) -> GraphQLRouter:
    """Builds the GraphQL router around the already loaded model, reusing the REST validation rules."""

    @strawberry.type
    class Query:
        @strawberry.field(description="The deployed model, its version and held out test metrics.")
        def model_info(self) -> ModelInfo:
            m = bundle.metrics or {}
            return ModelInfo(
                model=bundle.model_name, version=bundle.timestamp, dataset_size=bundle.dataset_size,
                majority_baseline=bundle.baseline, features=list(bundle.features),
                test_metrics=Metrics(accuracy=m.get("accuracy"), f1_macro=m.get("f1_macro"), log_loss=m.get("log_loss")),
            )

        @strawberry.field(description="Prediction with per feature SHAP contributions for one patient.")
        def explain(self, patient: PatientInput) -> Explanation:
            try:
                checked = rest_patient_model(**vars(patient))
            except ValidationError as e:
                fields = ", ".join(str(err["loc"][0]) for err in e.errors())
                raise ValueError(f"Invalid input: {fields}") from None
            out = explain_one(bundle, checked.model_dump())
            return Explanation(
                prediction=out["prediction"], confidence=out["confidence"], base_value=out["base_value"],
                method=out["method"], model_version=bundle.timestamp,
                probabilities=[ClassProbability(label=k, probability=v) for k, v in out["probabilities"].items()],
                contributions=[Contribution(**c) for c in out["contributions"]],
            )

    return GraphQLRouter(strawberry.Schema(query=Query))
