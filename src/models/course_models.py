"""Course-scoped factories: regression, decision trees, bagging and boosting."""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import make_pipeline

FAMILIES = {
    "logistic_regression": LogisticRegression,
    "polynomial_logistic": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "bagged_trees": BaggingClassifier,
    "random_forest": RandomForestClassifier,
    "adaboost": AdaBoostClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}
REGRESSION = {"logistic_regression", "polynomial_logistic"}


def make_course_pipeline(name, config):
    if name not in FAMILIES:
        raise ValueError(f"Outside the authorized course scope: {name}")
    settings = config["models"][name]
    params = {"random_state": config["seed"], **settings["fixed"]}
    if "base_tree" in settings:
        params["estimator"] = DecisionTreeClassifier(random_state=config["seed"], **settings["base_tree"])
    pipeline = make_pipeline(FAMILIES[name](**params), features=config["features"], scale=name in REGRESSION)
    if name == "polynomial_logistic":
        pipeline = Pipeline([pipeline.steps[0], ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                             ("expansion_scaler", StandardScaler()), pipeline.steps[1]])
    return pipeline


def course_grid(name, config):
    settings = config["models"][name]
    candidates = list(ParameterGrid(settings["grid"]))
    limit = settings.get("max_candidates", len(candidates))
    if limit < 1:
        raise ValueError("Search budget must be positive")
    if len(candidates) <= limit:
        return settings["grid"]
    anchors = settings.get("anchors", [])
    if len(anchors) > limit or any(anchor not in candidates for anchor in anchors):
        raise ValueError("Anchors must fit the declared grid and budget")
    chosen = {candidates.index(anchor) for anchor in anchors}
    available = [index for index in range(len(candidates)) if index not in chosen]
    chosen.update(np.random.default_rng(config["seed"]).choice(available, limit-len(chosen), replace=False).tolist())
    return [{key: [value] for key, value in candidates[index].items()} for index in sorted(chosen)]


def json_parameters(value):
    """Retain full nested estimator configuration as actual JSON, not repr/hash."""
    if hasattr(value, "get_params"):
        return {"class": f"{type(value).__module__}.{type(value).__name__}",
                "parameters": json_parameters(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {key: json_parameters(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_parameters(item) for item in value]
    if isinstance(value, np.generic):
        return json_parameters(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {"special_float": str(value)}
    return value
