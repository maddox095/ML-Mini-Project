"""Factories for implemented model families and the chance baseline."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
import numpy as np

MODELS = {"logistic_regression": LogisticRegression, "svm_linear": LinearSVC,
          "svm_rbf": SVC, "svm_polynomial": SVC, "decision_tree": DecisionTreeClassifier,
          "random_forest": RandomForestClassifier, "neural_network": MLPClassifier}
TITLES = {"logistic_regression": "Logistic Regression", "svm_linear": "Linear SVM",
          "svm_rbf": "RBF SVM", "svm_polynomial": "Polynomial SVM",
          "decision_tree": "Decision Tree", "random_forest": "Random Forest",
          "neural_network": "Six-unit Neural Network"}
TREE_MODELS = {"decision_tree", "random_forest"}


def make_estimator(name: str, config: dict):
    if name == "dummy":
        return DummyClassifier(strategy="prior", random_state=config["seed"])
    if name not in MODELS:
        raise ValueError(f"Model not implemented: {name}")
    parameters = {key: value for key, value in config[name].items()
                  if not key.endswith("_grid") and key != "search_max_candidates"}
    if name == "neural_network":
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    return MODELS[name](random_state=config["seed"], **parameters)


def search_space(name: str, config: dict) -> dict | list:
    if name == "dummy":
        return {}
    if name not in MODELS:
        raise ValueError(f"Model not implemented: {name}")
    settings = config[name]
    values = settings.get("C_grid")
    if values is not None and (not values or any(value <= 0 for value in values)):
        raise ValueError("C_grid must contain positive regularization strengths")
    grid = {f"model__{key[:-5]}": values for key, values in settings.items() if key.endswith("_grid")}
    if not grid or any(not values for values in grid.values()):
        raise ValueError("Every model must have nonempty tuning choices")
    candidates = list(ParameterGrid(grid))
    cap = settings.get("search_max_candidates", len(candidates))
    if cap <= 0:
        raise ValueError("Search candidate limit must be positive")
    if len(candidates) <= cap:
        return grid
    chosen = sorted(np.random.default_rng(config["seed"]).choice(len(candidates), cap, replace=False))
    return [{key: [value] for key, value in candidates[index].items()} for index in chosen]
