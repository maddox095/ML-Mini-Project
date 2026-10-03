"""User 2 preprocessing factories. Fit these only inside a training pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.common import load_config, model_features


def feature_order(config: dict | None = None, *, features: list[str] | None = None) -> list[str]:
    allowed = model_features(config)
    if features is None:
        return allowed
    if not features or len(features) != len(set(features)) or not set(features).issubset(allowed):
        raise ValueError("Features must be a nonempty, unique subset of the model input whitelist")
    return [feature for feature in allowed if feature in features]


def make_preprocessor(*, scale: bool = True, config: dict | None = None,
                      features: list[str] | None = None) -> ColumnTransformer:
    if config is None:
        config = load_config()
    selected = feature_order(config, features=features)
    numeric = [feature for feature in selected if feature != "artist_score"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if "artist_score" in selected:
        transformers.append(("binary", "passthrough", ["artist_score"]))
    return ColumnTransformer(transformers, remainder="drop")


def make_pipeline(estimator, *, scale: bool = True, config: dict | None = None,
                  features: list[str] | None = None) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor(scale=scale, config=config, features=features)),
                     ("model", estimator)])
