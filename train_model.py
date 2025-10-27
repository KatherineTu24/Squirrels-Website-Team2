import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ttl_parser import parse_ttl_file, convert_ttl_to_csv_format


ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")


def _ensure_artifacts_dir() -> None:

    if not os.path.isdir(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _load_data(ttl_path: str) -> pd.DataFrame:
    """
    Load data from TTL file and convert to DataFrame for ML training
    
    Args:
        ttl_path: Path to the TTL file
        
    Returns:
        DataFrame with normalized column names
    """
    # Parse TTL file
    ttl_data = parse_ttl_file(ttl_path)
    
    # Convert TTL data to CSV-like format
    csv_data = convert_ttl_to_csv_format(ttl_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(csv_data)
    
    # Normalize column names (same as original)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_")
    )
    
    print(f"Loaded {len(df)} records from TTL file")
    return df


def _engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:

    # Select target and candidate features
    if "park_name" not in df.columns:
        # merged csv stores park_name on the left; unify in case of suffixes
        park_cols = [c for c in df.columns if c.startswith("park_name")]
        if not park_cols:
            raise ValueError("park_name column not found in dataset")
        target_col = park_cols[0]
    else:
        target_col = "park_name"

    # Time of day bucketing from start_time if available
    def to_time_of_day(val: Any) -> str:
        try:
            s = str(val)
            if pd.isna(val) or s.strip() == "":
                return "unknown"
            # Try pandas to_datetime with infer
            ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                return "unknown"
            hour = ts.hour
            if 5 <= hour < 12:
                return "morning"
            if 12 <= hour < 17:
                return "afternoon"
            if 17 <= hour < 21:
                return "evening"
            return "night"
        except Exception:
            return "unknown"

    if "start_time" in df.columns:
        df["time_of_day"] = df["start_time"].apply(to_time_of_day)
    else:
        df["time_of_day"] = "unknown"

    # Weather bucketing if present
    def to_weather_bucket(val: Any) -> str:
        if pd.isna(val):
            return "unknown"
        s = str(val).lower()
        if any(k in s for k in ["sunny", "clear"]):
            return "clear"
        if any(k in s for k in ["cloud", "overcast"]):
            return "cloudy"
        if any(k in s for k in ["rain", "drizzle"]):
            return "rain"
        if any(k in s for k in ["snow", "sleet"]):
            return "snow"
        return "other"

    weather_col = None
    for cand in ["temperature_&_weather", "weather", "conditions", "park_conditions"]:
        if cand in df.columns:
            weather_col = cand
            break
    if weather_col is not None:
        df["weather_bucket"] = df[weather_col].apply(to_weather_bucket)
    else:
        df["weather_bucket"] = "unknown"

    # Normalize fur color
    fur_col = None
    for cand in ["fur_color", "primary_fur_color"]:
        if cand in df.columns:
            fur_col = cand
            break
    if fur_col is None:
        raise ValueError("fur_color column not found in dataset")

    df[fur_col] = df[fur_col].fillna("Unknown").str.strip().str.title()

    # Location normalization
    if "location" in df.columns:
        df["location"] = df["location"].fillna("Unknown").str.strip().str.title()
    else:
        df["location"] = "Unknown"

    # Feature set
    feature_cols = [fur_col, "location", "time_of_day", "weather_bucket"]
    X = df[feature_cols].copy()
    y = df[target_col].astype(str)

    categorical_features = feature_cols
    numeric_features: List[str] = []
    return X, y, categorical_features, numeric_features


def train(ttl_path: str = "Ontology.tll", random_state: int = 42, tune_hyperparameters: bool = False) -> Dict[str, Any]:
    """
    Train ML model using TTL data
    
    Args:
        ttl_path: Path to the TTL file containing squirrel sighting data
        random_state: Random seed for reproducibility
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary containing training results and metadata
    """
    _ensure_artifacts_dir()
    df = _load_data(ttl_path)
    X, y, cat_features, num_features = _engineer_features(df)

    # Preprocessor: OneHot for categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    base_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        random_state=random_state,
        n_jobs=1,  # Disable multiprocessing for Windows compatibility
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    # Handle rare classes: remove parks with < 2 samples so stratified split is valid
    class_counts = y.value_counts()
    valid_mask = y.map(class_counts) >= 2
    dropped_classes = sorted(class_counts[class_counts < 2].index.tolist())
    if not valid_mask.all():
        X = X.loc[valid_mask].reset_index(drop=True)
        y = y.loc[valid_mask].reset_index(drop=True)

    # Require at least 2 classes after filtering
    if y.nunique() < 2:
        raise ValueError(
            "Not enough classes to train after filtering rare parks (<2 samples). "
            f"Remaining classes: {sorted(y.unique())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    best_params: Dict[str, Any] | None = None
    best_cv_score: float | None = None
    cv_results_summary: List[Dict[str, Any]] | None = None

    if tune_hyperparameters:
        # Use at most 5 folds but do not exceed the smallest class count
        n_splits = int(min(5, y_train.value_counts().min()))
        if n_splits < 2:
            # Fallback: no tuning if not enough samples per class for CV
            clf.fit(X_train, y_train)
        else:
            param_grid = {
                "model__n_estimators": [200, 400, 800],
                "model__max_depth": [None, 10, 20, 40],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            }
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            grid = GridSearchCV(
                estimator=clf,
                param_grid=param_grid,
                scoring="accuracy",
                cv=cv,
                n_jobs=1,  # Disable multiprocessing for Windows compatibility
                refit=True,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_
            best_params = grid.best_params_
            best_cv_score = float(grid.best_score_)

            # Compact summary of top configs
            results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
            keep_cols = [
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
                *[c for c in results.columns if c.startswith("param_model__")],
            ]
            results = results[keep_cols].head(10)
            cv_results_summary = results.to_dict(orient="records")
    else:
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Derive feature names after one-hot
    ohe: OneHotEncoder = clf.named_steps["preprocessor"].named_transformers_["cat"]
    ohe_feature_names = ohe.get_feature_names_out(cat_features)

    # Approximate feature importance by mapping back the sum of importances per original feature group
    rf: RandomForestClassifier = clf.named_steps["model"]
    importances = getattr(rf, "feature_importances_", None)
    grouped: Dict[str, float] = {f: 0.0 for f in cat_features}
    if importances is not None:
        for fname, imp in zip(ohe_feature_names, importances):
            # fname like "fur_color_Brown"
            original = fname.split("_")[0]
            if original in grouped:
                grouped[original] += float(imp)

    feature_importance = [
        {"feature": k, "importance": v} for k, v in sorted(grouped.items(), key=lambda x: x[1], reverse=True)
    ] if any(grouped.values()) else [{"feature": f, "importance": None} for f in cat_features]

    # Persist artifacts
    try:
        import joblib
    except ImportError as e:
        raise RuntimeError("joblib is required; add it to requirements.txt") from e

    joblib.dump(clf, MODEL_PATH)
    # Preprocessor is embedded in pipeline; also persist separately for reference
    joblib.dump(preprocessor, ENCODER_PATH)

    metadata = {
        "features_used": cat_features,
        "feature_importance": feature_importance,
        "accuracy": acc,
        "classification_report": report,
        "target": "park_name",
        "n_samples": int(len(X)),
        "n_classes": int(y.nunique()),
        "parks": sorted(y.unique()),
        "dropped_parks_rare": dropped_classes,
        "tuning": {
            "enabled": bool(tune_hyperparameters),
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "cv_results_summary": cv_results_summary,
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    # Train using TTL data instead of CSV
    meta = train(ttl_path="Ontology.tll")
    print(json.dumps({
        "accuracy": meta["accuracy"],
        "features_used": meta["features_used"],
        "top_feature": meta["feature_importance"][0] if meta["feature_importance"] else None,
        "n_classes": meta["n_classes"],
    }, indent=2))


