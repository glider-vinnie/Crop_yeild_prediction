"""
Train and evaluate crop-yield regression models from FAOSTAT QCL data.
Target: FAOSTAT's Yield element. Production is deliberately NOT used as a feature.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MAX_TRAINING_ROWS = 50_000

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df, candidates):
    lookup = {c.lower().replace("_", " ").strip(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", " ").strip()
        if key in lookup:
            return lookup[key]
    return None

def prepare_training_data(raw_df):
    """Convert FAOSTAT long-format records to one row per area/item/year."""
    df = normalize_columns(raw_df)
    area_col = find_column(df, ["Area", "Country", "Area Name"])
    item_col = find_column(df, ["Item", "Item Name", "Crop"])
    element_col = find_column(df, ["Element", "Element Name"])
    year_col = find_column(df, ["Year", "Year Code"])
    value_col = find_column(df, ["Value", "Value Numeric"])
    unit_col = find_column(df, ["Unit"])
    if not all([area_col, item_col, element_col, year_col, value_col]):
        raise ValueError(
            "Could not identify required FAOSTAT columns. Found: "
            + ", ".join(df.columns.astype(str))
        )

    df = df.rename(columns={
        area_col: "Area", item_col: "Item", element_col: "Element",
        year_col: "Year", value_col: "Value"
    })
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(
        df["Value"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df = df.dropna(subset=["Area", "Item", "Year", "Element", "Value"])
    df["Year"] = df["Year"].astype(int)

    # Keep only country-level records; aggregate/subregional totals can duplicate
    # countries and should not be mixed into the same training observations.
    if "Area Code (M49)" in df.columns:
        df = df[df["Area Code (M49)"].notna()].copy()

    df["Element_norm"] = df["Element"].astype(str).str.strip().str.lower()
    # The QCL domain typically stores Yield in hg/ha. Preserve original unit
    # in metadata; convert to kg/ha for a more intuitive app output.
    yield_rows = df[df["Element_norm"].eq("yield")].copy()
    area_rows = df[df["Element_norm"].str.contains("area harvested", na=False)].copy()

    keys = ["Area", "Item", "Year"]
    if yield_rows.empty:
        raise ValueError("No rows with Element='Yield' found in the downloaded QCL data.")
    target = yield_rows[keys + ["Value"]].rename(columns={"Value": "Yield_raw"})
    target = target.groupby(keys, as_index=False)["Yield_raw"].mean()

    # Area harvested is an allowed feature and is matched to the same crop/year.
    if not area_rows.empty:
        area_feature = area_rows[keys + ["Value"]].rename(columns={"Value": "Area_harvested_ha"})
        area_feature = area_feature.groupby(keys, as_index=False)["Area_harvested_ha"].mean()
        target = target.merge(area_feature, on=keys, how="left")
    else:
        target["Area_harvested_ha"] = np.nan

    # Yield unit commonly hg/ha: 1 hg = 0.1 kg. If units differ, retain the
    # source values and clearly expose the original unit in the metadata.
    source_unit = "unknown"
    if unit_col:
        unit_lookup = df[df["Element_norm"].eq("yield")][unit_col].dropna().astype(str)
        if len(unit_lookup):
            source_unit = unit_lookup.mode().iloc[0]
    if source_unit.lower().replace(" ", "") in {"hg/ha", "hgperha"}:
        target["Yield_kg_ha"] = target["Yield_raw"] * 0.1
    elif source_unit.lower().replace(" ", "") in {"kg/ha", "kgperha"}:
        target["Yield_kg_ha"] = target["Yield_raw"]
    elif source_unit.lower().replace(" ", "") in {"t/ha", "tonnes/ha", "tonnesperha"}:
        target["Yield_kg_ha"] = target["Yield_raw"] * 1000
    else:
        # Keep the source-unit numeric scale when the unit is unexpected.
        target["Yield_kg_ha"] = target["Yield_raw"]
        warnings.warn(
            f"Yield unit '{source_unit}' not recognized; target retains source numeric scale. "
            "Check FAOSTAT units before interpreting the output."
        )

    target = target.rename(columns={"Area": "Country", "Item": "Crop"})
    target = target.replace([np.inf, -np.inf], np.nan)
    target = target.dropna(subset=["Yield_kg_ha"])
    target = target[(target["Yield_kg_ha"] > 0) & (target["Yield_kg_ha"] < target["Yield_kg_ha"].quantile(0.995))]
    target = target.drop_duplicates(subset=["Country", "Crop", "Year"])
    target = target[["Country", "Crop", "Year", "Area_harvested_ha", "Yield_kg_ha"]]
    metadata = {
        "source": "FAOSTAT QCL - Production: Crops and livestock products",
        "source_unit": source_unit,
        "target_unit": "kg/ha when source unit is recognized; otherwise source unit scale",
        "rows": int(len(target)),
        "features": ["Country", "Crop", "Year", "Area_harvested_ha"],
        "target": "Yield_kg_ha",
        "note": "Production is not used as an input feature to avoid target leakage."
    }
    return target, metadata

def train_models(data):
    features = ["Country", "Crop", "Year", "Area_harvested_ha"]
    target = "Yield_kg_ha"
    X = data[features].copy()
    y = data[target].copy()

    categorical = ["Country", "Crop"]
    numeric = ["Year", "Area_harvested_ha"]
    preprocess = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.08, l2_regularization=1.0, random_state=42
        )
    }
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    results, fitted = [], {}
    for name, estimator in models.items():
        # HistGradientBoosting does not accept sparse one-hot matrices, so use
        # Random Forest / Linear Regression for this shared preprocessing pipeline.
        if name == "Hist Gradient Boosting":
            continue
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        results.append({
            "Model": name,
            "R2": float(r2_score(y_test, pred)),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": rmse
        })
        fitted[name] = pipe

    # Choose lowest RMSE, with R² and MAE reported alongside it.
    metrics = pd.DataFrame(results).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    metrics.to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    metadata = {
        "best_model": best_name,
        "features": features,
        "target": target,
        "target_unit": "kg/ha when source unit is recognized; otherwise source unit scale",
        "training_rows": int(len(data)),
        "test_size": 0.2,
        "random_state": 42,
        "selection_metric": "lowest test-split RMSE (educational baseline; use grouped/time validation for final scientific claims)"
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, best_model, metadata

if __name__ == "__main__":
    raw_path = DATA_DIR / "raw" / "faostat_qcl_raw.csv"
    clean_path = DATA_DIR / "crop_yield_training.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path, low_memory=False)
        data, data_metadata = prepare_training_data(raw)
        data.to_csv(clean_path, index=False)
    elif clean_path.exists():
        # A Colab export already contains the prepared data, so it can be used
        # to rebuild a local-compatible model without downloading FAOSTAT again.
        data = pd.read_csv(clean_path, low_memory=False)
        legacy_area_column = "Area_harvested_source_value"
        if legacy_area_column in data.columns and "Area_harvested_ha" not in data.columns:
            data = data.rename(columns={legacy_area_column: "Area_harvested_ha"})
        required_columns = {"Country", "Crop", "Year", "Area_harvested_ha", "Yield_kg_ha"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError("Cleaned training data is missing columns: " + ", ".join(sorted(missing)))
        data_metadata = {
            "source": "FAOSTAT QCL - Production: Crops and livestock products",
            "source_unit": "kg/ha",
            "target_unit": "kg/ha",
            "rows": int(len(data)),
            "features": ["Country", "Crop", "Year", "Area_harvested_ha"],
            "target": "Yield_kg_ha",
            "note": "Rebuilt from the cleaned Colab export; production is not an input feature.",
        }
    else:
        raise FileNotFoundError(
            f"Neither {raw_path} nor {clean_path} exists. Run src/faostat_scraper.py first."
        )

    # Keep the local rebuild fast and reproducible, matching the original
    # Colab artifact's 50,000-row training sample.
    training_data = data.sample(n=min(MAX_TRAINING_ROWS, len(data)), random_state=42)
    metrics, model, metadata = train_models(training_data)
    metadata.update(data_metadata)
    metadata["full_cleaned_rows"] = int(len(data))
    metadata["training_sample_rows"] = int(len(training_data))
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print("Saved model to models/best_model.joblib")
