from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "crop_yield_training.csv"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"

st.set_page_config(page_title="CropYield AI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #f6faf5 0%, #ffffff 42%);
        color: #173f35;
    }
    [data-testid="stSidebar"] {background-color: #173f35;}
    [data-testid="stSidebar"] * {color: #ffffff !important;}
    .hero {padding: 1.5rem 1.7rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg, #173f35, #2f7254);}
    .hero h1 {margin-bottom: .3rem; font-size: 2.5rem;}
    .hero p {margin-bottom: 0; color: #e3f0e5;}
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
        color: #173f35;
    }
    .hero h1, .hero p {color: #ffffff !important;}
    .hero p {color: #e3f0e5 !important;}

    div[data-testid="stMetric"] {
        background: #e8f2ea;
        border: 1px solid #bfd9c4;
        padding: 1rem;
        border-radius: 12px;
    }
    div[data-testid="stMetric"] * {color: #173f35 !important;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0b3526 !important;
        font-weight: 700;
    }

    div[data-testid="stAlert"] {
        background: #dcf5e5;
        border: 1px solid #65a978;
        border-left: 5px solid #217a3d;
        color: #123d23;
    }
    div[data-testid="stAlert"] * {color: #123d23 !important;}
    div.stButton > button {
        background: #1e7a45;
        border-color: #1e7a45;
        color: #ffffff;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #145b32;
        border-color: #145b32;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()


def model_features(metadata, model):
    """Return the model input columns saved by the training workflow."""
    features = metadata.get("features", []) if metadata else []
    if features:
        return list(features)
    return list(getattr(model, "feature_names_in_", []))

st.markdown("""
<div class="hero">
  <h1>🌱 CropYield AI</h1>
  <p>Explore FAOSTAT crop statistics and estimate crop yield with machine learning.</p>
</div>
""", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.title("🌾 Navigation")
    page = st.radio("Go to", ["Predict Yield", "Data Explorer", "Model Analytics", "About & Sources"])

data = load_data()
model_exists = MODEL_PATH.exists() and META_PATH.exists()
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

if page == "Predict Yield":
    st.subheader("Predict crop yield")
    st.write("Enter the crop and historical context. Prediction quality depends on the training data and model validation.")
    if not model_exists:
        st.warning("The trained model has not been created yet. Run the Colab notebook first, then place the generated model files in the `models/` folder.")
        st.code("python src/faostat_scraper.py\npython src/train_model.py", language="bash")
    else:
        model = load_model()
        features = model_features(metadata, model)
        required_base_features = {"Country", "Crop", "Year"}
        area_features = [feature for feature in features if feature.lower().startswith("area_harvested")]
        unsupported_features = set(features) - required_base_features - set(area_features)
        if not required_base_features.issubset(features) or len(area_features) != 1 or unsupported_features:
            st.error(
                "The saved model has an unsupported feature schema: "
                + ", ".join(features or ["no feature metadata"])
                + ". Re-export the trained model together with compatible metadata."
            )
            st.stop()
        area_feature = area_features[0]
        countries = sorted(data["Country"].dropna().astype(str).unique()) if not data.empty and "Country" in data else []
        crops = sorted(data["Crop"].dropna().astype(str).unique()) if not data.empty and "Crop" in data else []
        c1, c2 = st.columns(2)
        with c1:
            country = st.selectbox("Country / area", countries or ["India"])
            crop = st.selectbox("Crop", crops or ["Rice"])
            year = st.slider("Prediction year", 1961, 2040, 2020,
                             help="Years after the historical training period are extrapolations from the model.")
        with c2:
            area = st.number_input("Area harvested (hectares)", min_value=0.0, value=1000.0, step=100.0)
            st.info("Use hectares, the unit used by the FAOSTAT training data. Historical estimates are not farm-specific guarantees.")
        if st.button("Estimate yield", type="primary", use_container_width=True):
            values = {
                "Country": country, "Crop": crop, "Year": int(year),
                area_feature: float(area),
            }
            # Match the persisted Colab model's column names and ordering exactly.
            row = pd.DataFrame([[values[feature] for feature in features]], columns=features)
            prediction = float(model.predict(row)[0])
            st.success("Prediction generated by the trained model.")
            if year > 2025:
                st.info(
                    "This is a future-year extrapolation. It is based on historical FAOSTAT patterns, "
                    "not a guaranteed crop outcome."
                )
            m1, m2 = st.columns(2)
            m1.metric("Estimated yield", f"{prediction:,.1f} kg/ha")
            m2.metric("Estimated production at entered area", f"{prediction * area / 1000:,.2f} tonnes")
            st.caption("Production estimate is derived from predicted yield × entered area; it is not a separately trained target.")
            st.caption("This is an educational estimate, not agronomic advice.")
    if not data.empty:
        st.markdown("### Historical dataset snapshot")
        a, b, c = st.columns(3)
        a.metric("Records", f"{len(data):,}")
        b.metric("Countries / areas", f"{data['Country'].nunique():,}" if "Country" in data else "—")
        c.metric("Crops", f"{data['Crop'].nunique():,}" if "Crop" in data else "—")

elif page == "Data Explorer":
    st.subheader("FAOSTAT data explorer")
    if data.empty:
        st.info("No cleaned dataset found yet. Run the Colab workflow to create `data/crop_yield_training.csv`.")
    else:
        view = data.copy()
        if "Country" in view:
            country_filter = st.selectbox("Filter country / area", ["All"] + sorted(view["Country"].dropna().astype(str).unique()))
            if country_filter != "All":
                view = view[view["Country"].astype(str) == country_filter]
        if "Crop" in view:
            crop_filter = st.selectbox("Filter crop", ["All"] + sorted(view["Crop"].dropna().astype(str).unique()))
            if crop_filter != "All":
                view = view[view["Crop"].astype(str) == crop_filter]
        st.dataframe(view.head(1000), use_container_width=True, hide_index=True)
        if "Year" in view and "Yield_kg_ha" in view:
            yearly = view.groupby("Year", as_index=False)["Yield_kg_ha"].mean()
            st.plotly_chart(px.line(yearly, x="Year", y="Yield_kg_ha", markers=True,
                                   title="Mean recorded yield over time"), use_container_width=True)
        if "Crop" in view and "Yield_kg_ha" in view:
            top = view.groupby("Crop", as_index=False)["Yield_kg_ha"].mean().nlargest(15, "Yield_kg_ha")
            st.plotly_chart(px.bar(top, x="Crop", y="Yield_kg_ha", title="15 crops with highest mean recorded yield"),
                            use_container_width=True)
        st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="faostat_crop_yield_filtered.csv", mime="text/csv")

elif page == "Model Analytics":
    st.subheader("Model performance")
    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption("R² closer to 1 is generally better; lower MAE and RMSE are generally better. Interpret metrics in light of split strategy and data structure.")
        if "RMSE" in metrics and "Model" in metrics:
            st.plotly_chart(px.bar(metrics, x="Model", y="RMSE", title="Model comparison: RMSE (lower is better)"),
                            use_container_width=True)
    else:
        st.info("Model metrics will appear here after training in Colab.")
    if metadata:
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected model", metadata.get("best_model", "—"))
        c2.metric("Training rows", f"{metadata.get('training_rows', 0):,}")
        c3.metric("Target", metadata.get("target", "—"))
    st.markdown("### Evaluation notes")
    st.markdown("- **MAE:** average absolute prediction error.\n- **RMSE:** penalizes larger errors more strongly.\n- **R²:** proportion of target variance explained relative to a mean baseline.\n- For a stronger final report, use a time-based split or group-based validation and compare against a simple baseline.")

else:
    st.subheader("About this project")
    st.markdown("""
**Objective:** Develop an end-to-end regression workflow for crop yield using historical FAOSTAT statistics.

**Workflow**
1. Scrape the public FAOSTAT dataset page to identify the data archive.
2. Download and parse the official normalized CSV archive.
3. Clean and reshape the data into crop–country–year observations.
4. Train and compare regression models.
5. Use the saved pipeline in this Streamlit interface.

**Data source:** Food and Agriculture Organization of the United Nations (FAO), FAOSTAT.
- Main portal: https://www.fao.org/faostat/en/
- Crops and livestock products domain: https://www.fao.org/faostat/en/#data/QCL
- Developer portal: https://www.fao.org/faostat/en/#developer-portal

**Limitations:** FAOSTAT observations are national/area-level historical statistics, not plot-level measurements. The model does not directly account for soil quality, local weather, pests, irrigation or farm management unless such variables are separately sourced and integrated. Predictions should be interpreted as educational estimates.
""")
    if metadata:
        st.markdown("### Model metadata")
        st.json(metadata)

st.markdown("---")
st.caption("CropYield AI · Educational use · Data attribution: FAOSTAT / Food and Agriculture Organization of the United Nations.")
