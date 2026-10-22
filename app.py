
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Sales Revenue Predictor",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    with open(BASE_DIR / "random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(BASE_DIR / "scaler (2).pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

NUMERIC_FEATURES = [
    "price",
    "discount_rate",
    "market_reach",
    "impressions",
    "click_through_rate",
    "competition_index",
    "seasonality_index",
    "campaign_duration_days",
]

MODEL_FEATURES = [
    "price", "discount_rate", "market_reach", "impressions",
    "click_through_rate", "competition_index", "seasonality_index",
    "campaign_duration_days",
    "channel_Email", "channel_Influencer", "channel_Social_Media",
    "channel_Tv", "product_category_Stationery", "product_category_Storage"
]

CHANNELS = ["Baseline / Other", "Email", "Influencer", "Social Media", "Tv"]
PRODUCT_CATEGORIES = ["Baseline / Other", "Stationery", "Storage"]

def make_model_input(
    price, discount_rate, market_reach, impressions, click_through_rate,
    competition_index, seasonality_index, campaign_duration_days,
    channel, product_category
):
    numeric_df = pd.DataFrame([{
        "price": price,
        "discount_rate": discount_rate,
        "market_reach": market_reach,
        "impressions": impressions,
        "click_through_rate": click_through_rate,
        "competition_index": competition_index,
        "seasonality_index": seasonality_index,
        "campaign_duration_days": campaign_duration_days,
    }])

    scaled = pd.DataFrame(
        scaler.transform(numeric_df),
        columns=NUMERIC_FEATURES,
        index=numeric_df.index
    )

    # The supplied model was trained with one-hot encoded categorical columns.
    encoded = pd.DataFrame(0.0, index=[0], columns=MODEL_FEATURES[8:])
    if channel != "Baseline / Other":
        encoded[f"channel_{channel.replace(' ', '_')}"] = 1.0
    if product_category != "Baseline / Other":
        encoded[f"product_category_{product_category}"] = 1.0

    return pd.concat([scaled, encoded], axis=1)[MODEL_FEATURES]

def predict_one(values):
    X = make_model_input(**values)
    return float(model.predict(X)[0])

st.title("📈 Sales Revenue Prediction")
st.write("Predict sales revenue using your trained Random Forest regression model.")

with st.sidebar:
    st.header("Model")
    st.success("Random Forest Regressor")
    st.caption("Loaded from random_forest_model.pkl")
    st.caption("Numeric features are transformed with the supplied StandardScaler.")

tab1, tab2 = st.tabs(["🔮 Single Prediction", "ℹ️ Model Info"])

with tab1:

    st.subheader("Enter campaign details")

    c1, c2 = st.columns(2)

    with c1:
        price = st.number_input(
            "Price", min_value=0.0, value=5.65, step=0.5,
            help="Product price."
        )
        discount_rate = st.number_input(
            "Discount Rate", min_value=0.0, max_value=1.0, value=0.23,
            step=0.01, format="%.2f",
            help="Enter as a decimal. Example: 20% = 0.20."
        )
        market_reach = st.number_input(
            "Market Reach", min_value=0.0, value=319.0, step=10.0
        )
        impressions = st.number_input(
            "Impressions", min_value=0.0, value=7309.0, step=100.0
        )

    with c2:
        click_through_rate = st.number_input(
            "Click Through Rate", min_value=0.0, max_value=1.0, value=0.057,
            step=0.001, format="%.3f",
            help="Enter as a decimal. Example: 5% = 0.05."
        )
        competition_index = st.number_input(
            "Competition Index", min_value=0.0, value=4.0, step=0.1
        )
        seasonality_index = st.number_input(
            "Seasonality Index", value=-0.01, step=0.1
        )
        campaign_duration_days = st.number_input(
            "Campaign Duration (days)", min_value=1.0, value=35.0, step=1.0
        )

    c3, c4 = st.columns(2)
    with c3:
        channel = st.selectbox("Marketing Channel", CHANNELS)
    with c4:
        product_category = st.selectbox("Product Category", PRODUCT_CATEGORIES)

    if st.button("🚀 Predict Sales Revenue", type="primary", use_container_width=True):
        values = {
            "price": price,
            "discount_rate": discount_rate,
            "market_reach": market_reach,
            "impressions": impressions,
            "click_through_rate": click_through_rate,
            "competition_index": competition_index,
            "seasonality_index": seasonality_index,
            "campaign_duration_days": campaign_duration_days,
            "channel": channel,
            "product_category": product_category,
        }

        prediction = predict_one(values)

        st.divider()
        st.metric("Predicted Sales Revenue", f"{prediction:,.2f}")
        st.info(
            "The prediction is generated by the Random Forest model you supplied. "
            "The displayed value uses the same model/scaler artifacts and feature order."
        )

with tab2:
    st.subheader("Batch prediction")
    st.write(
        "Upload a CSV containing the 8 numeric features plus optional "
        "`channel` and `product_category` columns."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)

            missing = [c for c in NUMERIC_FEATURES if c not in batch.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                batch_work = batch.copy()

                if "channel" not in batch_work.columns:
                    batch_work["channel"] = "Baseline / Other"
                if "product_category" not in batch_work.columns:
                    batch_work["product_category"] = "Baseline / Other"

                predictions = []
                errors = []

                for idx, row in batch_work.iterrows():
                    try:
                        values = {
                            "price": float(row["price"]),
                            "discount_rate": float(row["discount_rate"]),
                            "market_reach": float(row["market_reach"]),
                            "impressions": float(row["impressions"]),
                            "click_through_rate": float(row["click_through_rate"]),
                            "competition_index": float(row["competition_index"]),
                            "seasonality_index": float(row["seasonality_index"]),
                            "campaign_duration_days": float(row["campaign_duration_days"]),
                            "channel": str(row["channel"]),
                            "product_category": str(row["product_category"]),
                        }
                        predictions.append(predict_one(values))
                        errors.append("")
                    except Exception as e:
                        predictions.append(np.nan)
                        errors.append(str(e))

                batch_work["predicted_sales_revenue"] = predictions
                batch_work["prediction_error"] = errors

                st.dataframe(batch_work, use_container_width=True)

                csv_bytes = batch_work.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=csv_bytes,
                    file_name="sales_revenue_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Could not process the CSV: {e}")


with tab2:
    st.subheader("Model details")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Model:** RandomForestRegressor")
        st.write(f"**Input features:** {getattr(model, 'n_features_in_', 'N/A')}")
        st.write("**Scaler:** StandardScaler")
        st.write(f"**Scaler features:** {getattr(scaler, 'n_features_in_', 'N/A')}")

    with col2:
        st.write("**Numeric features**")
        st.code("\n".join(NUMERIC_FEATURES))
        st.write("**Encoded categorical features**")
        st.code("\n".join(MODEL_FEATURES[8:]))

    st.warning(
        "The supplied model contains one-hot columns for Email, Influencer, "
        "Social_Media, Tv, Stationery and Storage. 'Baseline / Other' leaves "
        "those dummy columns at 0, matching the dropped-reference-category "
        "encoding used by the trained model."
    )

st.caption("Sales Revenue Prediction • Streamlit + Scikit-learn")
