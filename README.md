# Sales Revenue Prediction – Streamlit

## Files
- `app.py` – Streamlit application
- `random_forest_model.pkl` – supplied Random Forest model
- `scaler (2).pkl` – supplied StandardScaler
- `predictions(3).csv` – supplied prediction output/reference CSV
- `requirements.txt` – Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app supports:
1. Single sales-revenue prediction
2. Model/scaler information

### Required numeric columns for batch CSV

`price, discount_rate, market_reach, impressions, click_through_rate, competition_index, seasonality_index, campaign_duration_days`

Optional categorical columns:

`channel, product_category`

Discount rate and click-through rate are entered as decimals, e.g. 0.20 = 20%.
