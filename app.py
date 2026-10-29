
from pathlib import Path


import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from generate_dataset import generate_dataset

st.set_page_config(
    page_title="EstateIQ AI",
    page_icon="🏡",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "house_prices.csv"

FEATURES = [
    "area_sqft", "bedrooms", "bathrooms", "age_years",
    "location", "parking", "furnishing",
]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "parking"]
CATEGORICAL = ["location", "furnishing"]
TARGET = "price_lakh"

LOCATIONS = [
    "City Center", "Market", "Near Railway station", "Near Metro", "IT Hub"
]
FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]

MODELS = [
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "Linear Regression",
]

# ---------------- SESSION STATE ----------------

defaults = {
    "theme": "Light",
    "history": [],
    "favorites": [],
    "last_prediction": None,
    "page": "Dashboard",
    "dataset_version": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

dark = st.session_state.theme == "Dark"

BG = "#0B1220" if dark else "#F3F6FC"
SURFACE = "#141F32" if dark else "#FFFFFF"
TEXT = "#EDF3FF" if dark else "#162544"
MUTED = "#A2B1CA" if dark else "#72819A"
ACCENT = "#79A7FF" if dark else "#3478F6"

st.markdown(f"""
<style>
.stApp {{
    background: {BG};
    color: {TEXT};
}}
.block-container {{
    max-width: 1500px;
    padding-top: 1.5rem;
}}
[data-testid="stSidebar"] {{
    background: #081426;
}}
[data-testid="stSidebar"] * {{
    color: #F4F7FF;
}}
.hero {{
    background: linear-gradient(120deg, {SURFACE}, {BG});
    border: 1px solid {ACCENT};
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
}}
.hero h1 {{
    font-size: 35px;
    font-weight: 800;
}}
.panel {{
    background: {SURFACE};
    border-radius: 16px;
    padding: 18px;
    border: 1px solid {ACCENT};
    margin-bottom: 16px;
}}
div.stButton > button,
div.stFormSubmitButton > button {{
    border-radius: 10px;
    min-height: 42px;
    font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------

@st.cache_data(show_spinner=False)
def load_data(version):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        generate_dataset().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    for col in NUMERIC + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required)

    if len(df) < 10:
        raise ValueError("At least 10 valid records are required.")

    return df


# ---------------- MODEL PIPELINE ----------------

def build_model(name):
    preprocessing = ColumnTransformer([
        ("numbers", "passthrough", NUMERIC),
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    if name == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "Extra Trees":
        estimator = ExtraTreesRegressor(
            n_estimators=150,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    elif name == "Gradient Boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    else:
        estimator = LinearRegression()

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(version):
    df = load_data(version)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
    )

    trained = {}
    results = []

    for name in MODELS:
        model = build_model(name)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        trained[name] = model

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        })

    metrics = pd.DataFrame(results).sort_values("RMSE")

    best = metrics.iloc[0]["Model"]

    return trained, metrics, best, X_test, y_test


try:
    df = load_data(st.session_state.dataset_version)

    models, metrics, best_model, X_test, y_test = train_models(
        st.session_state.dataset_version
    )

except Exception as error:
    st.error(f"Could not load data or train models: {error}")
    st.stop()


# ---------------- HELPER FUNCTIONS ----------------

def header(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def predict_price(
    area, bedrooms, bathrooms, age,
    location, parking, furnishing, model_name
):
    sample = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location": location,
        "parking": parking,
        "furnishing": furnishing,
    }], columns=FEATURES)

    prediction = float(models[model_name].predict(sample)[0])

    return sample, prediction


def record_prediction(record):
    st.session_state.history.append(record)
    st.session_state.history = st.session_state.history[-100:]


def navigate(page):
    st.session_state.page = page
    st.rerun()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🏡 EstateIQ AI")
    st.caption("SMART PROPERTY PORTAL")

  
    pages = [
        "Dashboard",
        "Predict Price",
        "Property Map",
        "Compare Properties",
        "Affordability",
        "Dataset Explorer",
        "Model Analytics",
        "Favorites & History",
        "About Project",
    ]


    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page),
    )

    if selected != st.session_state.page:
        st.session_state.page = selected
        st.rerun()

    if st.button("🌗 Switch Theme", width="stretch"):
        st.session_state.theme = (
            "Dark" if not dark else "Light"
        )
        st.rerun()

    st.divider()
    st.caption(f"Records: {len(df):,}")
    st.caption(f"Algorithms: {len(MODELS)}")
    st.caption(f"Favorites: {len(st.session_state.favorites)}")
    st.caption(f"Predictions: {len(st.session_state.history)}")


page = st.session_state.page


# ---------------- DASHBOARD ----------------

if page == "Dashboard":
    header(
        "Welcome to EstateIQ AI 🏡",
        "Explore property prices with machine learning.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Houses", f"{len(df):,}")
    c2.metric("Average Price", f"₹{df[TARGET].mean():.2f} L")
    c3.metric("Minimum Price", f"₹{df[TARGET].min():.2f} L")
    c4.metric("Maximum Price", f"₹{df[TARGET].max():.2f} L")

    left, right = st.columns(2)

    with left:
        st.subheader("📊 House Price Distribution")

        fig, ax = plt.subplots()
        ax.hist(df[TARGET], bins=25)
        ax.set_xlabel("Price (₹ lakh)")
        ax.set_ylabel("Number of houses")
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with right:
        st.subheader("📍 Average Price by Location")

        st.bar_chart(df.groupby("location")[TARGET].mean())

    st.subheader("🏆 Model Leaderboard")

    display_metrics = metrics.rename(columns={
        "MAE": "MAE (₹ lakh)",
        "RMSE": "RMSE (₹ lakh)",
        "R2": "R² Score",
    })

    st.dataframe(
        display_metrics.round(3),
        width="stretch",
        hide_index=True,
    )

    st.success(f"Best model on this test split: {best_model}")

    a, b, c = st.columns(3)

    if a.button("🏠 Predict Price", width="stretch"):
        navigate("Predict Price")

    if b.button("⚖️ Compare Homes", width="stretch"):
        navigate("Compare Properties")

    if c.button("📈 Model Analytics", width="stretch"):
        navigate("Model Analytics")


# ---------------- PREDICT PRICE ----------------

elif page == "Predict Price":
    header(
        "House Price Prediction",
        "Enter your property details to estimate its value.",
    )

    left, right = st.columns(2)

    with left:
        with st.form("prediction_form"):
            area = st.number_input(
                "Built-up area (sqft)",
                300, 10000, 1500, step=50,
            )

            c1, c2 = st.columns(2)

            with c1:
                bedrooms = st.selectbox(
                    "Bedrooms", [1, 2, 3, 4, 5, 6], index=2
                )
                bathrooms = st.selectbox(
                    "Bathrooms", [1, 2, 3, 4, 5], index=1
                )
                parking = st.selectbox(
                    "Parking spaces", [0, 1, 2, 3], index=1
                )

            with c2:
                age = st.slider("House age", 0, 100, 5)
                location = st.selectbox("Location", LOCATIONS)
                furnishing = st.selectbox(
                    "Furnishing", FURNISHING
                )

            algorithm = st.selectbox(
                "Machine Learning Algorithm",
                MODELS,
                index=MODELS.index(best_model),
            )

            submitted = st.form_submit_button(
                "✨ Predict House Price",
                width="stretch",
            )

    if submitted:
        _, prediction = predict_price(
            area, bedrooms, bathrooms, age,
            location, parking, furnishing, algorithm,
        )

        record = {
            "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "Area (sqft)": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Age": age,
            "Location": location,
            "Parking": parking,
            "Furnishing": furnishing,
            "Algorithm": algorithm,
            "Price (₹ lakh)": round(prediction, 2),
        }

        st.session_state.last_prediction = record
        record_prediction(record)

    with right:
        st.subheader("💎 Prediction Result")

        result = st.session_state.last_prediction

        if result:
            price = result["Price (₹ lakh)"]

            st.metric(
                "Estimated House Price",
                f"₹{price:,.2f} lakh",
            )

            st.metric(
                "Estimated Price per Sqft",
                f"₹{price * 100000 / result['Area (sqft)']:,.0f}",
            )

            st.write(f"**Location:** {result['Location']}")
            st.write(f"**Area:** {result['Area (sqft)']:,} sqft")
            st.write(f"**Bedrooms:** {result['Bedrooms']}")
            st.write(f"**Algorithm:** {result['Algorithm']}")

            if st.button("⭐ Save to Favorites", width="stretch"):
                if result not in st.session_state.favorites:
                    st.session_state.favorites.append(dict(result))
                    st.success("Property saved!")
                else:
                    st.info("This property is already saved.")

            similar = df[df["location"] == result["Location"]].copy()

            if not similar.empty:
                similar["area_difference"] = (
                    similar["area_sqft"] - result["Area (sqft)"]
                ).abs()

                similar = similar.nsmallest(5, "area_difference")

                st.subheader("Similar Sample Properties")

                st.dataframe(
                    similar[
                        ["area_sqft", "bedrooms", "bathrooms",
                         "furnishing", TARGET]
                    ].round(2),
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.info("Submit the form to calculate a prediction.")

    st.caption(
        "Educational estimates only. The synthetic dataset does not "
        "represent verified real-estate market prices."
    )


# ---------------- COMPARE PROPERTIES ----------------

elif page == "Compare Properties":
    header(
        "Compare Properties",
        "Estimate and compare three homes using the same algorithm.",
    )

    with st.form("compare_form"):
        algorithm = st.selectbox(
            "Algorithm",
            MODELS,
            index=MODELS.index(best_model),
        )

        profiles = []
        columns = st.columns(3)

        for i, col in enumerate(columns):
            with col:
                st.subheader(f"Property {i + 1}")

                area = st.number_input(
                    "Area (sqft)",
                    300, 10000,
                    [1200, 1600, 2200][i],
                    50,
                    key=f"area_{i}",
                )

                bedrooms = st.selectbox(
                    "Bedrooms", [1, 2, 3, 4, 5, 6],
                    index=min(i + 1, 5),
                    key=f"beds_{i}",
                )

                bathrooms = st.selectbox(
                    "Bathrooms", [1, 2, 3, 4, 5],
                    key=f"baths_{i}",
                )

                age = st.slider(
                    "Age", 0, 100, [5, 10, 2][i],
                    key=f"age_{i}",
                )

                location = st.selectbox(
                    "Location", LOCATIONS,
                    index=i,
                    key=f"location_{i}",
                )

                parking = st.selectbox(
                    "Parking", [0, 1, 2, 3],
                    key=f"parking_{i}",
                )

                furnishing = st.selectbox(
                    "Furnishing", FURNISHING,
                    key=f"furnishing_{i}",
                )

                profiles.append((
                    area, bedrooms, bathrooms, age,
                    location, parking, furnishing,
                ))

        compare = st.form_submit_button(
            "⚖️ Compare Properties",
            width="stretch",
        )

    if compare:
        rows = []

        for i, profile in enumerate(profiles, start=1):
            _, price = predict_price(*profile, algorithm)

            rows.append({
                "Property": f"Property {i}",
                "Area (sqft)": profile[0],
                "Location": profile[4],
                "Bedrooms": profile[1],
                "Estimated Price (₹ lakh)": round(price, 2),
                "Price per Sqft (₹)": round(
                    price * 100000 / profile[0]
                ),
            })

        comparison = pd.DataFrame(rows)

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

        st.bar_chart(
            comparison.set_index("Property")[
                ["Estimated Price (₹ lakh)"]
            ]
        )

        st.download_button(
            "⬇️ Download Comparison",
            comparison.to_csv(index=False).encode("utf-8"),
            file_name="property_comparison.csv",
            mime="text/csv",
            width="stretch",
        )

# ---------------- AFFORDABILITY CALCULATOR ----------------

elif page == "Affordability":
    header(
        "Property Affordability Calculator",
        "Estimate EMI, total interest, and repayment affordability.",
    )

    left, right = st.columns(2)

    with left:
        price = st.number_input(
            "Property Price (₹ lakh)", 5.0, 1000.0, 75.0, 5.0
        )
        down = st.slider("Down Payment (%)", 10, 80, 20)
        rate = st.slider("Annual Interest Rate (%)", 0.0, 18.0, 8.5, 0.25)
        years = st.slider("Loan Tenure (years)", 1, 30, 20)
        income = st.number_input(
            "Monthly Income (₹)", 10000, 10000000, 100000, 5000
        )

    with right:
        loan = price * 100000 * (1 - down / 100)
        months = years * 12
        monthly_rate = rate / 1200

        if monthly_rate == 0:
            emi = loan / months
        else:
            emi = (
                loan * monthly_rate * (1 + monthly_rate) ** months
                / ((1 + monthly_rate) ** months - 1)
            )

        total_payment = emi * months
        interest = total_payment - loan
        ratio = emi / income * 100

        st.metric("Loan Amount", f"₹{loan:,.0f}")
        st.metric("Monthly EMI", f"₹{emi:,.0f}")
        st.metric("Total Interest", f"₹{interest:,.0f}")
        st.metric("Total Payment", f"₹{total_payment:,.0f}")
        st.metric("EMI / Income", f"{ratio:.1f}%")

        if ratio <= 30:
            st.success("EMI is below 30% of monthly income.")
        elif ratio <= 40:
            st.warning("EMI uses a substantial share of income.")
        else:
            st.error("EMI is high relative to monthly income.")

    st.caption(
        "Illustrative calculation only. Actual loan terms, fees, taxes, "
        "insurance, and lender rules may differ."
    )


# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":
    header(
        "Dataset Explorer",
        "Search, filter, analyze, and download housing records.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", len(df))
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    a, b, c = st.columns(3)

    with a:
        location_filter = st.selectbox(
            "Location", ["All"] + sorted(df["location"].unique())
        )

    with b:
        furnishing_filter = st.selectbox(
            "Furnishing", ["All"] + sorted(df["furnishing"].unique())
        )

    with c:
        max_price = st.slider(
            "Maximum Price (₹ lakh)",
            float(df[TARGET].min()),
            float(df[TARGET].max()),
            float(df[TARGET].max()),
        )

    filtered = df[df[TARGET] <= max_price]

    if location_filter != "All":
        filtered = filtered[
            filtered["location"] == location_filter
        ]

    if furnishing_filter != "All":
        filtered = filtered[
            filtered["furnishing"] == furnishing_filter
        ]

    st.write(f"Matching records: {len(filtered)}")

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_house_prices.csv",
        mime="text/csv",
        width="stretch",
    )

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), width="stretch")

    st.subheader("Price by Furnishing")
    st.bar_chart(df.groupby("furnishing")[TARGET].mean())


# ---------------- MODEL ANALYTICS ----------------

elif page == "Model Analytics":
    header(
        "Machine Learning Analytics",
        "Compare model accuracy and inspect feature importance.",
    )

    st.subheader("Algorithm Comparison")

    st.dataframe(
        metrics.round(4),
        width="stretch",
        hide_index=True,
    )

    st.success(f"Lowest test RMSE: {best_model}")

    metric = st.selectbox("Select Metric", ["RMSE", "MAE", "R2"])

    st.bar_chart(metrics.set_index("Model")[[metric]])

    chosen = st.selectbox(
        "Inspect Algorithm",
        MODELS,
        index=MODELS.index(best_model),
    )

    predictions = models[chosen].predict(X_test)

    comparison = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": predictions,
    })

    comparison["Error"] = (
        comparison["Predicted Price"] - comparison["Actual Price"]
    )

    st.subheader("Actual vs Predicted Prices")

    fig, ax = plt.subplots()
    ax.scatter(
        comparison["Actual Price"],
        comparison["Predicted Price"],
        alpha=0.6,
    )

    low = min(
        comparison["Actual Price"].min(),
        comparison["Predicted Price"].min(),
    )
    high = max(
        comparison["Actual Price"].max(),
        comparison["Predicted Price"].max(),
    )

    ax.plot([low, high], [low, high], "--")
    ax.set_xlabel("Actual Price (₹ lakh)")
    ax.set_ylabel("Predicted Price (₹ lakh)")

    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.subheader("Prediction Error Distribution")

    fig, ax = plt.subplots()
    ax.hist(comparison["Error"], bins=20)
    ax.set_xlabel("Prediction Error (₹ lakh)")
    ax.set_ylabel("Count")
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    if chosen in ["Random Forest", "Extra Trees", "Gradient Boosting"]:
        pipeline = models[chosen]
        names = pipeline.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        importance = pipeline.named_steps[
            "model"
        ].feature_importances_

        importance_df = pd.DataFrame({
            "Feature": names,
            "Importance": importance,
        }).sort_values("Importance", ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance_df.head(15).set_index("Feature"))

    st.dataframe(
        comparison.round(3),
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download Test Results",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_test_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Metrics use a held-out test set. Synthetic-data performance "
        "does not establish real-market prediction accuracy."
    )


# ---------------- FAVORITES AND HISTORY ----------------

elif page == "Favorites & History":
    header(
        "Favorites & Prediction History",
        "Review and download your saved estimates.",
    )

    left, right = st.columns(2)

    with left:
        st.subheader("⭐ Favorites")

        if st.session_state.favorites:
            favorites = pd.DataFrame(st.session_state.favorites)

            st.dataframe(
                favorites,
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download Favorites",
                favorites.to_csv(index=False).encode("utf-8"),
                file_name="favorites.csv",
                mime="text/csv",
            )

            if st.button("Clear Favorites"):
                st.session_state.favorites = []
                st.rerun()
        else:
            st.info("No favorites saved yet.")

    with right:
        st.subheader("🕘 Prediction History")

        if st.session_state.history:
            history = pd.DataFrame(st.session_state.history)

            st.dataframe(
                history.iloc[::-1],
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "Download History",
                history.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
            )

            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No predictions yet.")


# =========================================================
# CHANDIGARH PROPERTY MAP
# =========================================================

elif page == "Property Map":
    header(
        "Chandigarh Property Explorer",
        "Explore residential societies on satellite imagery "
        "and compare educational ML price estimates.",
    )

    # Society names and addresses sourced from public listings.
    # Geocoded coordinates are obtained at runtime.
    societies = pd.DataFrame([
        {
            "Society": "New Light Co-operative Society",
            "Area": "Sector 51",
            "Address": "New Light Co-operative Society, "
                       "Sector 51, Chandigarh, India",
        },
        {
            "Society": "Pink Rose Enclave",
            "Area": "Sector 49-D",
            "Address": "Pink Rose Enclave, Sukhna Path, "
                       "Sector 49-D, Chandigarh, India",
        },
        {
            "Society": "SBI Officers Society",
            "Area": "Sector 49-D",
            "Address": "SBI Officers Society, Sector 49-D, "
                       "Chandigarh, India",
        },
        {
            "Society": "RCS & CPS Society",
            "Area": "Sector 48-A",
            "Address": "RCS & CPS Society, Sector 48-A, "
                       "Chandigarh, India",
        },
        {
            "Society": "Universal Enclave",
            "Area": "Sector 48-B",
            "Address": "Universal Enclave, Sector 48-B, "
                       "Chandigarh, India",
        },
        {
            "Society": "Progressive Enclave",
            "Area": "Sector 50-B",
            "Address": "Progressive Enclave, Sector 50-B, "
                       "Chandigarh, India",
        },
        {
            "Society": "IAS/IPS Society",
            "Area": "Sector 49-A",
            "Address": "IAS IPS Society, Sukhna Path, "
                       "Sector 49-A, Chandigarh, India",
        },
        {
            "Society": "Ajanta Enclave Society",
            "Area": "Sector 51-D",
            "Address": "Ajanta Enclave Society, Sarovar Path, "
                       "Sector 51, Chandigarh, India",
        },
        {
            "Society": "The Shivalik Cooperative House "
                       "Building Society",
            "Area": "Sector 49-A",
            "Address": "Shivalik Cooperative House Building "
                       "Society, Sector 49-A, Chandigarh, India",
        },
        {
            "Society": "Chandigarh Housing Board Flats",
            "Area": "Sector 63",
            "Address": "Chandigarh Housing Board Flats, "
                       "Sector 63, Chandigarh, India",
        },
    ])

    st.info(
        "Satellite imagery is provided by Esri. Society locations "
        "are geocoded from public address information and may "
        "need manual verification. Model prices are educational "
        "estimates, not verified local sale prices."
    )

    # ---------------- MAP FILTERS ----------------

    st.subheader("🔎 Find a Society")

    f1, f2 = st.columns(2)

    with f1:
        areas = ["All Areas"] + sorted(societies["Area"].unique())
        selected_area = st.selectbox("Filter by Area", areas)

    area_data = societies.copy()

    if selected_area != "All Areas":
        area_data = area_data[area_data["Area"] == selected_area]

    with f2:
        society_options = ["All Societies"] + sorted(
            area_data["Society"].tolist()
        )
        selected_society = st.selectbox(
            "Filter by Society",
            society_options,
        )

    filtered_societies = area_data.copy()

    if selected_society != "All Societies":
        filtered_societies = filtered_societies[
            filtered_societies["Society"] == selected_society
        ]

    st.caption(
        f"Showing {len(filtered_societies)} societies "
        f"across {filtered_societies['Area'].nunique()} areas."
    )

    # ---------------- PRICE ESTIMATE SETTINGS ----------------

    with st.expander("⚙️ Configure illustrative property estimate"):
        p1, p2, p3 = st.columns(3)

        with p1:
            map_area_sqft = st.number_input(
                "Property area (sqft)",
                min_value=300,
                max_value=10000,
                value=1500,
                step=50,
                key="map_area_sqft",
            )

            map_bedrooms = st.selectbox(
                "Bedrooms",
                [1, 2, 3, 4, 5, 6],
                index=2,
                key="map_bedrooms",
            )

        with p2:
            map_bathrooms = st.selectbox(
                "Bathrooms",
                [1, 2, 3, 4, 5],
                index=1,
                key="map_bathrooms",
            )

            map_age = st.slider(
                "Property age (years)",
                0,
                100,
                5,
                key="map_age",
            )

        with p3:
            map_parking = st.selectbox(
                "Parking spaces",
                [0, 1, 2, 3],
                index=1,
                key="map_parking",
            )

            map_furnishing = st.selectbox(
                "Furnishing",
                FURNISHING,
                key="map_furnishing",
            )

        map_model = st.selectbox(
            "Prediction algorithm",
            MODELS,
            index=MODELS.index(best_model),
            key="map_model",
        )

        map_location_category = st.selectbox(
            "Training dataset location category",
            LOCATIONS,
            key="map_location_category",
            help=(
                "Your existing model has broad location categories, "
                "not verified prices for individual societies."
            ),
        )

    # All mapped societies use the same entered property profile.
    # The current dataset cannot distinguish their real market prices.
    model_input = pd.DataFrame([{
        "area_sqft": map_area_sqft,
        "bedrooms": map_bedrooms,
        "bathrooms": map_bathrooms,
        "age_years": map_age,
        "location": map_location_category,
        "parking": map_parking,
        "furnishing": map_furnishing,
    }], columns=FEATURES)

    estimated_price = float(
        models[map_model].predict(model_input)[0]
    )

    st.metric(
        "Illustrative Model Estimate",
        f"₹{estimated_price:,.2f} lakh",
    )

    st.caption(
        "This is one model estimate for your selected property profile. "
        "It is not a separate verified valuation for each society."
    )

    # ---------------- GEOCODING ----------------

    @st.cache_data(
        ttl=60 * 60 * 24 * 30,
        show_spinner=False,
    )
    def geocode_societies(addresses):
        geocoder = Nominatim(
            user_agent="EstateIQ_Chandigarh_Property_Explorer"
        )

        geocode = RateLimiter(
            geocoder.geocode,
            min_delay_seconds=1.1,
            max_retries=1,
            error_wait_seconds=2,
            swallow_exceptions=True,
        )

        results = []

        for address in addresses:
            try:
                result = geocode(
                    address,
                    exactly_one=True,
                    timeout=12,
                    country_codes="in",
                )

                if result:
                    results.append({
                        "Address": address,
                        "Latitude": result.latitude,
                        "Longitude": result.longitude,
                        "Matched Address": result.address,
                        "Geocoded": True,
                    })
                else:
                    results.append({
                        "Address": address,
                        "Latitude": None,
                        "Longitude": None,
                        "Matched Address": "",
                        "Geocoded": False,
                    })

            except Exception:
                results.append({
                    "Address": address,
                    "Latitude": None,
                    "Longitude": None,
                    "Matched Address": "",
                    "Geocoded": False,
                })

        return pd.DataFrame(results)

    # Trigger geocoding only when requested.
    if st.button(
        "📍 Load / Refresh Society Locations",
        width="stretch",
        key="load_chandigarh_societies",
    ):
        geocode_societies.clear()
        st.session_state["chd_map_loaded"] = True

    if (
        "chd_map_loaded" not in st.session_state
        or not st.session_state["chd_map_loaded"]
    ):
        st.info(
            "Click 'Load / Refresh Society Locations' to locate "
            "the listed societies. Internet access is required."
        )

    else:
        addresses = tuple(societies["Address"].tolist())

        with st.spinner("Finding society locations..."):
            locations_df = geocode_societies(addresses)

        map_data = societies.merge(
            locations_df,
            on="Address",
            how="left",
        )

        map_data = map_data[
            map_data["Geocoded"].fillna(False)
        ].copy()

        map_data = map_data[
            map_data["Society"].isin(
                filtered_societies["Society"]
            )
        ].copy()

        if map_data.empty:
            st.warning(
                "No matching society addresses could be located. "
                "Try again later or verify the addresses manually."
            )

        else:
            # Esri World Imagery satellite tiles.
            property_map = folium.Map(
                location=[30.7333, 76.7794],
                zoom_start=12,
                tiles=None,
                control_scale=True,
            )

            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/ArcGIS/rest/"
                    "services/World_Imagery/MapServer/tile/"
                    "{z}/{y}/{x}"
                ),
                attr=(
                    "Tiles © Esri — Source: Esri, Maxar, Earthstar "
                    "Geographics, and the GIS User Community"
                ),
                name="Satellite Imagery",
                overlay=False,
                control=True,
            ).add_to(property_map)

            for _, row in map_data.iterrows():
                popup_html = f"""
                <div style="width:250px;font-family:Arial">
                    <h4>{row['Society']}</h4>
                    <b>Area:</b> {row['Area']}<br>
                    <b>Address:</b> {row['Matched Address']}<br>
                    <b>Entered property size:</b>
                    {map_area_sqft:,} sqft<br>
                    <b>Model:</b> {map_model}<br>
                    <b>Illustrative estimate:</b>
                    ₹{estimated_price:,.2f} lakh<br>
                    <hr>
                    <small>
                    Estimate uses a common property profile;
                    not a verified society valuation.
                    </small>
                </div>
                """

                folium.Marker(
                    location=[
                        row["Latitude"],
                        row["Longitude"],
                    ],
                    tooltip=f"{row['Society']} · {row['Area']}",
                    popup=folium.Popup(
                        popup_html,
                        max_width=320,
                    ),
                    icon=folium.Icon(
                        color="blue",
                        icon="home",
                        prefix="fa",
                    ),
                ).add_to(property_map)

            folium.LayerControl().add_to(property_map)

            st.subheader("🛰️ Chandigarh Satellite Map")

            map_result = st_folium(
                property_map,
                width=None,
                height=600,
                returned_objects=["last_object_clicked"],
                key="chandigarh_property_map",
            )

            st.caption(
                "Click a home marker to view its society, area, "
                "matched address, and illustrative price estimate."
            )

            # ---------------- SOCIETY DIRECTORY ----------------

            st.subheader("🏘️ Society Directory")

            display_columns = [
                "Society",
                "Area",
                "Matched Address",
                "Latitude",
                "Longitude",
            ]

            st.dataframe(
                map_data[display_columns].round({
                    "Latitude": 6,
                    "Longitude": 6,
                }),
                width="stretch",
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download Mapped Societies CSV",
                map_data[display_columns].to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="chandigarh_societies_map.csv",
                mime="text/csv",
                width="stretch",
            )

            st.caption(
                "Geocoding results are approximate search matches. "
                "Verify the precise entrance or building location "
                "before visiting or making property decisions."
            )

# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":
    header(
        "About EstateIQ AI",
        "Machine Learning House Price Prediction Mini Project.",
    )

    st.subheader("Algorithms Used")

    st.markdown("""
    **1. Linear Regression**
    - Fast baseline algorithm.
    - Models linear relationships.

    **2. Random Forest**
    - Combines multiple decision trees.
    - Supports nonlinear relationships.
    - Uses parallel tree training.

    **3. Extra Trees**
    - Uses randomized tree splits.
    - Can provide efficient ensemble predictions.

    **4. Gradient Boosting**
    - Builds trees sequentially to correct previous errors.
    - Can learn complex relationships.

    ### Efficiency Improvements
    - Cached dataset loading.
    - Cached model training.
    - Shared preprocessing pipeline.
    - Parallel processing for Random Forest and Extra Trees.
    - Automatic model ranking using test RMSE.

    ### Workflow
    1. Load and validate the dataset.
    2. Separate input features and target.
    3. Encode categorical variables.
    4. Split training and testing data.
    5. Train four algorithms.
    6. Evaluate MAE, RMSE and R².
    7. Predict and compare property prices.
    """)

    st.warning(
        "The supplied dataset is synthetic. Replace it with verified "
        "transaction data before drawing conclusions about actual prices."
    )

    if st.button("🔄 Retrain All Models", width="stretch"):
        st.session_state.dataset_version += 1
        load_data.clear()
        train_models.clear()
        st.session_state.last_prediction = None
        st.rerun()


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    f"EstateIQ AI | Python · Pandas · Scikit-learn · Streamlit | "
    f"Theme: {st.session_state.theme}"
)
