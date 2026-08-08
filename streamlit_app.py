import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Algerian Forest Fire Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------
       GLOBAL
    -------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(34, 197, 94, 0.12),
                transparent 35%
            ),
            radial-gradient(
                circle at top left,
                rgba(22, 163, 74, 0.10),
                transparent 30%
            ),
            #06110b;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent;
    }


    /* -------------------------------
       SIDEBAR
    -------------------------------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07150d 0%,
                #0b1f12 100%
            );

        border-right: 1px solid rgba(74, 222, 128, 0.15);
    }


    /* -------------------------------
       HERO
    -------------------------------- */

    .hero {
        padding: 42px 45px;
        border-radius: 26px;
        margin-bottom: 30px;

        background:
            linear-gradient(
                135deg,
                rgba(7, 30, 16, 0.96),
                rgba(15, 46, 27, 0.90)
            );

        border: 1px solid rgba(74, 222, 128, 0.22);

        box-shadow:
            0 25px 60px rgba(0, 0, 0, 0.35);
    }


    .hero-badge {
        display: inline-block;

        padding: 8px 15px;

        border-radius: 30px;

        background: rgba(34, 197, 94, 0.12);

        border: 1px solid rgba(74, 222, 128, 0.25);

        color: #86efac;

        font-size: 13px;

        font-weight: 700;

        letter-spacing: 0.5px;

        margin-bottom: 16px;
    }


    .hero h1 {
        font-size: 46px;

        font-weight: 800;

        margin: 0;

        color: #f0fdf4;
    }


    .hero h1 span {
        color: #4ade80;
    }


    .hero p {
        margin-top: 15px;

        color: #a7f3d0;

        font-size: 17px;

        line-height: 1.7;

        max-width: 850px;
    }


    /* -------------------------------
       INFO CARDS
    -------------------------------- */

    .info-card {
        padding: 24px;

        border-radius: 20px;

        background:
            rgba(7, 30, 16, 0.78);

        border:
            1px solid rgba(74, 222, 128, 0.15);

        min-height: 135px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.20);
    }


    .info-card h3 {
        color: #f0fdf4;

        margin-top: 0;

        margin-bottom: 10px;
    }


    .info-card p {
        color: #9ca3af;

        line-height: 1.6;

        margin: 0;
    }


    /* -------------------------------
       INPUT SECTION
    -------------------------------- */

    .section-title {
        font-size: 27px;

        font-weight: 750;

        color: #f0fdf4;

        margin-top: 35px;

        margin-bottom: 8px;
    }


    .section-subtitle {
        color: #86a996;

        font-size: 15px;

        margin-bottom: 22px;
    }


    /* -------------------------------
       RESULT
    -------------------------------- */

    .result-card {
        padding: 38px;

        border-radius: 25px;

        background:
            linear-gradient(
                135deg,
                rgba(20, 83, 45, 0.30),
                rgba(6, 30, 15, 0.90)
            );

        border:
            1px solid rgba(74, 222, 128, 0.35);

        text-align: center;

        margin-top: 30px;

        margin-bottom: 30px;

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.30);
    }


    .result-title {
        color: #86efac;

        font-size: 17px;

        margin-bottom: 10px;
    }


    .result-score {
        color: #4ade80;

        font-size: 64px;

        font-weight: 850;

        line-height: 1;
    }


    .result-label {
        color: #d1fae5;

        font-size: 18px;

        margin-top: 13px;
    }


    /* -------------------------------
       METRICS
    -------------------------------- */

    [data-testid="stMetric"] {
        background:
            rgba(7, 30, 16, 0.75);

        border:
            1px solid rgba(74, 222, 128, 0.15);

        padding: 20px;

        border-radius: 18px;
    }


    [data-testid="stMetricValue"] {
        color: #4ade80;
    }


    /* -------------------------------
       BUTTON
    -------------------------------- */

    .stButton > button {

        width: 100%;

        border-radius: 13px;

        border: 1px solid rgba(74, 222, 128, 0.35);

        background:
            linear-gradient(
                135deg,
                #16a34a,
                #22c55e
            );

        color: white;

        font-weight: 750;

        padding: 12px;

        transition: 0.2s;
    }


    .stButton > button:hover {

        border-color: #86efac;

        background:
            linear-gradient(
                135deg,
                #15803d,
                #16a34a
            );

        transform: translateY(-1px);

        box-shadow:
            0 8px 25px rgba(34, 197, 94, 0.25);
    }


    /* -------------------------------
       INPUTS
    -------------------------------- */

    .stNumberInput input {

        background:
            rgba(15, 23, 42, 0.75);

        border:
            1px solid rgba(74, 222, 128, 0.15);

        border-radius: 10px;

        color: white;
    }


    /* -------------------------------
       DIVIDER
    -------------------------------- */

    hr {

        border-color:
            rgba(74, 222, 128, 0.12);
    }


    /* -------------------------------
       FOOTER
    -------------------------------- */

    .custom-footer {

        text-align: center;

        padding: 30px;

        color: #6b8f78;

        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "ridgeregression.pkl"
)

SCALER_PATH = os.path.join(
    "models",
    "Scaler.pkl"
)


@st.cache_resource
def load_artifacts():

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


try:

    ridge_model, standard_scaler = load_artifacts()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error(
        "Unable to load the model files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    
        st.html("""
        <div style="
            text-align:center;
            padding:15px 5px 25px 5px;
        ">
    
            <div style="
                font-size:52px;
            ">
                🔥
            </div>
    
            <h2 style="
                color:#f0fdf4;
                margin-bottom:5px;
            ">
                Forest Fire
            </h2>
    
            <p style="
                color:#86a996;
                font-size:14px;
            ">
                FWI Prediction System
            </p>
    
        </div>
        """)
    
        st.markdown("---")
    
        st.markdown("""
    ### 🌲 About
    
    This application predicts the
    **Fire Weather Index (FWI)** using
    environmental and weather-related
    parameters.
    
    ### 🤖 Model
    
    **Ridge Regression**
    
    ### 🎯 Target
    
    **Fire Weather Index (FWI)**
    
    ### ⚙️ Pipeline
    
    Data → Scaling → Ridge Regression → FWI
    """)
    
        st.markdown("---")
    
        st.caption(
            "Built with Python • Scikit-Learn • Streamlit"
        )

    st.markdown("---")

    st.markdown(
        """
        ### 🌲 About

        This application predicts the
        **Fire Weather Index (FWI)** using
        environmental and weather-related
        parameters.

        ### 🤖 Model

        **Ridge Regression**

        ### 🎯 Target

        **Fire Weather Index (FWI)**

        ### ⚙️ Pipeline

        Data → Scaling → Ridge Regression → FWI
        """,
    )

    st.markdown("---")

    st.caption(
        "Built with Python • Scikit-Learn • Streamlit"
    )


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-badge">
            🔥 MACHINE LEARNING • FOREST FIRE ANALYTICS
        </div>

        <h1>
            Algerian Forest Fire
            <span>Predictor</span>
        </h1>

        <p>
            Estimate the Fire Weather Index (FWI)
            using meteorological and environmental
            conditions from the Algerian Forest Fires
            dataset.
        </p>

    </div>
    """
)


# ============================================================
# INFORMATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.html(
        """
        <div class="info-card">

            <h3>🎯 Target</h3>

            <p>
                Fire Weather Index (FWI)
                indicating the potential severity
                of fire weather conditions.
            </p>

        </div>
        """
    )


with col2:

    st.html(
        """
        <div class="info-card">

            <h3>🌡️ Weather Data</h3>

            <p>
                Temperature, humidity, wind,
                rainfall and fire-weather
                indicators.
            </p>

        </div>
        """
    )


with col3:

    st.html(
        """
        <div class="info-card">

            <h3>🤖 Model</h3>

            <p>
                Ridge Regression with
                StandardScaler preprocessing.
            </p>

        </div>
        """
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        🌲 Environmental Conditions
    </div>

    <div class="section-subtitle">
        Enter the weather and fire-weather parameters
        to estimate the Fire Weather Index.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    Temperature = st.number_input(
        "🌡️ Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=30.0,
        step=0.1
    )

    RH = st.number_input(
        "💧 Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

    Ws = st.number_input(
        "💨 Wind Speed (km/h)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
        step=0.1
    )


with col2:

    Rain = st.number_input(
        "🌧️ Rain (mm)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1
    )

    FFMC = st.number_input(
        "🔥 FFMC",
        min_value=0.0,
        max_value=110.0,
        value=85.0,
        step=0.1
    )

    DMC = st.number_input(
        "🌲 DMC",
        min_value=0.0,
        max_value=300.0,
        value=20.0,
        step=0.1
    )


with col3:

    ISI = st.number_input(
        "🔥 ISI",
        min_value=0.0,
        max_value=100.0,
        value=8.0,
        step=0.1
    )

    Classes = st.number_input(
        "📊 Classes",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=1.0
    )

    Region = st.number_input(
        "📍 Region",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=1.0
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(
    "🔥 Predict Fire Weather Index"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        input_data = np.array(
            [[
                Temperature,
                RH,
                Ws,
                Rain,
                FFMC,
                DMC,
                ISI,
                Classes,
                Region
            ]]
        )

        # Same preprocessing as original Flask application
        new_data_scaled = standard_scaler.transform(
            input_data
        )

        # Prediction
        prediction = ridge_model.predict(
            new_data_scaled
        )

        result = float(prediction[0])


        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.html(
            f"""
            <div class="result-card">

                <div class="result-title">
                    Predicted Fire Weather Index
                </div>

                <div class="result-score">
                    {result:.2f}
                </div>

                <div class="result-label">
                    FWI Prediction
                </div>

            </div>
            """
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        m1, m2, m3 = st.columns(3)


        with m1:

            st.metric(
                "🌡️ Temperature",
                f"{Temperature:.1f} °C"
            )


        with m2:

            st.metric(
                "💧 Humidity",
                f"{RH:.1f}%"
            )


        with m3:

            st.metric(
                "💨 Wind Speed",
                f"{Ws:.1f}"
            )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-title">
                🔥 Fire Weather Assessment
            </div>
            """,
            unsafe_allow_html=True
        )


        if result < 5:

            level = "Low"
            message = (
                "Current conditions indicate relatively "
                "low fire-weather potential."
            )

        elif result < 10:

            level = "Moderate"
            message = (
                "Conditions indicate a moderate level "
                "of fire-weather potential."
            )

        elif result < 20:

            level = "High"
            message = (
                "Conditions indicate elevated fire-weather "
                "potential. Increased caution is advised."
            )

        else:

            level = "Very High"
            message = (
                "Conditions indicate very high fire-weather "
                "potential."
            )


        st.html(
            f"""
            <div class="info-card">

                <h3>
                    🔥 Risk Level: {level}
                </h3>

                <p>
                    {message}
                </p>

            </div>
            """
        )


        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        st.markdown(
            "### 📊 FWI Visualization"
        )

        progress = min(
            max(result / 30.0, 0.0),
            1.0
        )

        st.progress(
            progress
        )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">

        🔥 Algerian Forest Fire Predictor

        <br>

        Machine Learning powered by
        Ridge Regression

        <br><br>

        Developed by <b>Sourav Mukherjee</b>

    </div>
    """,
    unsafe_allow_html=True
)
