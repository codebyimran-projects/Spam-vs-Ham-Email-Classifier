
import streamlit as st
import joblib
import os
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SpamShield | Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/spam_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
INFO_PATH = "models/model_info.pkl"


@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.error("Model files not found. Please run main.py first.")
    st.stop()

model, tfidf = load_models()

if os.path.exists(INFO_PATH):
    model_info = joblib.load(INFO_PATH)
else:
    model_info = {
        "model_name": "Unknown",
        "results": []
    }


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 15px;
    color: #94a3b8;
    margin-bottom: 30px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 15px;
}

.result-card {
    padding: 25px;
    border-radius: 15px;
    background: #111c30;
    border: 1px solid #26364f;
    margin-top: 20px;
}

.ham-result {
    color: #4ade80;
    font-size: 30px;
    font-weight: 800;
}

.spam-result {
    color: #fb7185;
    font-size: 30px;
    font-weight: 800;
}

.small-text {
    color: #94a3b8;
    font-size: 13px;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 600;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
    border: none;
}

textarea {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## SpamShield")
    st.caption("AI-Powered Email Classification")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Email Classifier",
            "Prediction History",
            "Model Information"
        ]
    )

    st.divider()

    st.markdown("### Classification")
    st.markdown("**Ham:** Legitimate email")
    st.markdown("**Spam:** Unwanted email")

    st.divider()

    st.caption("Built with Python, Scikit-learn and Streamlit")


# ============================================================
# EMAIL CLASSIFIER PAGE
# ============================================================

if page == "Email Classifier":

    st.markdown(
        '<div class="main-title">Email Spam Classifier</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Analyze an email and identify whether it is legitimate or spam.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Check Your Email</div>',
        unsafe_allow_html=True
    )

    email_text = st.text_area(
        "Paste your email message below",
        height=220,
        placeholder="Enter your email content here..."
    )

    if st.button("Analyze Email"):

        if not email_text.strip():

            st.warning("Please enter an email message.")

        else:

            cleaned_email = " ".join(email_text.split())

            email_vector = tfidf.transform([cleaned_email])

            prediction = int(model.predict(email_vector)[0])

            if prediction == 1:

                category = "Spam"
                result_class = "spam-result"
                description = "This email has been classified as unwanted or suspicious."

            else:

                category = "Ham"
                result_class = "ham-result"
                description = "This email has been classified as legitimate."

            # Confidence is only shown for models that expose
            # a meaningful probability estimate.
            confidence = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(email_vector)[0]
                confidence = float(probabilities[prediction]) * 100

            elif hasattr(model, "decision_function"):

                # A decision score is not a calibrated probability.
                score = float(model.decision_function(email_vector)[0])

                # Show the score rather than calling it confidence.
                decision_score = score

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="{result_class}">{category}</div>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if confidence is not None:

                st.metric(
                    "Prediction Probability",
                    f"{confidence:.2f}%"
                )

            elif hasattr(model, "decision_function"):

                st.metric(
                    "SVM Decision Score",
                    f"{decision_score:.4f}"
                )

            st.session_state.history.append({
                "Email": cleaned_email,
                "Prediction": category,
                "Probability": (
                    f"{confidence:.2f}%"
                    if confidence is not None
                    else "Not available"
                )
            })


# ============================================================
# PREDICTION HISTORY PAGE
# ============================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Review emails analyzed during this session.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("Clear History"):

            st.session_state.history = []
            st.rerun()

    else:

        st.info("No emails have been analyzed yet.")


# ============================================================
# MODEL INFORMATION PAGE
# ============================================================

elif page == "Model Information":

    st.markdown(
        '<div class="main-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Information about the trained machine learning model.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Selected Algorithm",
            model_info["model_name"]
        )

    with col2:
        st.metric(
            "Feature Extraction",
            "TF-IDF"
        )

    st.divider()

    st.markdown("### Model Comparison")

    if model_info["results"]:

        results_df = pd.DataFrame(model_info["results"])

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("Model evaluation information is unavailable.")
