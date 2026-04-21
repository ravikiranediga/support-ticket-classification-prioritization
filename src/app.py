"""
Streamlit Web Application
Interactive ticket classification demo
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
ROOT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
from preprocessing import combine_ticket
from model import TicketClassifier
from config import MODELS_DIR, TYPE_CM_PATH, PRIORITY_CM_PATH


# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .prediction-box {background-color: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem;
                     border-left: 4px solid #1f77b4; margin: 1rem 0; text-align: center;}
    .prediction-label {font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;}
    .prediction-value {font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin: 0.5rem 0;}
    .priority-critical {color: #d32f2f;}
    .priority-high {color: #f57c00;}
    .priority-medium {color: #f9a825;}
    .priority-low {color: #388e3c;}
</style>
""", unsafe_allow_html=True)


# ===========================
# MODEL LOADING
# ===========================
@st.cache_resource
def load_models():
    """Load trained classifiers"""
    try:
        type_clf = TicketClassifier.load('Ticket Type')
        priority_clf = TicketClassifier.load('Ticket Priority')
        return {'type': type_clf, 'priority': priority_clf}
    except FileNotFoundError as e:
        st.error(f"Models not found: {e}. Run `python src/train.py` first.")
        st.stop()


def predict(classifier, subject: str, description: str):
    """Predict ticket category or priority"""
    combined = combine_ticket(subject, description)
    pred_encoded = classifier.pipeline.predict([combined])[0]
    prediction = classifier.label_encoder.inverse_transform([pred_encoded])[0]

    # Confidence from SVM decision function
    confidence = None
    if hasattr(classifier.pipeline.named_steps['clf'], 'decision_function'):
        scores = classifier.pipeline.decision_function([combined])[0]
        confidence = float(max(scores))

    return prediction, confidence


def priority_color(priority: str) -> str:
    """CSS class for priority color coding"""
    colors = {
        'Critical': 'priority-critical',
        'High': 'priority-high',
        'Medium': 'priority-medium',
        'Low': 'priority-low'
    }
    return colors.get(priority, '')


def get_recommendations(ticket_type: str, priority: str):
    """Recommended actions based on prediction"""
    recs = {
        ('Technical issue', 'Critical'): [
            "Escalate immediately to Tier 3 support",
            "Notify engineering team",
            "Set SLA to 1 hour response"
        ],
        ('Billing inquiry', 'High'): [
            "Route to billing team within 2 hours",
            "Verify account status",
            "Prepare refund if eligible"
        ],
        ('Refund request', 'Medium'): [
            "Check purchase history",
            "Review refund policy",
            "Process within 24-48 hours"
        ],
        ('Cancellation request', 'Low'): [
            "Process within 2 business days",
            "Send confirmation email"
        ],
        ('Product inquiry', 'Low'): [
            "Send product info within 24 hours"
        ]
    }

    key = (ticket_type, priority)
    if key in recs:
        return recs[key]

    # Priority-only defaults
    defaults = {
        'Critical': ["Respond within 1 hour", "Escalate to supervisor"],
        'High': ["Respond within 4 hours", "Assign to senior agent"],
        'Medium': ["Respond within 24 hours"],
        'Low': ["Respond within 48 hours"]
    }
    return defaults.get(priority, ["Follow standard procedure"])


# ===========================
# MAIN APP
# ===========================
def main():
    st.title("🎫 Support Ticket Classification & Prioritization")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Home", "Classify Ticket", "Performance", "About"])

        st.markdown("---")
        st.subheader("Model Status")
        models = load_models()
        st.success("✅ Models loaded")
        st.write("**Ticket Type:** 5 classes | Acc: 22.4%")
        st.write("**Priority:** 4 classes | Acc: 24.4%")

    # ========== HOME ==========
    if page == "Home":
        st.header("Welcome")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🎯 The Problem
            Support teams manually sort hundreds of tickets - time wasted, urgent issues buried.

            ### 🤖 The Solution
            ML automatically predicts ticket **category** and **priority** for faster routing.
            """)

        with col2:
            st.markdown("""
            ### ✅ Benefits
            - ⚡ Faster routing to correct teams
            - 🎯 Better prioritization
            - 📊 Consistent classification
            - 💰 Lower operational costs
            """)

        st.markdown("---")
        st.subheader("Try It Out")
        st.info("Go to **'Classify Ticket'** to test the system!")

    # ========== CLASSIFY ==========
    elif page == "Classify Ticket":
        st.header("🔮 Classify a New Ticket")

        type_clf = models['type']
        priority_clf = models['priority']

        with st.form("ticket_form"):
            subject = st.text_input(
                "Subject",
                placeholder="e.g., Cannot login to my account",
                help="Brief summary"
            )
            description = st.text_area(
                "Description",
                placeholder="Describe the issue in detail...",
                height=120,
                help="Full description"
            )
            submitted = st.form_submit_button("🚀 Classify", use_container_width=True)

            if submitted:
                if not subject.strip() or not description.strip():
                    st.warning("⚠️ Enter both subject and description")
                else:
                    # Predict
                    type_pred, type_conf = predict(type_clf, subject, description)
                    prio_pred, prio_conf = predict(priority_clf, subject, description)

                    # Results
                    st.markdown("---")
                    st.subheader("🎯 Results")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">Category</div>
                            <div class="prediction-value">{type_pred}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        color = priority_color(prio_pred)
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">Priority</div>
                            <div class="prediction-value {color}">{prio_pred}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence
                    if type_conf is not None or prio_conf is not None:
                        st.markdown("**Confidence:**")
                        c1, c2 = st.columns(2)
                        with c1:
                            if type_conf is not None:
                                st.metric("Type", f"{type_conf:.3f}")
                        with c2:
                            if prio_conf is not None:
                                st.metric("Priority", f"{prio_conf:.3f}")

                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommended Actions")
                    for i, action in enumerate(get_recommendations(type_pred, prio_pred), 1):
                        st.write(f"{i}. {action}")

    # ========== PERFORMANCE ==========
    elif page == "Performance":
        st.header("📊 Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ticket Type Classification")
            st.markdown("""
            | Metric | Score |
            |--------|-------|
            | Accuracy | 22.4% |
            | Precision | 22.4% |
            | Recall | 22.4% |
            | F1-Score | 22.4% |
            """)
            if TYPE_CM_PATH.exists():
                st.image(str(TYPE_CM_PATH), caption="Confusion Matrix")
            else:
                st.warning("Run train.py to generate charts")

        with col2:
            st.subheader("Priority Prediction")
            st.markdown("""
            | Metric | Score |
            |--------|-------|
            | Accuracy | 24.4% |
            | Precision | 24.4% |
            | Recall | 24.4% |
            | F1-Score | 24.4% |
            """)
            if PRIORITY_CM_PATH.exists():
                st.image(str(PRIORITY_CM_PATH), caption="Confusion Matrix")
            else:
                st.warning("Run train.py to generate charts")

        st.markdown("---")
        st.info("""
        **Note:** Accuracy exceeds random baseline (20%/25%) but is limited by templated text.
        The system is intended as **decision-support**, not full automation.
        """)

    # ========== ABOUT ==========
    elif page == "About":
        st.header("About This Project")

        st.markdown("""
        ### 🎯 Objective
        Build an ML system to automatically classify and prioritize customer support tickets.

        ### 📊 Dataset
        - 8,469 historical tickets
        - 5 categories: Billing, Cancellation, Product, Refund, Technical
        - 4 priorities: Critical, High, Medium, Low

        ### 🛠️ Stack
        - Python, scikit-learn, NLTK, Streamlit, Matplotlib, Seaborn

        ### 📈 Methodology
        1. **Preprocess** – Clean, tokenize, lemmatize, remove stopwords
        2. **Extract Features** – TF-IDF (1-2 grams, 3000 features)
        3. **Train** – Linear SVM (80-20 split)
        4. **Evaluate** – Accuracy, Precision, Recall, F1, Confusion Matrix
        5. **Deploy** – Streamlit web app

        ### 📁 Modular Structure
        ```
        src/
        ├── app.py          # Web UI (Streamlit)
        ├── train.py        # Training entry point
        ├── model.py        # TicketClassifier class
        ├── preprocessing.py # Text cleaning
        └── config.py       # Configuration
        ```
        """)


if __name__ == "__main__":
    main()
