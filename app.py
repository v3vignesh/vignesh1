from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="MailGuard | Spam Detection",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# 2. Professional Styling + Purple Button
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #2563EB;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #64748B;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 15px;
    }

    /* Purple Analyze Message button */
    div.stButton > button[kind="primary"] {
        background-color: #7C3AED;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #6D28D9;
        color: white;
        border: none;
    }

    div.stButton > button[kind="primary"]:focus {
        background-color: #6D28D9;
        color: white;
        border: 2px solid #C4B5FD;
        box-shadow: 0 0 0 2px #EDE9FE;
    }

    .footer {
        color: #64748B;
        font-size: 13px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


DATASET_PATH = Path(__file__).resolve().parent / "spam.csv"


def load_dataset():
    dataset_path = DATASET_PATH

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Make sure spam.csv is in the same folder as app.py."
        )

    data = pd.read_csv(dataset_path, encoding="latin-1")

    if {"v1", "v2"}.issubset(data.columns):
        data = data[["v1", "v2"]].copy()
    else:
        data = data.iloc[:, :2].copy()

    data.columns = ["label", "message"]
    data = data.dropna(subset=["label", "message"]).copy()
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["message"] = data["message"].astype(str).str.strip()
    data = data[data["label"].isin(["ham", "spam"]) & data["message"].ne("")]
    data = data.drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError("The dataset is empty after cleaning. Check the spam.csv file.")

    return data


# --------------------------------------------------
# 3. Train and Evaluate Models
# --------------------------------------------------
@st.cache_resource
def train_models():

    data = load_dataset()

    X = data["message"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models_to_train = {
        "Linear SVM": LinearSVC(random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Multinomial Naive Bayes": MultinomialNB()
    }

    trained_models = {}
    results = {}
    confusion_matrices = {}

    for name, model in models_to_train.items():

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_test_tfidf)

        trained_models[name] = model

        results[name] = {
            "Accuracy": accuracy_score(
                y_test, predictions
            ) * 100,

            "Spam Precision": precision_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            ) * 100,

            "Spam Recall": recall_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            ) * 100,

            "Spam F1": f1_score(
                y_test,
                predictions,
                pos_label="spam",
                zero_division=0
            ) * 100
        }

        confusion_matrices[name] = confusion_matrix(
            y_test,
            predictions,
            labels=["ham", "spam"]
        )

    return (
        data,
        vectorizer,
        trained_models,
        results,
        confusion_matrices,
        len(X_train),
        len(X_test)
    )


# --------------------------------------------------
# 4. MailGuard Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">🛡️ MailGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine-learning powered email and message spam detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 5. Load Models
# --------------------------------------------------
try:
    (
        data,
        vectorizer,
        models,
        results,
        confusion_matrices,
        training_rows,
        test_rows
    ) = train_models()

except Exception as e:
    st.error(
        "Unable to load the dataset or train the models. "
        "Please check spam.csv and the required libraries."
    )
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 6. Analyze a Message
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Analyze a message</div>',
    unsafe_allow_html=True
)

st.write(
    "Paste the email subject and body, or any message you want to check."
)

selected_model_name = st.selectbox(
    "Choose a machine learning model",
    list(models.keys()),
    index=0
)

if "sample_message" not in st.session_state:
    st.session_state["sample_message"] = ""

message = st.text_area(
    "Email or message content",
    value=st.session_state["sample_message"],
    placeholder=(
        "Example: Congratulations! You have won a prize. "
        "Click here to claim your reward..."
    ),
    height=160
)

st.caption(
    "Privacy: this demo processes the text you enter for prediction. "
    "It does not send it to an email provider."
)

if st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
):

    if not message.strip():
        st.warning("Please enter a message before analyzing.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = models[selected_model_name].predict(
            message_tfidf
        )[0]

        st.subheader("Prediction Result")

        if prediction == "spam":
            st.error(
                "🚨 SPAM MESSAGE — The selected model "
                "classified this message as spam."
            )
        else:
            st.success(
                "✅ HAM — The selected model classified "
                "this as a normal message."
            )

        st.caption(f"Model used: {selected_model_name}")


# --------------------------------------------------
# 7. How It Works
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-title">How it works</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📝 Text processing")
    st.write(
        "TF-IDF converts words and phrases into numerical "
        "features that machine learning models can process."
    )

with col2:
    st.markdown("### 🤖 Classification")
    st.write(
        "Supervised machine learning models learn from labeled "
        "messages and predict spam or ham."
    )

with col3:
    st.markdown("### 📊 Evaluation")
    st.write(
        "The models are evaluated using a held-out test dataset "
        "separate from their training data."
    )

st.info(f"Active model: {selected_model_name}")


# --------------------------------------------------
# 8. Model Performance
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="section-title">Model performance</div>',
    unsafe_allow_html=True
)

st.write(
    "Metrics below come from the test split used during training, "
    "not from the message you just entered."
)

st.write(f"Selected model: **{selected_model_name}**")

selected_metrics = results[selected_model_name]

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Test accuracy",
    f"{selected_metrics['Accuracy']:.2f}%"
)

m2.metric(
    "Spam precision",
    f"{selected_metrics['Spam Precision']:.2f}%"
)

m3.metric(
    "Spam recall",
    f"{selected_metrics['Spam Recall']:.2f}%"
)

m4.metric(
    "Spam F1-score",
    f"{selected_metrics['Spam F1']:.2f}%"
)


# --------------------------------------------------
# 9. Compare Models
# --------------------------------------------------
st.divider()
st.subheader("Compare models")

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"

st.dataframe(
    results_df.style.format("{:.2f}%"),
    use_container_width=True
)

chart_data = results_df[
    ["Accuracy", "Spam F1", "Spam Precision", "Spam Recall"]
]

fig, ax = plt.subplots(figsize=(10, 5))

chart_data.plot(
    kind="bar",
    ax=ax,
    width=0.75
)

ax.set_ylabel("Score (%)")
ax.set_xlabel("Machine Learning Model")
ax.set_title("Model Performance Comparison")
ax.set_ylim(0, 105)
ax.tick_params(axis="x", rotation=15)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. Confusion Matrix
# --------------------------------------------------
st.divider()
st.subheader("Selected model confusion matrix")

st.write(
    "A false positive is a legitimate message incorrectly "
    "flagged as spam."
)

cm = confusion_matrices[selected_model_name]

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["HAM", "SPAM"],
    yticklabels=["HAM", "SPAM"],
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title(f"{selected_model_name} Confusion Matrix")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 11. Dataset Details
# --------------------------------------------------
st.divider()
st.subheader("Dataset details")

d1, d2, d3 = st.columns(3)

d1.metric("Dataset rows", f"{len(data):,}")
d2.metric("Training rows", f"{training_rows:,}")
d3.metric("Test rows", f"{test_rows:,}")

st.caption(
    f"Ham messages: {(data['label'] == 'ham').sum():,} | "
    f"Spam messages: {(data['label'] == 'spam').sum():,}"
)


# --------------------------------------------------
# 12. Sample Messages
# --------------------------------------------------
st.divider()
st.subheader("Sample messages to try")

sample_options = {
    "Spam example": (
        "Congratulations! You have won a free prize. "
        "Click here to claim your reward now!"
    ),
    "Normal example": (
        "Hi, are we meeting at the library after class today?"
    )
}

sample_choice = st.selectbox(
    "Choose a sample message",
    list(sample_options.keys())
)

st.code(sample_options[sample_choice], language=None)

if st.button("Use this sample"):
    st.session_state["sample_message"] = sample_options[sample_choice]
    st.rerun()


# --------------------------------------------------
# 13. Footer
# --------------------------------------------------
st.divider()

st.markdown(
    '<div class="footer">'
    'Educational prototype · Not a replacement for enterprise '
    'email security or phishing analysis.'
    '</div>',
    unsafe_allow_html=True
)