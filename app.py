import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from intent_classifier import classify_intent, extract_entities
from evaluator_model import run_evaluation

# ===== logging + config =====
from utils.logger import setup_logger
from config import LOG_LEVEL

logger = setup_logger(__name__, LOG_LEVEL)
logger.info("BotTrainer Streamlit application started")
# ============================

st.set_page_config(page_title="BotTrainer", page_icon="🤖", layout="wide")

st.title("🤖 BotTrainer – NLU Trainer and Evaluator")
st.write(
    "Train, test and evaluate intent and entity understanding for a multi-domain assistant."
)

tab_test, tab_eval = st.tabs(["🔍 Live Test", "📊 Evaluation"])

# ---------- Live Test ----------
with tab_test:
    st.subheader("Test a user message")

    text = st.text_area(
        "Enter a message",
        "book a flight from chennai to delhi tomorrow at 6 am",
        height=100
    )

    if st.button("Analyze message"):
        logger.info("Running live intent and entity analysis")

        intent_res = classify_intent(text)
        entity_res = extract_entities(text)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Predicted intent")
            st.metric("Intent", intent_res["intent"])
            st.metric("Confidence", f"{intent_res['confidence'] * 100:.1f}%")

        with c2:
            st.markdown("#### Extracted entities")
            st.json(entity_res)

    with st.expander("Help: example messages"):
        st.write("- book a flight from chennai to mumbai tomorrow")
        st.write("- order a pizza and coke for tonight")
        st.write("- set a reminder to study at 8 pm")
        st.write("- what is the weather in bangalore tomorrow?")

# ---------- Evaluation ----------
with tab_eval:
    st.subheader("Evaluate on labelled test data")
    st.write(
        "Runs the NLU pipeline on eval_dataset.json and shows accuracy and detailed metrics."
    )

    if st.button("Run evaluation"):
        logger.info("Running evaluation on labelled dataset")

        results = run_evaluation()

        st.metric("Accuracy", f"{results['accuracy'] * 100:.2f}%")

        st.markdown("#### Per-intent precision / recall / F1")
        report_df = (
            pd.DataFrame(results["report"])
            .T.reset_index()
            .rename(columns={"index": "Intent"})
        )
        st.dataframe(report_df, use_container_width=True)

        st.markdown("#### Confusion matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            results["confusion_matrix"],
            annot=True,
            fmt="d",
            xticklabels=results["labels"],
            yticklabels=results["labels"],
            cmap="Blues",
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
