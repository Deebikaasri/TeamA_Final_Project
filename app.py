import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from intent_classifier import classify_intent, extract_entities
from evaluator_model import run_evaluation

st.set_page_config(page_title="BotTrainer", page_icon="🤖", layout="wide")

st.title("🤖 BotTrainer – NLU Demo")
st.write("Individual NLU implementation submitted under team evaluation.")

tab_test, tab_eval = st.tabs(["🔍 Live Test", "📊 Evaluation"])

# ---------- Live Test ----------
with tab_test:
    st.subheader("Test a user message")

    text = st.text_area(
        "Enter a message",
        "book a flight from chennai to delhi tomorrow at 6 am",
        height=100
    )

    if st.button("Analyze"):
        intent_result = classify_intent(text)
        entity_result = extract_entities(text)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Intent", intent_result["intent"])
            st.metric("Confidence", f"{intent_result['confidence']*100:.1f}%")

        with col2:
            st.subheader("Entities")
            st.json(entity_result)

# ---------- Evaluation ----------
with tab_eval:
    st.subheader("Evaluate model on test dataset")

    if st.button("Run Evaluation"):
        results = run_evaluation()

        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")

        report_df = (
            pd.DataFrame(results["report"])
            .T.reset_index()
            .rename(columns={"index": "Intent"})
        )
        st.dataframe(report_df, use_container_width=True)

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
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
