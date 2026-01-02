import streamlit as st
import pandas as pd
from intent_classifier import classify_intent
from evaluate_model import evaluate_model

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["NLU Tester", "Model Evaluation"])

if page == "NLU Tester":
    st.title("🤖 BotTrainer NLU Demo")

    user_input = st.text_input("Enter your message:")

    if user_input:
        result = classify_intent(user_input)

        st.write("**Predicted Intent:**", result.get("intent"))
        st.write("**Confidence:**", result.get("confidence"))

        st.write("**Entities:**")
        st.json(result.get("entities"))

elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")

    if st.button("▶ Run Evaluation"):
        results = evaluate_model("eval_data.json")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{results['precision']*100:.2f}%")
        col3.metric("Recall", f"{results['recall']*100:.2f}%")
        col4.metric("F1 Score", f"{results['f1']*100:.2f}%")

        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            results["confusion_matrix"],
            index=results["labels"],
            columns=results["labels"]
        )
        st.dataframe(cm_df)
