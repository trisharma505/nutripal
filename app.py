import streamlit as st
import pandas as pd
from backend import food_extraction, directQuery, normalize_entity

st.set_page_config(page_title="NutriPal", page_icon="🥗")
st.title("🥗 NutriPal")
st.subheader("A Nutrition Information Chatbot")

st.info(
    "NutriPal provides nutrition information using USDA data. "
    "This is informational only and is not medical advice.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your nutrition question:")
if st.button("Ask NutriPal"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching nutrition information..."):
            food = food_extraction(query, st.session_state.history)
            intent, response = directQuery(query, food, st.session_state.history)
        st.session_state.history.append({"user": query})
        st.session_state.history.append({"ai": str(response)})
        st.write("### Results")
        normalized_food = normalize_entity(food)
        display_food = "None" if intent == "groq_api" and ("lose weight" in query.lower() or "what foods should i eat" in query.lower()) else food
        st.write(f"**Detected Food:** {display_food}")
        st.write(f"**Detected Intent:** {intent}")
        st.write("### NutriPal Response")
        if isinstance(response, pd.DataFrame):
            st.dataframe(response)
        else:
            st.write(response)
st.write("---")
st.write("### Chat History")

for message in st.session_state.history:
    if "user" in message:
        st.write(f"**You:** {message['user']}")
    else:
        st.write(f"**NutriPal:** {message['ai']}")
