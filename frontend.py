import streamlit as st
from backend import get_answer, vector_store

# Set page configuration for a wide layout
st.set_page_config(page_title="Consumer Knowledge Assistant", layout="wide")

# Use custom CSS to reduce the top padding, moving content higher
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Center align the title using columns
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("grahak-GPT ⚖️🏛️ ")

st.write("Ask questions about consumer protection and get instant answers!")

# Add 4 line spaces after the descriptive text
st.write("\n" * 4)

# -----------------------------
# User query input
# -----------------------------
st.write("ENTER YOUR QUESTION HERE:")

# Add 5 line spaces before the input box
st.write("\n" * 5)

query = st.text_input("", "")

if query:
    with st.spinner("thinking and generating the response..."):

        # -----------------------------
        # Retrieve top 3 documents
        # -----------------------------
        retrieved_docs = vector_store.similarity_search(query, k=3)

        # Put them inside an expander so they don’t clutter the page
        with st.expander("📄 Retrieved Context"):
            if retrieved_docs:
                for i, doc in enumerate(retrieved_docs, start=1):
                    content = doc.page_content
                    keywords = query.lower().split()  # simple keyword highlighting
                    for kw in keywords:
                        content = content.replace(kw, f"**{kw}**")
                    st.markdown(f"**Doc {i}:**\n\n" + content[:500] + "...\n")

        # -----------------------------
        # Generate answer
        # -----------------------------
        answer = get_answer(query)
        st.subheader("💡 Answer")
        st.write(answer)
