import streamlit as st
import json
import time

st.set_page_config(
    page_title="SSN AI Dashboard",
    layout="wide"
)

st.title("SSN College AI Chatbot Dashboard")

STATS_FILE = "stats.json"

def load_stats():

    with open(STATS_FILE, "r") as f:

        return json.load(f)

placeholder = st.empty()

while True:

    stats = load_stats()

    with placeholder.container():

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Queries",
            stats["total_queries"]
        )

        col2.metric(
            "Documents Uploaded",
            stats["documents_uploaded"]
        )

        col3.metric(
            "Average Latency",
            f"{stats['average_latency']} sec"
        )

        st.divider()

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Tokens/sec",
            stats["tokens_per_second"]
        )

        col5.metric(
            "RAM Usage",
            f"{stats['ram_usage']} GB"
        )

        col6.metric(
            "Total Chunks",
            stats["total_chunks"]
        )

        st.divider()

        st.subheader("Last User Question")

        st.write(stats["last_question"])

        st.subheader("Last Detected Language")

        st.write(stats["last_language"])

    time.sleep(2)

    st.rerun()
