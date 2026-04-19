import streamlit as st
from wiki_loader import load_all_wiki
from predict import predict_with_evidence_list
from retrieval import Retriever

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Fact Verification System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fact Verification System")
st.markdown("Verify claims using **Hybrid Retrieval + DistilBERT**")
st.divider()

# --------------------------
# LOAD (CACHED)
# --------------------------
@st.cache_resource
def load_resources():
    wiki = load_all_wiki("data/wiki/")
    retriever = Retriever(wiki)
    return retriever

retriever = load_resources()

# --------------------------
# SIDEBAR SETTINGS
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Evidence sentences", 1, 10, 5)
    st.caption("Higher = more evidence, slower speed")

# --------------------------
# INPUT
# --------------------------
claim = st.text_input(
    "Enter a claim:",
    placeholder="e.g. Virat Kohli is a cricketer"
)

verify = st.button("🚀 Verify Claim")

# --------------------------
# RUN
# --------------------------
if verify:

    if not claim.strip():
        st.warning("⚠️ Please enter a valid claim")
        st.stop()

    with st.spinner("🔍 Retrieving evidence & verifying..."):

        evidence = retriever.retrieve(claim, k=k)
        evidence_texts = [e["text"] for e in evidence]

        # --------------------------
        # SAFETY GATE
        # --------------------------
        if not evidence_texts:
            result = {
                "label": "NOT ENOUGH INFO",
                "scores": {
                    "SUPPORTS": 0.0,
                    "REFUTES": 0.0,
                    "NOT ENOUGH INFO": 100.0
                },
                "confidence": 100.0,
                "used_evidence": "No evidence found"
            }
        else:
            result = predict_with_evidence_list(claim, evidence_texts)

    st.divider()

    label = result["label"]
    scores = result["scores"]
    confidence = result.get("confidence", 0)

    # --------------------------
    # RESULT DISPLAY
    # --------------------------
    if label == "SUPPORTS":
        st.success(f"✅ SUPPORTS ({confidence:.2f}%)")
    elif label == "REFUTES":
        st.error(f"❌ REFUTES ({confidence:.2f}%)")
    else:
        st.warning(f"⚠️ NOT ENOUGH INFO ({confidence:.2f}%)")

    # --------------------------
    # SCORES
    # --------------------------
    st.subheader("📊 Prediction Scores")

    c1, c2, c3 = st.columns(3)
    c1.metric("SUPPORTS", f"{scores.get('SUPPORTS', 0):.2f}%")
    c2.metric("REFUTES", f"{scores.get('REFUTES', 0):.2f}%")
    c3.metric("NEI", f"{scores.get('NOT ENOUGH INFO', 0):.2f}%")

    # --------------------------
    # BEST EVIDENCE
    # --------------------------
    st.subheader("⭐ Key Evidence")

    st.info(result.get("used_evidence", "No key evidence available"))

    # --------------------------
    # ALL EVIDENCE
    # --------------------------
    st.subheader("📄 All Evidence")

    if not evidence:
        st.info("No strong evidence found.")
    else:
        for i, ev in enumerate(evidence):
            with st.expander(f"{i+1}. {ev['page']} (score {ev['score']:.3f})"):
                st.write(ev["text"])

    # --------------------------
    # SUMMARY TABLE
    # --------------------------
    st.divider()

    st.subheader("🧾 Summary")

    st.markdown(f"""
    | Field | Value |
    |---|---|
    | Claim | {claim} |
    | Verdict | {label} |
    | Confidence | {confidence:.2f}% |
    | SUPPORTS | {scores.get('SUPPORTS', 0):.2f}% |
    | REFUTES | {scores.get('REFUTES', 0):.2f}% |
    | NEI | {scores.get('NOT ENOUGH INFO', 0):.2f}% |
    """)