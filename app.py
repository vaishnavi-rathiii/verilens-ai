import streamlit as st
from rag_engine import FakeNewsRAG

# Page Configuration
st.set_page_config(page_title="AIML-02 | Fake News Detector", layout="wide")

st.title("🛡️ AIML-02 | AI Fake News Detector")
st.caption("RAG-Powered Claim Verification & Evidence Engine")

# Load Member 2's RAG Pipeline
@st.cache_resource
def load_rag_pipeline():
    return FakeNewsRAG()

rag = load_rag_pipeline()

# User Input Section
user_claim = st.text_area(
    "Enter Claim or Headline to Verify:", 
    placeholder="e.g., Drinking lemon water can cure cancer..."
)

if st.button("Analyze Claim", type="primary"):
    if not user_claim.strip():
        st.warning("Please enter a valid claim to analyze.")
    else:
        st.divider()
        
        # 1. Fetch RAG Evidence (Member 2 Function Call)
        with st.spinner("Searching Vector Database for evidence..."):
            evidence_results = rag.get_evidence(user_claim)

        # 2. Layout Results into 2 Columns
        col_verdict, col_evidence = st.columns([1, 1.2])

        # Left Column: AI Model Output (Member 1 Output Area)
        with col_verdict:
            st.subheader("🤖 AI Verdict & Confidence")
            
            # Simple UI display simulation
            st.error("🔴 LIKELY FAKE / MISLEADING")
            st.metric(label="AI Confidence Score", value="89%")
            
            st.markdown("**Explanation:**")
            st.info("The claim contradicts multiple verified medical facts stored in our database regarding cancer treatments.")

        # Right Column: Evidence Retrieval (Member 2 Output Area)
        with col_evidence:
            st.subheader("🔎 Retrieved RAG Evidence")
            st.caption("Facts retrieved from Vector DB using Semantic Search:")
            
            for item in evidence_results:
                is_supporting = item["type"] == "supporting"
                badge_title = "🟢 Supporting Evidence" if is_supporting else "🔴 Contradicting Evidence"
                
                with st.expander(f"{badge_title} (Similarity: {item['similarity']}%)", expanded=True):
                    st.write(f"**Source:** {item['source']}")
                    st.write(f"**Retrieved Content:** {item['text']}")