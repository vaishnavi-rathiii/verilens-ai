import chromadb
from sentence_transformers import SentenceTransformer

class FakeNewsRAG:
    def __init__(self):
        # 1. Initialize lightweight, in-memory ChromaDB vector store
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="fact_check_db")
        
        # 2. Load lightweight embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Seed database with initial sample knowledge base
        self._seed_database()

    def _seed_database(self):
        """Pre-loads verified factual statements and known hoaxes into vector store."""
        sample_facts = [
            {
                "id": "fact_1", 
                "text": "WHO confirms standard mRNA COVID-19 vaccines do not alter human DNA.", 
                "type": "supporting", 
                "source": "Reuters Health"
            },
            {
                "id": "fact_2", 
                "text": "NASA and FCC confirm 5G network signals do not alter weather monitoring satellite trajectories.", 
                "type": "supporting", 
                "source": "NASA FactSheet"
            },
            {
                "id": "fact_3", 
                "text": "Drinking warm lemon water cures all forms of cancer is a debunked medical myth.", 
                "type": "contradicting", 
                "source": "FactCheck.org"
            },
            {
                "id": "fact_4", 
                "text": "The Reserve Bank announced no official plans to discontinue 500 currency notes in 2025/2026.", 
                "type": "contradicting", 
                "source": "RBI Official Release"
            }
        ]
        
        documents = [item["text"] for item in sample_facts]
        metadatas = [{"type": item["type"], "source": item["source"]} for item in sample_facts]
        ids = [item["id"] for item in sample_facts]
        
        # Convert documents to embeddings and add to collection
        embeddings = self.model.encode(documents).tolist()
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def get_evidence(self, claim: str, top_k: int = 3):
        """Retrieves top matching evidence cards for a given claim."""
        query_embedding = self.model.encode([claim]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        evidence_list = []
        if results and results['documents']:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                # Convert distance metric to similarity percentage
                similarity_score = max(0, min(100, int((1.0 - dist) * 100)))
                evidence_list.append({
                    "text": doc,
                    "type": meta.get("type", "neutral"),
                    "source": meta.get("source", "Unknown"),
                    "similarity": similarity_score
                })
                
        return evidence_list