import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    VECTOR_DB_PATH = config["vector_db"]["path"]
    GOOGLE_API_KEY = config["google_ai_studio"]["api_key"]
    GOOGLE_MODEL = config["google_ai_studio"].get("model", "gemini-1.5-flash-latest")
else:
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

# --------------------------------------------------
# Step 1: NER System
# --------------------------------------------------
class NERSystem:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

    def extract_entities(self, query: str) -> List[Dict[str, str]]:
        return self.ner_pipeline(query)

# --------------------------------------------------
# Step 2: Vector Embedding and Query Understanding
# --------------------------------------------------
class QueryUnderstanding:
    def __init__(self, vector_model_name: str, vector_db_path: str):
        self.vector_model = SentenceTransformer(vector_model_name)
        self.vector_db = self.load_vector_db(vector_db_path)

    def load_vector_db(self, path: str) -> Dict[str, np.ndarray]:
        # Simulated vector database loading
        if os.path.exists(path):
            return torch.load(path)
        else:
            raise FileNotFoundError(f"Vector database not found at {path}")

    def find_correlated_tables(self, entities: List[Dict[str, str]]) -> List[str]:
        entity_embeddings = [self.vector_model.encode(entity['word']) for entity in entities]
        results = []
        for table_name, table_embedding in self.vector_db.items():
            similarity = cosine_similarity(entity_embeddings, [table_embedding])
            if np.max(similarity) > 0.8:  # Threshold for correlation
                results.append(table_name)
        return results

# --------------------------------------------------
# Step 3: Answer Generation
# --------------------------------------------------
class AnswerGeneration:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def generate_answer(self, query: str, context: List[str]) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        data = {
            "contents": [{"parts": [{"text": query}]}],
            "context": context
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()

# --------------------------------------------------
# Main Chatbot Workflow
# --------------------------------------------------
def chatbot_workflow(user_query: str):
    # Initialize components
    ner_system = NERSystem(model_name="fine-tuned-ner-model")
    query_understanding = QueryUnderstanding(vector_model_name="sentence-transformers/all-MiniLM-L6-v2", vector_db_path=VECTOR_DB_PATH)
    answer_generation = AnswerGeneration(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)

    # Step 1: Extract entities
    entities = ner_system.extract_entities(user_query)
    print(f"Extracted Entities: {entities}")

    # Step 2: Find correlated tables
    correlated_tables = query_understanding.find_correlated_tables(entities)
    print(f"Correlated Tables: {correlated_tables}")

    # Step 3: Generate answer
    answer = answer_generation.generate_answer(user_query, context=correlated_tables)
    print(f"Generated Answer: {answer}")

    return answer

# Example usage
if __name__ == "__main__":
    user_query = "What is the configuration of EnodeB in SubNetwork 1?"
    chatbot_workflow(user_query)