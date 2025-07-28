import os
import yaml
import requests
import hashlib
import json
import pandas as pd
import time
from typing import Dict, List, Tuple, Any
import numbers
from requests.exceptions import HTTPError
from transformers import AutoTokenizer, AutoModel
import torch

# Load a sentence transformer model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# --------------------------------------------------
# Google AI Studio LLM Wrapper (Gemini 1.5 Flash, no-cost tier)
# --------------------------------------------------
class GoogleAIStudioLLM:
    """
    Wrapper for Google AI Studio API (Gemini 1.5 Flash, no-cost tier).
    """
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        self.api_key = api_key
        self.model = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def __call__(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        data = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    

# --------------------------------------------------
# Configuration
# --------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    GOOGLE_API_KEY = config["google_ai_studio"]["api_key"]
    GOOGLE_MODEL = config["google_ai_studio"].get("model", "gemini-1.5-flash-latest")
    NEO4J_URI = config["neo4j"]["uri"]
    NEO4J_USER = config["neo4j"]["user"]
    NEO4J_PASS = config["neo4j"]["password"]
else:
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

llm = GoogleAIStudioLLM(api_key=GOOGLE_API_KEY, model=GOOGLE_MODEL)
CACHE_DIR = os.getenv("KG_CACHE_DIR", ".kg_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# --------------------------------------------------
# Helper: Schema fingerprint
# --------------------------------------------------
def fingerprint_schema(df: pd.DataFrame) -> str:
    schema = {
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    raw = json.dumps(schema, sort_keys=True).encode('utf-8')
    return hashlib.md5(raw).hexdigest()

# --------------------------------------------------
# Parser & Description
# --------------------------------------------------

def describe_tables(dfs: Dict[str, pd.DataFrame], llm: "GoogleAIStudioLLM") -> Dict[str, str]:
    """
    Describe tables using Google AI Studio LLM (Gemini 1.5 Flash, no-cost tier).
    Handles rate limits by waiting and retrying on HTTP 429 errors.
    Caches results to avoid repeated calls for the same data.
    """
    descriptions = {}
    for name, df in dfs.items():
        fp = fingerprint_schema(df)
        cache_file = os.path.join(CACHE_DIR, f"{name}_{fp}.json")
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                descriptions[name] = json.load(f)["description"]
            continue

        summary = [f"Table: {name}", f"Shape: {df.shape}"]
        summary.append("Columns:")
        for col in df.columns:
            sample = df[col].dropna().unique()[:3].tolist()
            summary.append(f"- {col} (dtype={df[col].dtype}, sample={sample})")
        
        prompt = (
            "You are a data and radio access network telecommunication expert. Given the table schema below, generate a one-line description of the table and each column.\n"
            + "\n".join(summary)
        )

        # Retry logic for rate limiting
        max_retries = 5
        retry_wait = 60  # seconds to wait after 429
        for attempt in range(max_retries):
            try:
                response = llm(prompt)
                break
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit (429). Waiting {retry_wait} seconds before retrying...")
                        time.sleep(retry_wait)
                        continue
                    else:
                        raise
                else:
                    raise
        else:
            raise RuntimeError("Max retries exceeded for LLM API.")

        descriptions[name] = response.strip()
        with open(cache_file, 'w') as f:
            json.dump({"schema_fp": fp, "description": response}, f)
        time.sleep(2)  # Reduce frequency of requests
    return descriptions

# --------------------------------------------------
# Entity & Relationship Extraction
# --------------------------------------------------
def extract_entities_and_relations(
    descriptions: Dict[str, str],
    llm: GoogleAIStudioLLM,
    cache_dir: str = CACHE_DIR,
    batch_size: int = 10
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract entities and relationships using Google AI Studio LLM (Gemini).
    Handles large outputs by batching and merging results.
    Caches results to avoid repeated calls for the same data.
    Saves raw LLM output before attempting to extract JSON.
    """
    import re

    # Ensure descriptions is a dict and all values are strings
    if not isinstance(descriptions, dict):
        raise TypeError(f"descriptions must be a dict, got {type(descriptions).__name__}: {descriptions}")
    desc_serializable = {k: str(v) for k, v in descriptions.items()}

    desc_fp = hashlib.md5(json.dumps(desc_serializable, sort_keys=True).encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"entities_relations_{desc_fp}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
            if isinstance(cache_data, dict):
                entities = cache_data.get('entities', [])
                relationships = cache_data.get('relationships', [])
                return entities, relationships
            else:
                raise ValueError(f"Unexpected cache format in {cache_file}")

    # Batching
    table_names = list(descriptions.keys())
    all_entities = []
    all_relationships = []
    for i in range(0, len(table_names), batch_size):
        batch_tables = {k: descriptions[k] for k in table_names[i:i+batch_size]}
        batch_fp = hashlib.md5(json.dumps(batch_tables, sort_keys=True).encode('utf-8')).hexdigest()
        raw_file = os.path.join(cache_dir, f"entities_relations_{batch_fp}_raw.txt")
        prompt = (
            "You are a graph modeling expert. Given the following table descriptions, "
            "1) Identify core entity types and their key properties. "
            "2) Identify relationships between these entities. "
            f"Return ONLY a valid JSON object with two keys: 'entities' (a list of entities with 'label' and 'properties'), "
            f"and 'relationships' (a list of relationships with 'start_label', 'end_label', 'type'). "
            f"If the output is too large, only return the first {batch_size} entities. "
            "Do not include any explanation or markdown, just the JSON.\n"
            "Descriptions:\n" + json.dumps(batch_tables, indent=2)
        )

        max_retries = 5
        retry_wait = 60
        for attempt in range(max_retries):
            try:
                output = llm(prompt)
                break
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit (429). Waiting {retry_wait} seconds before retrying...")
                        time.sleep(retry_wait)
                        continue
                    else:
                        raise
                else:
                    raise
        else:
            raise RuntimeError("Max retries exceeded for LLM API.")

        # Save raw LLM output before any extraction
        with open(raw_file, "w") as f:
            f.write(output)

        # Check for empty output
        if not output or not output.strip():
            raise ValueError("LLM returned empty output! Please check your API key, quota, or prompt size.")

        print(f"LLM output for batch {i//batch_size+1}:\n", repr(output))

        # Robust JSON extraction from LLM output
        def extract_json_from_llm_output(output: str):
            output = output.strip()
            if output.startswith("```json"):
                output = output[7:]
            if output.startswith("```"):
                output = output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
            matches = list(re.finditer(r'(\{[\s\S]*\})', output))
            if matches:
                json_str = max(matches, key=lambda m: len(m.group(1))).group(1)
                try:
                    return json.loads(json_str)
                except Exception as e:
                    raise ValueError(f"Failed to parse extracted JSON (may be truncated): {json_str[:500]}...") from e
            else:
                raise ValueError(f"Failed to extract JSON from LLM output. Output was:\n{output[:500]}...")

        try:
            data = extract_json_from_llm_output(output)
        except Exception as e:
            print("Failed to parse LLM output as JSON.")
            raise e

        # Merge results
        all_entities.extend(data.get('entities', []))
        all_relationships.extend(data.get('relationships', []))

    # Save merged results in the same format as describe_tables: always a dict with keys
    with open(cache_file, "w") as f:
        json.dump({
            "desc_fp": desc_fp,
            "entities": all_entities,
            "relationships": all_relationships
        }, f)

    return all_entities, all_relationships


def load_dataframes_to_neo4j(dfs: Dict[str, pd.DataFrame],
                            entities: List[Dict],
                            relationships: List[Dict],
                            llm: GoogleAIStudioLLM,
                            batch_size: int = 100):
    """
    Loads all DataFrame tables into Neo4j as entity nodes and relationships.
    - For each DataFrame, creates nodes for each row, with properties and embedding/description.
    - For each relationship, creates relationships between nodes.
    """
    from neo4j import GraphDatabase
    import logging
    
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    
    # Map entity label to DataFrame name
    label_to_df = {ent['label']: name for name, ent in zip(dfs.keys(), entities)}
    
    # 1. Create entity nodes with properties, description, and embedding
    for ent in entities:
        label = ent['label']
        df_name = label_to_df.get(label)
        if not df_name or df_name not in dfs:
            logging.warning(f"No DataFrame found for entity label {label}")
            continue
        df = dfs[df_name]
        desc = ent.get('description', '')
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start+batch_size]
            for _, row in batch.iterrows():
                raw_props = row.to_dict()
                props = {}
                for k, v in raw_props.items():
                    safe_k = sanitize_key(k)
                    props[safe_k] = clean_neo4j_value(v)
                node_id = props.get('id') or hashlib.md5(json.dumps(props, sort_keys=True).encode('utf-8')).hexdigest()
                props['id'] = node_id
                if not desc:
                    desc = f"Entity {label} with properties: {list(props.keys())}"
                embedding = get_embedding(desc)
                props['description'] = desc
                props['embedding'] = embedding
                set_clause = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
                cypher = f"MERGE (n:{label} {{id: $id}}) SET {set_clause} RETURN n;"
                try:
                    with driver.session() as session:
                        session.run(cypher, **props)
                except Exception as e:
                    logging.error(f"Failed to insert node for {label}: {e}")
    # 2. Create relationships
    for rel in relationships:
        start_label = rel['start_label']
        end_label = rel['end_label']
        rel_type = rel['type']
        start_df = dfs.get(label_to_df.get(start_label, ''))
        end_df = dfs.get(label_to_df.get(end_label, ''))
        if start_df is None or end_df is None:
            logging.warning(f"Missing DataFrame for relationship {rel_type} between {start_label} and {end_label}")
            continue
        start_ids = set(start_df['id']) if 'id' in start_df else set()
        end_ids = set(end_df['id']) if 'id' in end_df else set()
        common_ids = start_ids & end_ids
        for node_id in common_ids:
            cypher = (
                f"MATCH (a:{start_label} {{id: $start_id}}), (b:{end_label} {{id: $end_id}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) RETURN r;"
            )
            params = {"start_id": node_id, "end_id": node_id}
            try:
                with driver.session() as session:
                    session.run(cypher, **params)
            except Exception as e:
                logging.error(f"Failed to create relationship {rel_type}: {e}")
    driver.close()

def sanitize_key(key):
    # Replace dots with underscores or just take the part after the last dot
    return key.split('.')[-1]

def clean_neo4j_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (str, bool, numbers.Number)):
        return val
    if isinstance(val, (list, tuple)):
        # Only allow lists of primitives
        if all(isinstance(x, (str, bool, numbers.Number)) or pd.isna(x) for x in val):
            return [clean_neo4j_value(x) for x in val]
        return str(val)
    # For dicts, sets, or other objects, convert to string
    return str(val)

def get_embedding(text: str):
    """
    Generate a vector embedding for the given text using a sentence transformer.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings[0].tolist()
