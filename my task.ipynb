import json
import math
import os
from typing import List, Dict, Any
from collections import Counter
from transformers import AutoTokenizer

class TokenBasedReranker:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        By loading a subword tokenizer (WordPiece), we inherently capture morphological
        variations (e.g., 'running' -> 'run', '##ning'). This allows the algorithm to match
        word roots even if the exact word forms differ between the query and the document.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _extract_text(self, document: Any) -> str:
        """
        Since the constraint dictates not altering the original data structure, flattening
        the JSON dynamically into a single string ensures no text field is missed while
        strictly preserving the original object for the final output.
        """
        texts = []
        if isinstance(document, str):
            texts.append(document)
        elif isinstance(document, dict):
            for value in document.values():
                texts.append(self._extract_text(value))
        elif isinstance(document, list):
            for item in document:
                texts.append(self._extract_text(item))
        return " ".join(texts)

    def _tokenize(self, text: str) -> List[int]:
        """
        Special tokens ([CLS], [SEP]) are explicitly excluded because their guaranteed presence
        in every tokenized string would artificially inflate the baseline cosine similarity
        across all documents, ruining the strict 0.0 lower bound for irrelevant texts.
        """
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _calculate_idf(self, documents: List[Dict]) -> Dict[int, float]:
        """
        Subword tokenization tends to produce highly frequent, uninformative fragments
        (like '##s' or '##ing'). A local Inverse Document Frequency (IDF) weights down
        these common subwords while heavily rewarding rare, highly specific query tokens.
        """
        num_docs = len(documents)
        if num_docs == 0:
            return {}

        df = Counter()
        for doc in documents:
            text = self._extract_text(doc)
            unique_tokens = set(self._tokenize(text))
            for token_id in unique_tokens:
                df[token_id] += 1

        idf = {}
        for token_id, freq in df.items():
            idf[token_id] = math.log(num_docs / (freq + 1)) + 1.0

        return idf

    def _compute_tf(self, tokens: List[int]) -> Dict[int, float]:
        """
        Normalizing Term Frequency (TF) prevents extremely long documents from
        dominating the score simply by having more raw token counts.
        """
        tf = Counter(tokens)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return {}
        return {token_id: count / total_tokens for token_id, count in tf.items()}

    def score(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Cosine Similarity of TF-IDF vectors is chosen over standard BM25 because
        it is mathematically bounded to [0.0, 1.0]. This perfectly fulfills the
        strict range constraint for the output scores.
        """
        if not documents:
            return []

        idf = self._calculate_idf(documents)

        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)

        query_mag = 0.0
        query_vec = {}

        # Build query TF-IDF vector
        for token_id, tf_val in query_tf.items():
            token_idf = idf.get(token_id, math.log(len(documents) + 1) + 1.0)
            weight = tf_val * token_idf
            query_vec[token_id] = weight
            query_mag += weight ** 2

        query_mag = math.sqrt(query_mag)

        results = []
        for doc in documents:
            text = self._extract_text(doc)
            doc_tokens = self._tokenize(text)
            doc_tf = self._compute_tf(doc_tokens)

            doc_mag = 0.0
            doc_vec = {}
            # Build document TF-IDF vector
            for token_id, tf_val in doc_tf.items():
                token_idf = idf.get(token_id, 1.0)
                weight = tf_val * token_idf
                doc_vec[token_id] = weight
                doc_mag += weight ** 2

            doc_mag = math.sqrt(doc_mag)

            dot_product = 0.0
            for token_id, q_weight in query_vec.items():
                if token_id in doc_vec:
                    dot_product += q_weight * doc_vec[token_id]

            score = 0.0
            if query_mag > 0 and doc_mag > 0:
                score = dot_product / (query_mag * doc_mag)

            score = max(0.0, min(1.0, score))

            results.append({
                "document": doc,
                "score": float(f"{score:.3f}")
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


def main():
    docs_path = 'documents.json'
    queries_path = 'queries.txt'

    if not os.path.exists(docs_path) or not os.path.exists(queries_path):
        print("Error: Ensure 'documents.json' and 'queries.txt' are in the same directory.")
        return

    with open(docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]

    reranker = TokenBasedReranker()
    all_results = {}

    for query in queries:
        print(f"--- Query: '{query}' ---")
        scored_docs = reranker.score(query, documents)
        all_results[query] = scored_docs

        # Displaying only the top 3 highest-scoring documents for readability
        for res in scored_docs[:3]:
            doc_id = res['document'].get('id', 'N/A')
            doc_name = res['document'].get('name', 'N/A')
            print(f"Score: {res['score']:.3f} | ID: {doc_id} | Name: {doc_name}")
        print("\n")

    # Optional: Save the complete output (including the original unmodified documents) to a new file
    output_path = 'scored_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Complete results successfully saved to {output_path}")

if __name__ == "__main__":
    main()
