import json
import numpy as np
import faiss
import sentence_transformers
from hashlib import sha256
from tqdm import tqdm
import os
import argparse

# Set environment variables for thread optimization
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

class FaissRAG:
    def __init__(self, docs: list[str], model_path: str):
        """
        Optimized RAG class using exact index for small datasets
        """
        self.model = sentence_transformers.SentenceTransformer(model_path)
        self.docs = docs
        
        # Batch generate embeddings (NPU optimized)
        batch_size = 128  # Reduce batch_size to decrease memory pressure [5](@ref)
        embeddings = []
        print("Generating embeddings...")
        for i in tqdm(range(0, len(docs), batch_size)):
            batch = docs[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_tensor=False)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Normalize vectors (makes inner product equivalent to cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Build exact index (for small datasets) [7](@ref)
        dim = embeddings.shape[1]

        if len(docs) > 50000:  # Use IVFFlat for large datasets
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(1024, len(docs)//100)  # Dynamically calculate number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)  # Training required
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)  # Add data directly, no training needed

    def top_k(self, query: str, k=1):
        """Query top-k most similar documents"""
        query_embed = self.model.encode([query], convert_to_tensor=False).astype('float32')
        faiss.normalize_L2(query_embed)
        scores, indices = self.index.search(query_embed, k)
        return [{
            'score': scores[0][i],
            'text': self.docs[indices[0][i]],
            'idx': int(indices[0][i])
        } for i in range(k)]


def save_and_return_decontaminated_questions(
        target_questions: dict[int | str, str],
        contaminated_questions: list[dict[str, int | str | list[dict[int | str, str]]]],
        output_file_prefix: str,
    ) -> dict[int | str, str]:
    if contaminated_questions:
        contaminated_filename = f'{output_file_prefix}_contaminated.json'
        with open(contaminated_filename, 'w') as f:
            json.dump(contaminated_questions, f, ensure_ascii=False, indent=4)
        decontaminated_question_ids = {contam['question_id'] for contam in contaminated_questions}
        decontaminated_questions = {
            question_id: question
            for question_id, question in target_questions.items()
            if question_id not in decontaminated_question_ids
        }
        decontaminated_filename = f'{output_file_prefix}_decontaminated.json'
        with open(decontaminated_filename, 'w') as f:
            json.dump(decontaminated_questions, f, ensure_ascii=False, indent=4)
        print(
            'Contamination detection completed: '
            f'Original data: {len(target_questions)} items, '
            f'Contaminated items: {len(contaminated_questions)}, '
            f'Contamination rate: {len(contaminated_questions) / len(target_questions) * 100:.2f}%, '
            f'Contaminated data saved to {contaminated_filename}, '
            f'Decontaminated data saved to {decontaminated_filename}'
        )
        return decontaminated_questions
    else:
        print('No contamination detected')
        return target_questions


def save_and_return_deduplicated_questions(
        questions: dict[int | str, str],
        duplicated_questions: list[dict[str, int | str | list[dict[int | str, str]]]],
        output_file_prefix: str,
    ) -> dict[int | str, str]:
    if duplicated_questions:
        duplicated_filename = f'{output_file_prefix}_duplicated.json'
        with open(duplicated_filename, 'w') as f:
            json.dump(duplicated_questions, f, ensure_ascii=False, indent=4)
        duplicated_question_ids = {dup['question_id'] for dup in duplicated_questions}
        deduplicated_questions = {
            question_id: question
            for question_id, question in questions.items()
            if question_id not in duplicated_question_ids
        }
        deduplicated_filename = f'{output_file_prefix}_deduplicated.json'
        with open(deduplicated_filename, 'w') as f:
            json.dump(deduplicated_questions, f, ensure_ascii=False, indent=4)
        print(
            'Deduplication completed: '
            f'Original data: {len(questions)} items, '
            f'Duplicated items: {len(duplicated_questions)}, '
            f'Duplication rate: {len(duplicated_questions) / len(questions) * 100:.2f}%, '
            f'Duplicated data saved to {duplicated_filename}, '
            f'Deduplicated data saved to {deduplicated_filename}'
        )
        return deduplicated_questions
    else:
        print('No duplication detected')
        return questions

# Decontamination function (optimized)
def decontaminate(
        model: str,
        cos_sim_threshold: float,
        target_questions: dict,
        contaminant_questions: dict,
        output_file_prefix: str,
    ) -> dict:
    """
    Detect target questions similar to contamination sources
    Optimization: Use batch queries to accelerate processing
    """
    # Build contamination source index (using exact index)
    print('Building contamination source index...')
    rag = FaissRAG(
        docs=list(contaminant_questions.values()),
        model_path=model
    )
    
    # Batch generate target question embeddings (accelerates 400k+ queries)
    print('Batch generating target question embeddings...')
    target_embeddings = []
    target_ids = []
    batch_size = 64
    target_items = list(target_questions.items())
    
    for i in tqdm(range(0, len(target_items), batch_size)):
        batch = target_items[i:i+batch_size]
        texts = [item[1] for item in batch]
        emb = rag.model.encode(texts, convert_to_tensor=False)
        target_embeddings.append(emb)
        target_ids.extend([item[0] for item in batch])
    
    target_embeddings = np.vstack(target_embeddings).astype('float32')
    faiss.normalize_L2(target_embeddings)
    
    # Batch query similarity (core acceleration) [2](@ref)
    print('Batch querying similarity...')
    scores, indices = rag.index.search(target_embeddings, 1)
    
    # Collect contaminated questions
    contaminated_questions = []
    for i, qid in tqdm(enumerate(target_ids)):
        if scores[i][0] > cos_sim_threshold:
            contaminated_questions.append({
                'question_id': qid,
                'question': target_questions[qid],
                'match_question': rag.docs[indices[i][0]],
                'score': float(scores[i][0])
            })
    
    return save_and_return_decontaminated_questions(
        target_questions, contaminated_questions, output_file_prefix
    )

# Deduplication function (full precomputation optimization)
def deduplicate(
        model: str,
        cos_sim_threshold: float,
        questions: dict,
        output_file_prefix: str,
    ) -> dict:
    """
    Detect duplicate questions (optimized for 400k+ data)
    Optimization: Dynamically compute nlist to avoid insufficient training
    """
    print('Precomputing all question embeddings...')
    docs = list(questions.values())
    model = sentence_transformers.SentenceTransformer(model)
    
    # Batch generate embeddings
    batch_size = 128  # Reduce batch_size to decrease memory pressure
    embeddings = []
    for i in tqdm(range(0, len(docs), batch_size)):
        batch = docs[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')
    
    # Build Faiss index (dynamically set nlist) [1,7](@ref)
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    
    # Dynamically select index type based on data size
    if len(docs) < 10000:  # Use exact index for small datasets
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:  # Use inverted index for large datasets
        # Dynamically calculate nlist to ensure sufficient training data (nlist*39 <= data size)
        max_nlist = len(docs) // 39
        nlist = min(1000, max(10, max_nlist))  # Ensure between 10-1000
        
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print(f"Training Faiss index (nlist={nlist})...")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = min(20, nlist)  # nprobe shouldn't exceed nlist
    
    # Batch query top-2 similar items for all questions
    print('Detecting duplicate questions...')
    scores, indices = index.search(embeddings, 2)
    duplicated_questions = []
    for i, qid in tqdm(enumerate(questions.keys())):
        if scores[i][1] > cos_sim_threshold:  # Skip self (index 0)
            duplicated_questions.append({
                'question_id': qid,
                'question': questions[qid],
                'match_question': questions[list(questions.keys())[indices[i][1]]],
                'score': float(scores[i][0])
            })
    
    return save_and_return_deduplicated_questions(
        questions, duplicated_questions, output_file_prefix
    )


def stream_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line.strip()) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data', type=str, required=True) # Data to be decontaminated
    parser.add_argument('--contaminant_source', type=str, required=True) # Contamination source data (e.g., evaluation datasets like aime2425, livecodebench)
    parser.add_argument('--model_path', type=str, required=True) # Embeding model, recommand: sentence-transformers/all-MiniLM-L6-v2
    parser.add_argument('--output_file_prefix', type=str, required=True) # Output file prefix
    parser.add_argument('--threshold', type=float, required=False, default=0.9)
    args = parser.parse_args()
    

    # Load data
    questions = {}
    for record in stream_jsonl(args.target_data):
        question = record['conversations'][0]['value']
        id = sha256(question.encode()).hexdigest()[:16]
        questions[id] = question

    print(f'Target question count: {len(questions)}')

    with open(args.contaminant_source, 'r', encoding='utf-8') as f:
        contaminant_source = json.load(f)
    print(f'Contamination source question count: {len(contaminant_source)}')

    # Set environment variables to prevent memory errors
    os.environ['PYTORCH_NPU_ALLOC_CONF'] = "expandable_segments:True"
    
    # Execute decontamination (optimized)
    decontaminated_questions = decontaminate(
        model=args.model_path,
        cos_sim_threshold=0.9,
        target_questions=questions,
        contaminant_questions=contaminant_source,
        output_file_prefix=args.output_file_prefix,
    )
    print(f'Question count after decontamination: {len(decontaminated_questions)}')
