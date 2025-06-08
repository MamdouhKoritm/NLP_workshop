import os
# Disable symlink warning and set cache directory
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './cache'  # Store cache in local directory
os.environ['SENTENCE_TRANSFORMERS_HOME'] = './cache'  # Cache for sentence transformers

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSeq2SeqGeneration,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import sys
sys.path.append('../task1')
sys.path.append('../task2')
from task1_classification import CustomTrainer as QuestionClassifier
from task2_classification import CustomTrainer as AnswerClassifier
import pyarabic.araby as araby

class QARetriever:
    def __init__(self, questions, answers, question_types, answer_strategies):
        self.questions = questions
        self.answers = answers
        self.question_types = question_types
        self.answer_strategies = answer_strategies
        
        # Initialize FAISS index for dense retrieval
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.question_embeddings = self.model.encode(questions, convert_to_tensor=True)
        
        # Create FAISS index
        self.dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.question_embeddings.cpu().numpy())
        
    def retrieve(self, query, query_type, top_k=5):
        # Get query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Perform dense retrieval
        D, I = self.index.search(query_embedding.cpu().numpy().reshape(1, -1), top_k * 2)
        
        # Get candidate examples
        candidates = []
        for idx in I[0]:
            candidates.append({
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'question_type': self.question_types[idx],
                'answer_strategy': self.answer_strategies[idx],
                'score': float(D[0][len(candidates)])
            })
        
        # Rerank based on question type and semantic similarity
        reranked = self.rerank_candidates(candidates, query_type, query)
        
        return reranked[:top_k]
    
    def rerank_candidates(self, candidates, query_type, query):
        for candidate in candidates:
            # Base score from dense retrieval
            score = candidate['score']
            
            # Boost score if question types match
            if any(qtype in candidate['question_type'] for qtype in query_type):
                score *= 1.5
                
            # Calculate BM25-style keyword match score
            keyword_score = self.calculate_keyword_similarity(query, candidate['question'])
            
            # Final score combines dense retrieval, type matching, and keyword matching
            candidate['final_score'] = score * 0.6 + keyword_score * 0.4
            
        # Sort by final score
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        return reranked
    
    def calculate_keyword_similarity(self, query, candidate):
        # Simple keyword matching score
        query_words = set(query.lower().split())
        candidate_words = set(candidate.lower().split())
        overlap = len(query_words.intersection(candidate_words))
        return overlap / len(query_words)

def preprocess_text(text):
    """Normalizes Arabic text."""
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    return text

def construct_prompt(query, retrieved_examples, query_type, target_strategy):
    prompt = "System: You are an Arabic mental health counselor. Answer the question professionally and empathetically.\n\n"
    
    # Add context about question type and desired answer strategy
    prompt += f"Question Type: {', '.join(query_type)}\n"
    prompt += f"Answer Strategy: {target_strategy}\n\n"
    
    # Add relevant examples
    prompt += "Similar Cases:\n"
    for i, example in enumerate(retrieved_examples[:3], 1):
        prompt += f"Example {i}:\n"
        prompt += f"Q: {example['question']}\n"
        prompt += f"A: {example['answer']}\n\n"
    
    # Add the current question
    prompt += f"Current Question: {query}\n"
    prompt += "Answer:"
    
    return prompt

def load_data():
    # Load QA pairs
    questions_df = pd.read_csv('questions_train.tsv', sep='\t', header=None, names=['question'])
    answers_df = pd.read_csv('answers_train.tsv', sep='\t', header=None, names=['answer'])
    
    # Load and initialize classifiers
    question_classifier = QuestionClassifier.from_pretrained('../task1/best_model')
    answer_classifier = AnswerClassifier.from_pretrained('../task2/best_model_task2')
    
    # Preprocess text
    questions_df['question'] = questions_df['question'].apply(preprocess_text)
    answers_df['answer'] = answers_df['answer'].apply(preprocess_text)
    
    # Get classifications
    q_preds = question_classifier.predict(questions_df['question'].tolist())
    a_preds = answer_classifier.predict(answers_df['answer'].tolist())
    
    # Add classifications to dataframe
    questions_df['question_type'] = q_preds
    answers_df['answer_strategy'] = a_preds
    
    # Combine into one dataframe
    df = pd.concat([questions_df, answers_df], axis=1)
    
    # Split into train/dev/test
    train_df = df.iloc[:300]
    dev_df = df.iloc[300:350]
    test_df = df.iloc[350:]
    
    return train_df, dev_df, test_df

class AdvancedMentalQADataset(Dataset):
    def __init__(self, questions, answers, question_types, answer_strategies, tokenizer, retriever, max_length=512):
        self.questions = questions
        self.answers = answers
        self.question_types = question_types
        self.answer_strategies = answer_strategies
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        q_type = self.question_types[idx]
        a_strategy = self.answer_strategies[idx]
        
        # Retrieve similar examples
        retrieved = self.retriever.retrieve(question, q_type)
        
        # Construct prompt with retrieved examples
        prompt = construct_prompt(question, retrieved, q_type, a_strategy)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Calculate BERTScore
    P, R, F1 = score(decoded_preds, decoded_labels, lang='ar', verbose=True)
    
    return {
        'bertscore_precision': P.mean().item(),
        'bertscore_recall': R.mean().item(),
        'bertscore_f1': F1.mean().item()
    }

def main():
    # Load and prepare data
    train_df, dev_df, test_df = load_data()
    
    # Initialize retriever with training data
    retriever = QARetriever(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        train_df['question_type'].tolist(),
        train_df['answer_strategy'].tolist()
    )
    
    # Initialize tokenizer and model
    model_name = "arabic-nlp/arabartsec2seq"  # Arabic-specific seq2seq model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(
        model_name,
        max_length=512,
        num_beams=5,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    
    # Create datasets
    train_dataset = AdvancedMentalQADataset(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        train_df['question_type'].tolist(),
        train_df['answer_strategy'].tolist(),
        tokenizer,
        retriever
    )
    
    dev_dataset = AdvancedMentalQADataset(
        dev_df['question'].tolist(),
        dev_df['answer'].tolist(),
        dev_df['question_type'].tolist(),
        dev_df['answer_strategy'].tolist(),
        tokenizer,
        retriever
    )
    
    test_dataset = AdvancedMentalQADataset(
        test_df['question'].tolist(),
        test_df['answer'].tolist(),
        test_df['question_type'].tolist(),
        test_df['answer_strategy'].tolist(),
        tokenizer,
        retriever
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_task3_rag',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_task3_rag',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bertscore_f1",
        greater_is_better=True,
        generation_max_length=512,
        predict_with_generate=True
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save best model
    trainer.save_model('./best_model_task3_rag')
    
    # Generate predictions on test set
    predictions = trainer.predict(test_dataset)
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(
        predictions.predictions, skip_special_tokens=True
    )
    
    # Save results
    results_df = pd.DataFrame({
        'question': test_df['question'],
        'true_answer': test_df['answer'],
        'predicted_answer': decoded_preds,
        'question_type': test_df['question_type'],
        'answer_strategy': test_df['answer_strategy']
    })
    
    results_df.to_csv('prediction_results_task3_rag.csv', index=False)
    print("\nPredictions saved to prediction_results_task3_rag.csv")
    print("\nTest set metrics:")
    print(predictions.metrics)

if __name__ == '__main__':
    main()
