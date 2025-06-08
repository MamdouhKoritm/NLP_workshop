import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqGeneration,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from bert_score import score
import sys
sys.path.append('../task1')
sys.path.append('../task2')
from task1_classification import CustomTrainer as QuestionClassifier
from task2_classification import CustomTrainer as AnswerClassifier
import pyarabic.araby as araby

def preprocess_text(text):
    """Normalizes Arabic text."""
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    return text

def load_data():
    # Load QA pairs
    questions_df = pd.read_csv('questions_train.tsv', sep='\t', header=None, names=['question'])
    answers_df = pd.read_csv('answers_train.tsv', sep='\t', header=None, names=['answer'])
    
    # Load classifiers
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

class MentalQADataset(Dataset):
    def __init__(self, questions, answers, question_types, answer_strategies, tokenizer, max_length=512):
        self.questions = questions
        self.answers = answers
        self.question_types = question_types
        self.answer_strategies = answer_strategies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Combine question with its type for better context
        question = self.questions[idx]
        q_type = self.question_types[idx]
        answer = self.answers[idx]
        a_strategy = self.answer_strategies[idx]
        
        # Create prompt with classification info
        prompt = f"Question Type: {q_type}\nAnswer Strategy: {a_strategy}\nQuestion: {question}\nAnswer:"
        
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
    
    # Initialize tokenizer and model
    model_name = "UBC-NLP/MARBERTv2"  # Using same base model as Task 1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = MentalQADataset(
        train_df['question'].tolist(),
        train_df['answer'].tolist(),
        train_df['question_type'].tolist(),
        train_df['answer_strategy'].tolist(),
        tokenizer
    )
    
    dev_dataset = MentalQADataset(
        dev_df['question'].tolist(),
        dev_df['answer'].tolist(),
        dev_df['question_type'].tolist(),
        dev_df['answer_strategy'].tolist(),
        tokenizer
    )
    
    test_dataset = MentalQADataset(
        test_df['question'].tolist(),
        test_df['answer'].tolist(),
        test_df['question_type'].tolist(),
        test_df['answer_strategy'].tolist(),
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_task3',
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_task3',
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
    trainer.save_model('./best_model_task3')
    
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
    
    results_df.to_csv('prediction_results_task3.csv', index=False)
    print("\nPredictions saved to prediction_results_task3.csv")
    print("\nTest set metrics:")
    print(predictions.metrics)

if __name__ == '__main__':
    main()
