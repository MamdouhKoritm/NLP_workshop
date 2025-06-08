import os
# Disable symlink warning and set cache directory
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_CACHE'] = './cache'  # Store cache in local directory

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
import pyarabic.araby as araby

def preprocess_text(text):
    """Normalizes Arabic text."""
    # Ensure text is a string
    text = str(text)
    text = araby.strip_tashkeel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    return text

def load_data():
    # Load datasets
    train_df = pd.read_csv('train_data.tsv', sep='\t', header=None, names=['answer', 'labels'])
    dev_df = pd.read_csv('dev_data.tsv', sep='\t', header=None, names=['answer', 'labels'])
    test_df = pd.read_csv('test_data.tsv', sep='\t', header=None, names=['answer', 'labels'])
    
    # Preprocess labels
    for df in [train_df, dev_df, test_df]:
        df['labels'] = df['labels'].apply(lambda x: eval(x))

    # Map labels to names
    answer_strategy_map = {'1': 'Information', '2': 'Direct Guidance', '3': 'Emotional Support'}
    
    for df in [train_df, dev_df, test_df]:
        df['labels'] = df['labels'].apply(lambda x: [answer_strategy_map.get(str(label)) for label in x if str(label) in answer_strategy_map])
        df['answer'] = df['answer'].apply(preprocess_text)

    print("Dataset sizes:")
    print(f"Training: {len(train_df)}")
    print(f"Development: {len(dev_df)}")
    print(f"Test: {len(test_df)}")
    
    return train_df, dev_df, test_df

class MentalQADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Apply sigmoid and threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    
    y_true = p.label_ids

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    jaccard = jaccard_score(y_true=y_true, y_pred=y_pred, average='samples')

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'jaccard': jaccard
    }

def main():
    train_df, dev_df, test_df = load_data()

    # Prepare labels
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_df['labels'])
    dev_labels = mlb.transform(dev_df['labels'])
    test_labels = mlb.transform(test_df['labels'])
    
    # Calculate class weights to handle imbalance
    class_counts = train_labels.sum(axis=0)
    total_samples = len(train_labels)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print(f"Class counts: {class_counts}")
    print(f"Calculated class weights: {class_weights}")
    
    num_labels = len(mlb.classes_)
    print(f"Number of labels: {num_labels}")
    print(f"Classes: {mlb.classes_}")

    # Tokenizer and Model
    model_name = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

    # Prepare data
    train_answers = train_df['answer'].tolist()
    dev_answers = dev_df['answer'].tolist()
    test_answers = test_df['answer'].tolist()

    train_encodings = tokenizer(train_answers, truncation=True, padding=True, max_length=512)
    dev_encodings = tokenizer(dev_answers, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_answers, truncation=True, padding=True, max_length=512)

    train_dataset = MentalQADataset(train_encodings, train_labels)
    dev_dataset = MentalQADataset(dev_encodings, dev_labels)
    test_dataset = MentalQADataset(test_encodings, test_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results_task2',
        num_train_epochs=5,  # Increased from 3 to 5 with early stopping
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs_task2',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
    )

    # Custom trainer with weighted loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.args.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), 
                            labels.float().view(-1, self.model.config.num_labels))
            return (loss, outputs) if return_outputs else loss

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # Use dev set for validation
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the best model
    trainer.save_model('./best_model_task2')

    # Load best model and evaluate on test set
    best_model = AutoModelForSequenceClassification.from_pretrained('./best_model_task2')
    trainer = CustomTrainer(
        model=best_model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # Evaluate on test set
    test_results = trainer.predict(test_dataset)
    print("\nTest set results:")
    print(test_results.metrics)

    # Process predictions
    preds = test_results.predictions[0] if isinstance(test_results.predictions, tuple) else test_results.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    
    y_true = test_results.label_ids

    # Inverse transform labels to get original names
    predicted_labels = mlb.inverse_transform(y_pred)
    true_labels = mlb.inverse_transform(y_true)

    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'answer': test_df['answer'],
        'true_labels': [', '.join(labels) for labels in true_labels],
        'predicted_labels': [', '.join(labels) for labels in predicted_labels]
    })

    # Save results to CSV
    results_df.to_csv('prediction_results_task2.csv', index=False)
    print("\nPredictions saved to prediction_results_task2.csv")

if __name__ == '__main__':
    main()