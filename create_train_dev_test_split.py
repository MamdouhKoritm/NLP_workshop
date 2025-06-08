import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits():
    # Load the main dataset
    df = pd.read_csv('data/train.tsv', sep='\t')    # We have 350 samples total, need 300/50/150 split
      # First split out the test set (150 samples)
    train_dev_df, test_df = train_test_split(df, test_size=150, random_state=42)
    
    # Split remaining samples into train (300) and dev (50)
    train_df, dev_df = train_test_split(train_dev_df, test_size=50/(len(train_dev_df)), random_state=42)

    # --- Save files for Task 1 ---
    # Save training set (questions and labels)
    train_df[['question', 'final_QT']].to_csv('task1/train_data.tsv', sep='\t', index=False, header=False)
    
    # Save development set
    dev_df[['question', 'final_QT']].to_csv('task1/dev_data.tsv', sep='\t', index=False, header=False)
    
    # Save test set
    test_df[['question', 'final_QT']].to_csv('task1/test_data.tsv', sep='\t', index=False, header=False)
    print("Task 1 train/dev/test sets created.")

    # --- Save files for Task 2 ---
    # Save training set (answers and labels)
    train_df[['answer', 'final_AS']].to_csv('task2/train_data.tsv', sep='\t', index=False, header=False)
    
    # Save development set
    dev_df[['answer', 'final_AS']].to_csv('task2/dev_data.tsv', sep='\t', index=False, header=False)
    
    # Save test set
    test_df[['answer', 'final_AS']].to_csv('task2/test_data.tsv', sep='\t', index=False, header=False)
    print("Task 2 train/dev/test sets created.")

if __name__ == '__main__':
    create_splits()
