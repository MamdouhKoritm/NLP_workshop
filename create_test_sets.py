import pandas as pd
from sklearn.model_selection import train_test_split

def create_and_split_data():
    # Load the main dataset
    df = pd.read_csv('data/train.tsv', sep='\t')

    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train and test sets (20 for test)
    train_df, test_df = train_test_split(df, test_size=20, random_state=42)

    # --- Save files for Task 1 ---
    # Save the new, smaller training set (questions and labels)
    train_df_task1 = train_df[['question', 'final_QT']]
    train_df_task1.to_csv('task1/train_data.tsv', sep='\t', index=False, header=False)
    
    # Save the test set (questions and labels)
    test_df_task1 = test_df[['question', 'final_QT']]
    test_df_task1.to_csv('task1/test_data.tsv', sep='\t', index=False, header=False)
    print("Task 1 train and test sets created.")

    # --- Save files for Task 2 ---
    # Save the new, smaller training set (answers and labels)
    train_df_task2 = train_df[['answer', 'final_AS']]
    train_df_task2.to_csv('task2/train_data.tsv', sep='\t', index=False, header=False)
    
    # Save the test set (answers and labels)
    test_df_task2 = test_df[['answer', 'final_AS']]
    test_df_task2.to_csv('task2/test_data.tsv', sep='\t', index=False, header=False)
    print("Task 2 train and test sets created.")


if __name__ == '__main__':
    create_and_split_data() 