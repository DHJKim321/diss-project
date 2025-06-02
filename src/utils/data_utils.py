import pandas as pd

def load_and_prepare_data(csv_file):
    """
    Load and prepare data from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: A DataFrame with the prepared data.
    """
    # Load the dataset
    data = pd.read_csv(csv_file)
    
    # Fill missing values with empty strings
    data.fillna('', inplace=True)
    
    # Replace subreddit with class label
    data['class'] = data['subreddit'].apply(lambda x: 0 if x.lower() == 'casualuk' else 1)
    
    # Drop the original subreddit column
    data.drop(columns=['subreddit'], inplace=True)
    
    # Concatenate title and selftext into a single text column
    data['text'] = data['title'] + ' ' + data['selftext']
    
    # Drop the original title and selftext columns
    data.drop(columns=['title', 'selftext'], inplace=True)
    
    return data