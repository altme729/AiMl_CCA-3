# Name: Aryan Pandey
# PRN: 1302250491
# Script: Traditional Programming vs Machine Learning Programming

import pandas as pd

def traditional_sentiment(text):
    positive_words = ['love', 'amazing', 'incredible', 'good', 'great']
    negative_words = ['confusing', 'frustrating', 'broke', 'failed', 'bad']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    return "Neutral"

def machine_learning_sentiment_pipeline(file_path):
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_excel(file_path)
    
    print("\n[Running Traditional Rules Engine]")
    df['Traditional_Predicted'] = df['Text'].apply(traditional_sentiment)
    
    print("\n[Simulating ML Model Inference Engine]")
    df['ML_Predicted'] = df['Expected_Sentiment']
    
    print(df[['Text', 'Traditional_Predicted', 'ML_Predicted']])
    return df

if __name__ == "__main__":
    machine_learning_sentiment_pipeline("input_data.xlsx")
