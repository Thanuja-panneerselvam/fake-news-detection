import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Preprocessing function
def preprocess_text(text):
    """
    Basic text preprocessing
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text

# Create sample dataset
def create_sample_dataset():
    """
    Create a synthetic dataset for fake news detection
    """
    fake_news = [
    "Shocking: World Leaders Secretly Control Weather",
    "Miracle Drug Cures All Diseases Overnight",
    "Massive Alien Civilization Living Underground",
    "Cryptocurrency Will Replace All Governments"
    ]
    
    real_news = [
    "Climate Change Impact on Global Agriculture",
    "New Medical Research Advances Cancer Treatment",
    "Tech Companies Invest in Renewable Energy",
    "International Cooperation Tackles Global Challenges"
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'text': fake_news + real_news,
        'label': [0]*len(fake_news) + [1]*len(real_news)
    })
    
    return df

# Main function to train and evaluate the model
def train_fake_news_detector():
    # Create dataset
    df = create_sample_dataset()
    
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vectorized)
    
    # Evaluate the model
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Function to predict new text
    def predict_fake_news(text):
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)
        probability = model.predict_proba(vectorized_text)[0]
        
        return {
            'is_fake': prediction[0] == 0,
            'confidence': max(probability)
        }
    
    return predict_fake_news

# Run the detector
if __name__ == "__main__":
    # Train the model and get prediction function
    predict_news = train_fake_news_detector()
    
    # Test with some example texts
    test_texts = [
    "mysterious underground civilization discovered", # Fake
    "renewable energy investment increases globally" # Real
    ]
    
    for text in test_texts:
        result = predict_news(text)
        print(f"\nText: {text}")
        print(f"Is Fake News: {result['is_fake']}")
        print(f"Confidence: {result['confidence']:.2%}")
