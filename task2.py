import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

######################################################### sentiment analysis #############################################

# Download VADER Lexicon
nltk.download("vader_lexicon")

# Load cleaned data
df = pd.read_csv("data/cleaned_reddit_posts.csv")
print(df.columns)
# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_sentiment(text):
    if isinstance(text, str):  
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    return "Neutral"  # Default for empty content

# Apply sentiment analysis
df["Content"].fillna("", inplace=True)
df["Sentiment"] = df["Content"].apply(get_sentiment)

######################################################### crisis terms #############################################
# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=100, stop_words="english"
)

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df["Content"])

# Get important words
important_words = tfidf_vectorizer.get_feature_names_out()
print(f"Top TF-IDF terms: {important_words[:20]}")  # Display first 20 terms


# Define risk categories
high_risk_keywords = ["suicide", "kill myself", "i dont want to live", "hopeless", "overdose"]
moderate_risk_keywords = ["struggling", "lost", "help", "relapse", "alone", "depressed"]
low_risk_keywords = ["mental health", "self care", "stress"]

# Function to classify risk level
def classify_risk(text):
    if isinstance(text, str):
        text_lower = text.lower()
        
        if any(term in text_lower for term in high_risk_keywords):
            return "High-Risk"
        elif any(term in text_lower for term in moderate_risk_keywords):
            return "Moderate Concern"
        else:
            return "Low Concern"
    
    return "Low Concern"

# Apply risk classification
df["Risk Level"] = df["Content"].apply(classify_risk)

# Save final dataset
df.to_csv("data/classified_reddit_posts.csv", index=False)
print("✅ Posts classified into risk levels. Saved as classified_reddit_posts.csv")


# Create a list of dictionaries for JSON storage
data_for_json_with_crisis = []

for index, row in df.iterrows():
    data_for_json_with_crisis.append({
        "Post ID": index,  # or use a specific post ID if available
        "Content": row["Content"],
        "Sentiment": row["Sentiment"],
        "Crisis Risk": row["Risk Level"]
    })

# Save the sentiment and crisis risk results as JSON
with open("data/sentiment_and_crisis_results.json", "w") as json_file:
    json.dump(data_for_json_with_crisis, json_file, indent=4)

print("✅ Sentiment and Crisis Risk results have been saved as JSON!")


######################################################### visualization #############################################
# Plot sentiment distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Sentiment"], palette="coolwarm")
plt.title("Sentiment Distribution")
plt.savefig("figures/sentiment_distribution.png")  # Save as PNG file
plt.show()
plt.close()  # Close the plot to avoid display during processing

# Plot risk level distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Risk Level"], palette="viridis", order=["Low Concern", "Moderate Concern", "High-Risk"])
plt.title("Crisis Risk Level Distribution")
plt.savefig("figures/Crisis Risk Level Distribution.png")  # Save as PNG file
plt.show()
plt.close()  # Close the plot to avoid display during processing
