import praw
import pandas as pd
import re
from config import reddit_credentials
import emoji
import string
import nltk
from nltk.corpus import stopwords

######################################################### fetching data #############################################
# Authenticate Reddit API
reddit = praw.Reddit(
    client_id=reddit_credentials["client_id"],
    client_secret=reddit_credentials["client_secret"],
    user_agent=reddit_credentials["user_agent"],
    username=reddit_credentials["username"],
    password=reddit_credentials["password"]
)

# Define target subreddits & distress keywords
subreddits = ["depression", "addiction", "suicidewatch"]
keywords = ["depressed", "suicidal", "overwhelmed", "addiction", "help", "relapse", "struggling", "anxiety"]

# Store results
data = []

for subreddit in subreddits:
    for post in reddit.subreddit(subreddit).hot(limit=200):  # Fetch top 200 posts
        # Check if post contains any keyword
        if any(keyword in post.title.lower() or keyword in post.selftext.lower() for keyword in keywords):
            data.append({
                "Post ID": post.id,
                "Timestamp": post.created_utc,
                "Subreddit": subreddit,
                "Title": post.title,
                "Content": post.selftext,
                "Upvotes": post.score,
                "Comments": post.num_comments,
                "Shares": post.num_crossposts,  # Number of times shared
                "URL": post.url
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV & JSON
df.to_csv("data/raw_reddit_posts.csv", index=False)
df.to_json("data/raw_reddit_posts.json", orient="records", indent=2)

print(f"✅ Saved {len(df)} filtered posts to raw_reddit_posts.csv & raw_reddit_posts.json")

######################################################### cleaning data #############################################
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if not isinstance(text, str):  
        return ""  # Handle non-string cases

    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+|\#\w+", "", text)  # Remove @mentions and #hashtags
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = emoji.replace_emoji(text, replace="")  # Remove emojis
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text.strip()

# Load CSV and Apply Cleaning
df = pd.read_csv("data/raw_reddit_posts.csv")  
df["Content"] = df["Content"].astype(str).apply(clean_text)  # Replace raw content
df["Content"].fillna("", inplace=True)

# Save only cleaned content (excluding the original raw data)
df = df[["Content"]]  # Keep only the cleaned text column
df.to_csv("data/cleaned_reddit_posts.csv", index=False, header=True) 

print("✅ Cleaned dataset saved successfully with only cleaned text!")

