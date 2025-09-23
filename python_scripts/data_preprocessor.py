import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv('/home/pperez40/symbiosis_llm/data_pre_processing/full_symbiosis_articles.csv')

# Initialize the lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Handle potential non-string input
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize and remove stop words and lemmatize
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]

    return ' '.join(cleaned_tokens)

# Apply the cleaning function to the title and abstract columns
df['cleaned_text'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
df['cleaned_text'] = df['cleaned_text'].apply(clean_text)

# Save the cleaned data to a new CSV file
df[['pmid', 'cleaned_text']].to_csv('/home/pperez40/symbiosis_llm/data_pre_processing/full_cleaned_symbiosis_articles.csv', index=False)

print("Data pre-processing complete. Saved to '/home/pperez40/symbiosis_llm/data_pre_processing/_fullcleaned_symbiosis_articles.csv'")
print("First 5 entries of the cleaned data:")
print(df[['pmid', 'cleaned_text']].head())
