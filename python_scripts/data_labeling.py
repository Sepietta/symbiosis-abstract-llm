import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import re

# Load the cleaned data
df = pd.read_csv('/home/pperez40/symbiosis_llm/data_pre_processing/full_cleaned_symbiosis_articles.csv')

# Ensure there is no empty text and convert to a list of lists of words
texts = [text.split() for text in df['cleaned_text'] if isinstance(text, str)]

# Create a dictionary from the documents
dictionary = Dictionary(texts)

# Create a corpus (bag-of-words representation)
corpus = [dictionary.doc2bow(text) for text in texts]

# Load the previously trained LDA model
# Make sure this path is correct
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=10,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)


# Function to get the main topic for each document
def get_main_topic(lda_model, bow_vector):
    # Get all the topics for a given document
    topics_per_doc = lda_model.get_document_topics(bow_vector)
    
    # Return the topic with the highest score
    if not topics_per_doc:
        return -1
    else:
        return max(topics_per_doc, key=lambda item: item[1])[0]

# Apply the function to each document in the corpus
df['topic_id'] = [get_main_topic(lda_model, doc) for doc in corpus]

# Let's map the topic IDs to human-readable names
topic_names = {
    0: 'Sponge_Symbiosis_Genomics',
    1: 'Lichen_Symbiosis_Genomics',
    2: 'Common_Bean_Nitrogen_Fixation',
    3: 'General_Gene_Expression',
    4: 'Legume_Rhizobium_Symbiosis',
    5: 'Squid_Vibrio_Symbiosis',
    6: 'Human_Microbiome',
    7: 'Environmental_Microbes_Ecology',
    8: 'Plant_Mycorrhizal_Symbiosis',
    9: 'Coral_Algae_Symbiosis'
}

df['topic_label'] = df['topic_id'].map(topic_names)

# Save the final labeled dataset
df[['pmid', 'cleaned_text', 'topic_label']].to_csv('/home/pperez40/symbiosis_llm/data_pre_processing/labeled_symbiosis_data.csv', index=False)

print("Data labeling complete. Saved to '/home/pperez40/symbiosis_llm/data_pre_processing/labeled_symbiosis_data.csv'")
print("\nFinal labeled data (first 5 rows):")
print(df[['pmid', 'cleaned_text', 'topic_label']].head())
