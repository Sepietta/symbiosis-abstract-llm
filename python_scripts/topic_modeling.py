import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import warnings
import os

warnings.filterwarnings('ignore')

# Load the cleaned data
df = pd.read_csv('/home/pperez40/symbiosis_llm/data_pre_processing/full_cleaned_symbiosis_articles.csv')

# Ensure there is no empty text and convert to a list of lists of words
texts = [text.split() for text in df['cleaned_text'] if isinstance(text, str)]

# Create a dictionary from the documents
dictionary = Dictionary(texts)

# Create a corpus (a bag-of-words representation of the documents)
corpus = [dictionary.doc2bow(text) for text in texts]

# Set the number of topics you want to find
num_topics = 10  # We will start with a small number

# Build the LDA model
lda_model = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=num_topics,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# Print the keywords for each topic
print("Top words for each topic:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")

# For visualization, create a directory if it doesn't exist
if not os.path.exists('lda_vis'):
    os.makedirs('lda_vis')

# Prepare the data for visualization
vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# Save the visualization to an HTML file
pyLDAvis.save_html(vis_data, 'lda_vis/lda_visualization.html')

print("\nLDA visualization saved to lda_vis/lda_visualization.html. You can view it by downloading the file and opening it in your web browser.")
