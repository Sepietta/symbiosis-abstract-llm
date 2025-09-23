import pandas as pd
from Bio import Entrez
import time
import re
import os

Entrez.email = "pperez40@ucmerced.edu"

# Define a list of search terms
search_terms = [
    '"symbiosis" AND "gene"',
    '"mutualism" AND "gene"',
    '"commensalism" AND "gene"',
    '"parasitism" AND "gene"',
    '"mycorrhiza" AND "gene"',
    '"Rhizobium" AND "gene"',
    '"lichen" AND "gene"',
    '"coral" AND "algae" AND "symbiosis" AND "gene"',
    '"host-microbe interaction" AND "gene"',
    '"Euprymna scolopes" AND "Vibrio fischeri" AND "symbiosis" AND "gene"',
    '"bobtail squid" AND "bioluminescence" AND "symbiosis" AND "gene"'
]

def fetch_and_parse_articles(search_term, retmax=200):
    """
    Fetches and parses articles from PubMed for a given search term.
    """
    try:
        # Search for article IDs
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=retmax, idtype="acc")
        record = Entrez.read(handle)
        handle.close()
        id_list = record["IdList"]

        if not id_list:
            print(f"No articles found for '{search_term}'.")
            return []

        print(f"Found {len(id_list)} articles for '{search_term}'. Fetching details...")

        # Fetch full article details in batches to avoid overwhelming the server
        parsed_data = []
        batch_size = 250
        for i in range(0, len(id_list), batch_size):
            batch_ids = id_list[i:i + batch_size]
            fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, rettype="medline", retmode="xml")
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()

            for article in articles.get('PubmedArticle', []):
                pubmed_data = article['MedlineCitation']['Article']
                title = pubmed_data.get('ArticleTitle', 'No Title')
                abstract_list = pubmed_data.get('Abstract', {}).get('AbstractText', [])
                if isinstance(abstract_list, list):
                    abstract = ' '.join(str(s) for s in abstract_list)
                else:
                    abstract = abstract_list

                parsed_data.append({
                    'search_term': search_term,
                    'title': str(title).strip(),
                    'abstract': str(abstract).strip(),
                    'pmid': article['MedlineCitation']['PMID']
                })
            
            time.sleep(1) # Be a good citizen and wait between requests

        return parsed_data

    except Exception as e:
        print(f"An error occurred while fetching for '{search_term}': {e}")
        return []

# Main execution loop
all_articles = []
for term in search_terms:
    articles = fetch_and_parse_articles(term)
    all_articles.extend(articles)
    time.sleep(2) # Wait a bit longer between different search terms

# Convert to a pandas DataFrame and save
if all_articles:
    df = pd.DataFrame(all_articles)
    output_file = '/home/pperez40/symbiosis_llm/data_pre_processing/full_symbiosis_articles.csv'
    
    # Clean the data of non-ASCII characters
    for col in ['title', 'abstract']:
        df[col] = df[col].apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))

    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(df)} articles to {output_file}")
else:
    print("No articles were collected. Please check your search terms and network connection.")
