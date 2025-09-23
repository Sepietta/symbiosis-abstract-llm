from Bio import Entrez
import pandas as pd
import time

# Set your email address. This is a requirement for using the NCBI Entrez API.
Entrez.email = "your.email@example.com"

# Search query for symbiosis-related papers
search_term = '"symbiosis" AND "gene"'

# Function to fetch and parse abstracts
def fetch_abstracts(search_term, retmax=10):
    try:
        # Search for articles
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=retmax, idtype="acc")
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        if not id_list:
            print("No articles found for the search term.")
            return []

        print(f"Found {len(id_list)} articles. Fetching details...")

        # Fetch full article details
        fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="xml")

        articles = Entrez.read(fetch_handle)
        fetch_handle.close()

        parsed_data = []
        for article in articles['PubmedArticle']:
            pubmed_data = article['MedlineCitation']['Article']

            title = pubmed_data.get('ArticleTitle', 'No Title')

            abstract_list = pubmed_data.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_list, list):
                abstract = ' '.join(str(s) for s in abstract_list)
            else:
                abstract = abstract_list

            parsed_data.append({
                'title': title,
                'abstract': abstract,
                'pmid': article['MedlineCitation']['PMID']
            })

        return parsed_data

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Fetch the data
abstracts_list = fetch_abstracts(search_term, retmax=10)

# Convert to a pandas DataFrame and save
if abstracts_list:
    df = pd.DataFrame(abstracts_list)

    for col in ['title', 'abstract']:
        df[col] = df[col].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

    output_file = '/home/pperez40/symbiosis_llm/data_pre_processing/symbiosis_articles.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved {len(df)} articles to {output_file}")
    print("First 5 rows of the DataFrame:")
    print(df.head())
