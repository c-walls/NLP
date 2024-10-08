import os
import string
import pickle
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

wikiDS_path = "Resources/Wiki-corpus-20231101"
corpus_path = "Resources/corpus.pkl"

def load_wikipedia_dataset(wikiDS_path):
    if not os.path.exists(wikiDS_path):
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en")
        wiki.save_to_disk(wikiDS_path)
        print(f"Downloaded and saved Wikipedia dataset to '{wikiDS_path}'")
    else:
        wiki = load_from_disk(wikiDS_path)
        print(f"Loaded Wikipedia dataset from '{wikiDS_path}'")
    return wiki

def build_wiki_corpus(corpus_path, wikiDS_path):
    wiki = load_wikipedia_dataset(wikiDS_path)
    translator = str.maketrans('', '', string.punctuation + string.whitespace)
    article_count = len(wiki['train'])
    chunk_size = 1000

    print(f"\nProcessing {article_count} articles from Wikipedia...")
    with open(corpus_path, "wb") as file:
        for i in tqdm(range(0, article_count, chunk_size), desc="Processing articles", unit="chunk"):
            corpus = []
            articles = wiki['train'].select(range(i, min(i + chunk_size, article_count)))
            for article in articles:
                words = article['text'].split()
                processed_words = [word.lower().translate(translator) for word in words if word.isalpha()]
                corpus.extend(processed_words)
            
            pickle.dump(corpus, file)

def main():
    if os.path.exists(corpus_path) and os.path.exists(wikiDS_path):
        print(f"\nWikipedia corpus and dataset already exist:\n'{wikiDS_path}'\n'{corpus_path}'")
        exit(0)
    elif os.path.exists(corpus_path):
        print(f"\nProcessed corpus exists as a .pkl file in '{corpus_path}'.")
        user_input = input("\nWould you like to also download the full unprocessed dataset? [Y/n]: ").strip().lower()
        if user_input == 'y' or user_input == '':
            wiki = load_wikipedia_dataset(wikiDS_path)
        else:
            print("\nDownload Skipped\n.")
            exit(0)
    else:
        build_wiki_corpus(corpus_path, wikiDS_path)

if __name__ == "__main__":
    main()