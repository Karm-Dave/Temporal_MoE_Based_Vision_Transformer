# scripts/build_vocab.py

import os
import json
import yaml
import pandas as pd
from dotmap import DotMap
from collections import Counter

if __name__ == '__main__':
    print("--- Starting Vocabulary Building Script ---")

    # Load the master config to get paths and parameters
    with open("config/training_msvd.yaml") as f:
        config = DotMap(yaml.safe_load(f))
    
    annotation_dir = os.path.join(config.data.data_root, "annotations")
    corpus_path = os.path.join(annotation_dir, "video_corpus.csv")
    vocab_path = os.path.join(annotation_dir, "first_word_vocab.json")

    # 1. Load the single source of truth: the CSV file
    corpus_df = pd.read_csv(corpus_path)
    
    # 2. Filter for only English captions and drop rows with no description
    english_df = corpus_df[corpus_df['Language'] == 'English'].dropna(subset=['Description'])
    
    # 3. Build the vocabulary based on the first word of the description
    print("Building first-word vocabulary...")
    first_words = [sample['Description'].split()[0].lower() for index, sample in english_df.iterrows() if sample['Description']]
    word_counts = Counter(first_words)
    
    # We will use all words that appear more than once to keep the vocab clean
    top_words = [word for word, count in word_counts.items() if count > 1]
    
    word_to_id = {word: i for i, word in enumerate(top_words)}
    word_to_id['<unk>'] = len(word_to_id) # Add an "unknown" token for rare words
    id_to_word = {i: word for word, i in word_to_id.items()}
    
    answer_vocab = {'word_to_id': word_to_id, 'id_to_word': id_to_word}
    
    # 4. Save the vocabulary file for all other scripts to use
    with open(vocab_path, 'w') as f:
        json.dump(answer_vocab, f, indent=4)
        
    print(f"SUCCESS: Saved vocabulary with {len(word_to_id)} words to {vocab_path}")
    print("--- Vocabulary Building Complete ---")