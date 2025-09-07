# data/dataset_factory.py
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import Counter
import numpy as np
from .loaders.video_text_dataset import VideoTextDataset

def get_and_split_data(config):
    # ... (most of the function is the same)
    annotation_dir = os.path.join(config.data.data_root, "annotations")
    corpus_path = os.path.join(annotation_dir, "video_corpus.csv")
    vocab_path = os.path.join(annotation_dir, "first_word_vocab.json")
    
    print("--- Central Data Factory: Loading and Verifying Data ---")
    corpus_df = pd.read_csv(corpus_path)
    
    # To prevent SettingWithCopyWarning, work on a copy for modifications
    english_df = corpus_df[corpus_df['Language'] == 'English'].copy()
    
    # Safely convert to numeric, making invalid entries NaN
    english_df['Start'] = pd.to_numeric(english_df['Start'], errors='coerce')
    english_df['End'] = pd.to_numeric(english_df['End'], errors='coerce')
    english_df.dropna(subset=['VideoID', 'Start', 'End', 'Description'], inplace=True)
    
    # THIS IS THE FIX FOR THE WARNING
    english_df.loc[:, 'join_key'] = english_df.apply(
        lambda row: f"{row['VideoID']}_{int(row['Start'])}_{int(row['End'])}", axis=1
    )
    # ... (rest of the function is the same, no changes needed)
    flow_dir_path = os.path.join(config.data.data_root, "flow")
    if not os.path.exists(flow_dir_path):
        raise NotADirectoryError(f"CRITICAL ERROR: Directory for flow vectors not found at {flow_dir_path}")

    existing_feature_keys = {os.path.splitext(f)[0] for f in os.listdir(flow_dir_path)}
    metadata = english_df[english_df['join_key'].isin(existing_feature_keys)].to_dict('records')
    
    if not metadata:
        raise FileNotFoundError("CRITICAL ERROR: Zero matching videos found with pre-processed features.")
    print(f"SUCCESS: Found {len(metadata)} valid video/caption pairs.")
    
    unique_video_ids = sorted(list({sample['VideoID'] for sample in metadata}))
    np.random.RandomState(42).shuffle(unique_video_ids)
    train_ratio, val_ratio, _ = config.data.train_val_test_split
    train_end = int(len(unique_video_ids) * train_ratio)
    val_end = train_end + int(len(unique_video_ids) * val_ratio) # Simplified val_ratio logic
    train_ids, val_ids, test_ids = set(unique_video_ids[:train_end]), set(unique_video_ids[train_end:val_end]), set(unique_video_ids[val_end:])
    
    train_meta = [m for m in metadata if m['VideoID'] in train_ids]
    val_meta = [m for m in metadata if m['VideoID'] in val_ids]
    test_meta = [m for m in metadata if m['VideoID'] in test_ids]

    print("Building vocabulary from training split...")
    first_words = [s['Description'].split()[0].lower() for s in train_meta if s.get('Description')]
    word_counts = Counter(first_words)
    # Get top 999 answers, plus one for the '<unk>' token.
    top_words = [word for word, count in word_counts.most_common(999)]
    word_to_id = {'<unk>': 0}
    for i, word in enumerate(top_words):
        word_to_id[word] = i + 1
    id_to_word = {i: word for word, i in word_to_id.items()}
    vocab = {'word_to_id': word_to_id, 'id_to_word': id_to_word}
    
    with open(vocab_path, 'w') as f: json.dump(vocab, f, indent=4)
    print(f"Saved vocabulary ({len(word_to_id)} words) to {vocab_path}")
    
    return train_meta, val_meta, test_meta, vocab

def create_dataloader(config, split, metadata, vocab):
    tokenizer = AutoTokenizer.from_pretrained(config.data.text_tokenizer)
    config.tokenizer_vocab_size = tokenizer.vocab_size
    dataset = VideoTextDataset(config, split, tokenizer, metadata, vocab)
    return DataLoader(
        dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=(split == 'train')
    )