# check_filenames.py
# A dedicated script to diagnose the filename mismatch problem.

import os
import yaml
import pandas as pd
from dotmap import DotMap

if __name__ == '__main__':
    print("\n" + "="*50)
    print("--- FILENAME MISMATCH DIAGNOSTIC SCRIPT ---")
    print("="*50)
    
    # 1. Load the same configuration file used by your scripts
    try:
        with open("config/training_msvd.yaml") as f:
            config = DotMap(yaml.safe_load(f))
    except FileNotFoundError:
        print("\nFATAL ERROR: Could not find 'config/training_msvd.yaml'. Make sure you are in the correct directory.")
        exit()

    # 2. Get the path to your videos
    DATA_ROOT = config.data.data_root
    VIDEO_DIR = os.path.join(DATA_ROOT, "videos")
    ANNOTATION_DIR = os.path.join(DATA_ROOT, "annotations")
    
    print(f"\nConfiguration loaded. Checking for video files in:\n>>> {VIDEO_DIR}\n")
    
    # 3. Get the first few filenames that ACTUALLY exist on your hard drive
    try:
        existing_filenames = os.listdir(VIDEO_DIR)
        print("--- 1. Files found on your DISK (sample of first 5): ---")
        if existing_filenames:
            for fname in existing_filenames[:5]:
                print(f"  - {fname}")
        else:
            print("\nFATAL ERROR: The 'videos' directory is empty or the path is incorrect!")
            print(f"Please check the 'data_root' path in your config file: {config.data.data_root}")
            exit()
    except FileNotFoundError:
        print(f"\nFATAL ERROR: The directory specified for videos does not exist: {VIDEO_DIR}")
        print(f"Please make sure the 'data_root' path in your config file is correct: {config.data.data_root}")
        exit()
        
    # 4. Get the first few filenames the code is TRYING TO CREATE from the CSV
    print("\n--- 2. Filenames GENERATED from the CSV file (sample of first 5): ---")
    try:
        corpus_path = os.path.join(ANNOTATION_DIR, "video_corpus.csv")
        corpus_df = pd.read_csv(corpus_path)
        
        # Clean the dataframe just like the main script does
        corpus_df['Start'] = pd.to_numeric(corpus_df['Start'], errors='coerce')
        corpus_df['End'] = pd.to_numeric(corpus_df['End'], errors='coerce')
        english_df = corpus_df.dropna(subset=['VideoID', 'Start', 'End', 'Language'])
        english_df = english_df[english_df['Language'] == 'English']
        
        generated_filenames = []
        for index, row in english_df.head(5).iterrows(): # Show first 5
            # This is the exact logic from your dataloader
            fname = f"{row['VideoID']}_{int(row['Start'])}_{int(row['End'])}.avi"
            generated_filenames.append(fname)
            print(f"  - {fname}")
            
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not find 'video_corpus.csv' at: {corpus_path}")
        exit()
        
    # 5. The Final Check
    print("\n--- 3. COMPARISON ---")
    if set(generated_filenames).isdisjoint(set(existing_filenames)):
        print("RESULT: ❌ As expected, there are NO matches.")
        print("Please carefully compare the filenames in list #1 and list #2 to find the difference.")
        print("The most likely issue is the VideoID format (e.g., 'vid1234' vs 'video1234') or the path.")
    else:
        print("RESULT: ✅ A match was found! This is unexpected. Please check the 'data_root' path.")
        
    print("\n" + "="*50)