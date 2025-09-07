# data/loaders/video_text_dataset.py

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
from transformers import AutoTokenizer # Make sure this import is here

class VideoTextDataset(Dataset):
    def __init__(self, config, split, tokenizer, metadata, vocab):
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.answer_vocab = vocab

        self.data_root = config.data.data_root
        self.video_dir = os.path.join(self.data_root, "videos")
        self.flow_dir = os.path.join(self.data_root, "flow")
        self.deltas_dir = os.path.join(self.data_root, "deltas")
        
        # This update is safe because the factory now controls the vocab and config updates.
        self.config.model.num_answer_classes = len(self.answer_vocab['word_to_id'])
        num_patches = (224 // self.config.model.video_patch_size)**2
        self.config.model.max_seq_len = 1 + self.config.model.text_seq_len + num_patches
        self.config.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.metadata)

    def _load_frames(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2: return None
        
        indices = np.linspace(0, total_frames - 1, self.config.model.frames_per_video, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (224, 224)) / 255.0)
            else:
                # Handle corrupted frames by returning None
                cap.release()
                return None
        cap.release()
        return np.stack(frames)

    def __getitem__(self, idx):
        sample = self.metadata[idx]
        video_id_no_ext = sample['join_key']
        video_filename = f"{video_id_no_ext}.avi"
        caption = sample['Description']

        # Loop to find a valid, uncorrupted video file
        while True:
            frames_np = self._load_frames(os.path.join(self.video_dir, video_filename))
            if frames_np is not None and frames_np.shape[0] == self.config.model.frames_per_video:
                break
            # If the video is corrupted or too short, try the next one in the dataset
            idx = (idx + 1) % len(self)
            sample = self.metadata[idx]
            video_id_no_ext = sample['join_key']
            video_filename = f"{video_id_no_ext}.avi"
            caption = sample['Description']
        
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2)
        tokenized_caption = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.config.model.text_seq_len, return_tensors='pt')
        first_word = caption.split()[0].lower() if caption else '<unk>'
        answer_id = self.answer_vocab['word_to_id'].get(first_word, self.answer_vocab['word_to_id']['<unk>'])
        
        flow_vectors = torch.load(os.path.join(self.flow_dir, f"{video_id_no_ext}.pt"), map_location='cpu')
        frame_deltas = torch.load(os.path.join(self.deltas_dir, f"{video_id_no_ext}.pt"), map_location='cpu')
        
        # --- THE FINAL, VICTORIOUS FIX ---
        # 1. Average the frames along the time dimension to get a representative image.
        avg_frame = torch.tensor(frames_np, dtype=torch.float32).mean(dim=0) # Shape: [H, W, C]
        
        # 2. Now, create patches from this single average image. The shape will be correct.
        raw_patches = rearrange(avg_frame, 
                                '(ph p1) (pw p2) c -> (ph pw) c p1 p2', 
                                p1=self.config.model.video_patch_size, 
                                p2=self.config.model.video_patch_size) # Correctly produces [196, 3, 16, 16]
        
        num_patches, vid_start_idx, full_seq_len = raw_patches.shape[0], 1 + self.config.model.text_seq_len, self.config.model.max_seq_len
        
        flow_full = torch.zeros(full_seq_len, flow_vectors.shape[-1])
        deltas_full = torch.zeros(full_seq_len, frame_deltas.shape[-1])
        patches_full = torch.zeros(full_seq_len, *raw_patches.shape[1:])
        
        flow_full[vid_start_idx : vid_start_idx + num_patches] = flow_vectors
        deltas_full[vid_start_idx : vid_start_idx + num_patches] = frame_deltas
        patches_full[vid_start_idx : vid_start_idx + num_patches] = raw_patches
        
        return {
            'video': frames_tensor, 
            'question_ids': tokenized_caption['input_ids'].squeeze(0),
            'answer_label': torch.tensor(answer_id, dtype=torch.long), 
            'raw_patches': patches_full,
            'flow_vectors': flow_full, 
            'frame_deltas': deltas_full
        }