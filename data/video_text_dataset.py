import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import cv2
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class VideoTextDataset(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.text_tokenizer)

        self.matadata = self.load_metadata()

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        sample_info = self.metadata[idx]

        video_frames = self._load_sample_frames(sample_info['video_path'])


        flow_vectors = torch.load(sample_info['flow_path'])
        frame_deltas = torch.load(sample_info['delta_path'])

        raw_patches = self._extract_raw_patches(video_frames)

        tokenized = self.tokenizer(
            sample_info['question'],
            padding='max_length',
            truncation=True,
            max_length=self.config.data.max_seq_len,
            return_tensors='pt'
        )

        batch = {
            'video': video_frames,
            'question_ids': tokenized['input_ids'].squeeze(0),
            'raw_patches': raw_patches,
            'flow_vectors': flow_vectors,
            'frame_deltas': frame_deltas,
            'answer_label': torch.tensor(sample_info['answer_id'], dtype=torch.long)
        }

        return batch
    
    def _load_metadata(self):
        print(f"Loading data for {self.split} split ...")

        return []
    
    def _load_and_sample_frames(self,path):
        # video = cv2.VideoCapture(path)
        # frames = []
        # while True:
        #     ret, frame = video.read()
        #     if not ret:
        #         break
        #     frames.append(frame)
        
        # video.release()

        # if len(frames) > self.config.data.max_frames:
        #     indices = np.linspace(0, len(frames) - 1, self.config.data.max_frames, dtype=int)
        #     frames = [frames[i] for i in indices]
        
        # return torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        return torch.randn(self.config.data.max_frames, 3, 224, 224)
    
    def _extract_raw_patches(self, video_frames):
        # Assuming video_frames is of shape (num_frames, channels, height, width)
        # num_frames, channels, height, width = video_frames.shape
        # patch_size = self.config.model.video_patch_size
        # patches = rearrange(video_frames, 'f c (h ph) (w pw) -> f (ph pw) c h w', ph=height // patch_size, pw=width // patch_size)
        # return patches
        return torch.randn(196, 16, 16, 3)  # Dummy patches for now