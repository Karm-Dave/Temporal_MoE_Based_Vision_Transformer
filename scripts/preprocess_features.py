# scripts/preprocess_features.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import yaml
from tqdm import tqdm
from dotmap import DotMap
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from einops import rearrange

def get_device(): return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_video_frames(video_path, num_frames, resize_shape=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2: return None
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = TF.to_tensor(cv2.resize(frame, resize_shape))
            frames.append(frame_tensor)
        else:
            cap.release(); return None # Return None if a frame fails to load
    cap.release()
    return torch.stack(frames) if len(frames) == num_frames else None

def compute_optical_flow(frames, raft_model, device, config):
    # --- THIS IS THE FINAL, CORRECTED LOGIC ---
    T_frames, C, H, W = frames.shape
    patch_size = config.model.video_patch_size
    num_patches_per_frame = (H // patch_size) * (W // patch_size)

    # Move frames to device and prepare for RAFT model (needs B, T, C, H, W)
    frames = frames.to(device) * 255.0 # RAFT expects 0-255 range
    
    # Process pairs of frames
    frame_pairs_1 = frames[:-1]
    frame_pairs_2 = frames[1:]
    
    with torch.no_grad():
        # Get flow predictions from the RAFT model for all pairs at once
        flow_preds = raft_model(frame_pairs_1, frame_pairs_2)[-1] # Get the last, most refined flow
    
    # Average the flow maps across the time dimension
    # Shape: [T-1, 2, H, W] -> [2, H, W]
    avg_flow_map = torch.mean(flow_preds, dim=0)

    # Use einops to perfectly average the flow within each patch grid
    # This reshapes the flow map into patches and then takes the mean
    # c=2 (x,y flow), H=(ph p1), W=(pw p2) -> ph pw (c p1 p2) -> ph*pw c
    avg_flow_per_patch = rearrange(avg_flow_map, 'c (ph p1) (pw p2) -> (ph pw) (c p1 p2)', p1=patch_size, p2=patch_size).mean(dim=-1)

    flow_dim = config.model.moe.experts.motion.flow_dim
    # Pad the 2D flow vector to the desired final dimension
    return F.pad(avg_flow_per_patch, (0, flow_dim - 2)).cpu()

def compute_frame_deltas(frames, config):
    num_patches_per_frame = (224 // config.model.video_patch_size) ** 2
    # Calculate the average pixel change between frames
    deltas = (frames[1:] - frames[:-1]).abs().mean()
    # Create a tensor of this delta value for each patch
    return torch.full((num_patches_per_frame, config.model.moe.experts.fast_change.delta_dim), deltas.item()).cpu()

if __name__ == '__main__':
    with open("config/training_msvd.yaml") as f: config = DotMap(yaml.safe_load(f))
    
    DATA_ROOT, VIDEO_DIR = config.data.data_root, os.path.join(config.data.data_root, "videos")
    FLOW_DIR, DELTAS_DIR = os.path.join(DATA_ROOT, "flow"), os.path.join(DATA_ROOT, "deltas")
    os.makedirs(FLOW_DIR, exist_ok=True); os.makedirs(DELTAS_DIR, exist_ok=True)
    
    device = get_device(); print(f"Using device: {device}")
    weights = Raft_Small_Weights.DEFAULT; raft_model = raft_small(weights=weights).to(device); raft_model.eval()
    
    video_filenames = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.avi')]
    print(f"Found {len(video_filenames)} AVI files to process.")

    for video_filename in tqdm(video_filenames, desc="Processing videos"):
        video_id_no_ext = os.path.splitext(video_filename)[0]
        flow_path = os.path.join(FLOW_DIR, f"{video_id_no_ext}.pt")
        delta_path = os.path.join(DELTAS_DIR, f"{video_id_no_ext}.pt")
        
        # We will re-run everything, so let's skip existing files if you want to save time,
        # but it's safer to re-compute all to ensure they are correct.
        # if os.path.exists(flow_path) and os.path.exists(delta_path): continue

        video_path = os.path.join(VIDEO_DIR, video_filename)
        frames = load_video_frames(video_path, config.model.frames_per_video)
        if frames is None:
            print(f"Warning: Could not load frames for {video_filename}. Skipping.")
            continue
            
        flow_vectors = compute_optical_flow(frames, raft_model, device, config)
        delta_vectors = compute_frame_deltas(frames, config)
        torch.save(flow_vectors, flow_path)
        torch.save(delta_vectors, delta_path)
            
    print("\nPre-computation of REAL feature files complete.")