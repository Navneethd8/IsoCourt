import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

class FineBadmintonDataset(Dataset):
    """
    Dataset class for FineBadminton data.
    Supports Multi-Task training for:
    - stroke_type (Main hit_type)
    - stroke_subtype (Detailed variation)
    - technique (Backhand/Forehand)
    - placement (Direction/Characteristics)
    - position (Court Area)
    - intent (Strategy/Tactics)
    - quality (Execution Score)
    """
    def __init__(self, data_root: str, list_file: str, transform=None, sequence_length: int = 16, frame_interval: int = 2):
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_interval = frame_interval
        
        # Define Class Mappings
        self.classes = {
            "stroke_type": [
                "Serve", "Clear", "Smash", "Drop", "Drive", 
                "Net_Shot", "Lob", "Defensive_Shot", "Other"
            ],
            "stroke_subtype": [
                "None", "Short_Serve", "Flick_Serve", "High_Serve",
                "Common_Smash", "Jump_Smash", "Full_Smash", "Stick_Smash", "Slice_Smash",
                "Slice_Drop", "Stop_Drop", "Reverse_Slice_Drop", "Blocked_Drop",
                "Flat_Lift", "High_Lift", "Net_Lift",
                "Attacking_Clear", "Spinning_Net", "Flat_Drive", "High_Drive", "Other"
            ],
            "technique": ["Forehand", "Backhand", "Turnaround", "Unknown"],
            "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head", "Passing_Shot", "Wide", "Unknown"],
            "position": [
                "Mid_Front", "Mid_Court", "Mid_Back", 
                "Left_Front", "Left_Mid", "Left_Back",
                "Right_Front", "Right_Mid", "Right_Back", "Unknown"
            ],
            "intent": [
                "Intercept", "Passive", "Defensive", "To_Create_Depth", 
                "Move_To_Net", "Early_Net_Shot", "Deception", "Hesitation", "Seamlessly", "None"
            ],
            "quality": ["Developing", "Emerging", "Competent", "Proficient", "Advanced", "Expert", "Elite"]
        }
        
        # Build index maps
        self.maps = {k: {cls: i for i, cls in enumerate(v)} for k, v in self.classes.items()}

        if os.path.exists(list_file):
            self.samples = self._load_annotations(list_file)
        else:
            print(f"Warning: Annotation file {list_file} not found.")
            self.samples = []

    def _load_annotations(self, list_file: str) -> List[Dict[str, Any]]:
        import json
        with open(list_file, 'r') as f:
            data = json.load(f)
            
        samples = []
        for video_item in data:
            video_filename = video_item['video']
            if 'hitting' not in video_item: continue
                
            for hit in video_item['hitting']:
                if 'start_frame' not in hit or 'end_frame' not in hit: continue
                    
                samples.append({
                    'video_path': os.path.join(self.data_root, video_filename),
                    'start_frame': hit['start_frame'],
                    'end_frame': hit['end_frame'],
                    'hit_type': hit.get('hit_type', 'Other'),
                    'subtype': hit.get('subtype', []),
                    'player_actions': hit.get('player_actions', []),
                    'shot_characteristics': hit.get('shot_characteristics', []),
                    'ball_area': hit.get('ball_area', 'Unknown'),
                    'strategies': hit.get('strategies', []),
                    'quality': hit.get('quality', 1)
                })
        return samples

    def _map_labels(self, sample: Dict) -> Dict[str, int]:
        # 1. Map Stroke Type
        type_map = {
            "serve": "Serve", "clear": "Clear", "smash": "Smash", "kill": "Smash", 
            "net kill": "Smash", "drop": "Drop", "drop shot": "Drop", "drive": "Drive",
            "net shot": "Net_Shot", "cross-court net shot": "Net_Shot", "lob": "Lob", 
            "push shot": "Lob", "net lift": "Lob", "block": "Defensive_Shot", "defensive shot": "Defensive_Shot"
        }
        raw_type = sample['hit_type'].lower()
        type_mapped = type_map.get(raw_type, "Other")
        
        # 2. Map Subtype (Take first if multiple, else None)
        st_map = {
            "short serve": "Short_Serve", "flick serve": "Flick_Serve", "high serve": "High_Serve",
            "common smash": "Common_Smash", "jump smash": "Jump_Smash", "full smash": "Full_Smash",
            "stick smash": "Stick_Smash", "slice smash": "Slice_Smash", "slice drop shot": "Slice_Drop",
            "stop drop shot": "Stop_Drop", "reverse slice drop shot": "Reverse_Slice_Drop", "blocked drop shot": "Blocked_Drop",
            "flat lift": "Flat_Lift", "high lift": "High_Lift", "net lift": "Net_Lift",
            "attacking clear": "Attacking_Clear", "spinning net": "Spinning_Net", "flat drive": "Flat_Drive", "high drive": "High_Drive"
        }
        subtypes = [s.lower() for s in sample['subtype']]
        st_mapped = st_map.get(subtypes[0], "None") if subtypes else "None"

        # 3. Map Technique (Player Action)
        pa_map = {"forehand": "Forehand", "backhand": "Backhand", "turnaround": "Turnaround"}
        actions = [a.lower() for a in sample['player_actions']]
        pa_mapped = pa_map.get(actions[0], "Unknown") if actions else "Unknown"

        # 4. Map Placement (Shot Characteristic)
        char_map = {
            "straight": "Straight", "cross-court": "Cross-court", "body hit": "Body_Hit",
            "over head": "Over_Head", "passing shot": "Passing_Shot", "wide placement": "Wide"
        }
        chars = [c.lower() for c in sample['shot_characteristics']]
        ch_mapped = char_map.get(chars[0], "Unknown") if chars else "Unknown"

        # 5. Map Position (Ball Area)
        pos_map = {
            "mid front court": "Mid_Front", "mid court": "Mid_Court", "mid back court": "Mid_Back",
            "left front court": "Left_Front", "left mid court": "Left_Mid", "left back court": "Left_Back",
            "right front court": "Right_Front", "right mid court": "Right_Mid", "right back court": "Right_Back"
        }
        pos_mapped = pos_map.get(sample['ball_area'].lower(), "Unknown")

        # 6. Map Intent (Strategy)
        strat_map = {
            "intercept": "Intercept", "passive": "Passive", "defensive": "Defensive",
            "to create depth": "To_Create_Depth", "move to the net": "Move_To_Net",
            "a high net early shot": "Early_Net_Shot", "deception": "Deception",
            "hesitation": "Hesitation", "seamlessly": "Seamlessly"
        }
        strats = [s.lower() for s in sample['strategies']]
        strat_mapped = strat_map.get(strats[0], "None") if strats else "None"

        # Convert back to indices
        return {
            "stroke_type": self.maps["stroke_type"][type_mapped],
            "stroke_subtype": self.maps["stroke_subtype"][st_mapped],
            "technique": self.maps["technique"][pa_mapped],
            "placement": self.maps["placement"][ch_mapped],
            "position": self.maps["position"][pos_mapped],
            "intent": self.maps["intent"][strat_mapped],
            "quality": max(0, min(int(sample['quality']) - 1, 6))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_base = os.path.splitext(os.path.basename(sample['video_path']))[0]
        
        labels_dict = self._map_labels(sample)
        
        # Load as list of PIL images or Tensors for consistent sequence transformation
        frames = self._load_image_frames_as_list(video_base, sample['start_frame'], sample['end_frame'])
        
        if self.transform:
            # Apply transform to each frame. 
            # Note: For ColorJitter, we should ideally use the same parameters for the whole sequence.
            # Some transforms in torchvision.transforms.v2 support this, but for simplicity:
            frames = self.transform(frames)

        # Convert list of tensors to (T, C, H, W)
        if isinstance(frames, list):
            frames = torch.stack(frames)
        
        # Convert labels to tensors
        tensor_labels = {k: torch.tensor(v, dtype=torch.long) for k, v in labels_dict.items()}
        
        return frames, tensor_labels

    def _load_image_frames_as_list(self, video_base: str, start_frame: int, end_frame: int) -> List[torch.Tensor]:
        # Implementation of frame loading as a list of Tensors
        # Images are flat files: {video_base}_{frame_idx}.jpg
        frames = []
        duration = end_frame - start_frame
        if duration <= 0:
            return [torch.zeros((3, 224, 224)) for _ in range(self.sequence_length)]
            
        indices = np.linspace(start_frame, end_frame - 1, self.sequence_length).astype(int)
        
        abs_data_root = os.path.abspath(self.data_root)
        image_dir = os.path.join(abs_data_root, "image")
        
        # Search for image dir if not found in root
        if not os.path.exists(image_dir):
            potential_dirs = [
                os.path.join(abs_data_root, "FineBadminton-master", "dataset", "image"),
                os.path.join(abs_data_root, "dataset", "image")
            ]
            for p in potential_dirs:
                if os.path.exists(p):
                    image_dir = p
                    break

        for idx in indices:
            img_path = os.path.join(image_dir, f"{video_base}_{idx}.jpg")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                frames.append(img_tensor)
            else:
                frames.append(torch.zeros((3, 224, 224)))
                
        return frames

    def _load_image_frames(self, video_base: str, start_frame: int, end_frame: int) -> torch.Tensor:
        abs_data_root = os.path.abspath(self.data_root)
        image_dir = os.path.join(abs_data_root, "image")
        
        if not os.path.exists(image_dir):
             potential_paths = [
                 os.path.join(abs_data_root, "FineBadminton-master", "dataset", "image"),
                 os.path.join(abs_data_root, "dataset", "image"),
                 os.path.join(abs_data_root, "../data/FineBadminton-master/dataset/image")
             ]
             for path in potential_paths:
                 if os.path.exists(path):
                     image_dir = path
                     break
        
        duration = end_frame - start_frame
        if duration <= 0:
             return torch.zeros((self.sequence_length, 3, 224, 224), dtype=torch.float32)
             
        indices = np.linspace(start_frame, end_frame - 1, self.sequence_length).astype(int)
        
        frames = []
        for idx in indices:
            img_path = os.path.join(image_dir, f"{video_base}_{idx}.jpg")
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames_np = np.array(frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        return frames_tensor.permute(0, 3, 1, 2)

if __name__ == "__main__":
    print("Dataset class refined with Multi-Task support.")
