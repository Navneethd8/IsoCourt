import torch
from backend.dataset import FineBadmintonDataset
from backend.model import CNN_LSTM_Model
from torch.utils.data import DataLoader
import os

def check_dataset_and_model():
    print("Checking dataset...")
    data_root = "backend/data"
    list_file = "backend/data/transformed_combined_rounds_output_en_evals_translated.json"
    
    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found")
        return

    try:
        dataset = FineBadmintonDataset(data_root=data_root, list_file=list_file)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) == 0:
            print("Dataset is empty!")
            return

        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        print("Fetching one batch...")
        frames, action_labels, quality_labels = next(iter(loader))
        
        print(f"Frames shape: {frames.shape}")
        print(f"Action labels shape: {action_labels.shape}, Type: {action_labels.dtype}")
        print(f"Quality labels shape: {quality_labels.shape}, Type: {quality_labels.dtype}")
        
        if quality_labels.dtype != torch.long:
             print("ERROR: Quality labels are not LongTensor!")
        else:
             print("SUCCESS: Quality labels are LongTensor.")

        print("Checking model forward pass...")
        num_actions = len(dataset.classes)
        num_qualities = 7
        
        model = CNN_LSTM_Model(num_classes=num_actions, num_quality_classes=num_qualities)
        
        # Forward pass
        action_preds, quality_preds = model(frames)
        print(f"Action preds shape: {action_preds.shape}")
        print(f"Quality preds shape: {quality_preds.shape}")
        
        # Loss check
        criterion_quality = torch.nn.CrossEntropyLoss()
        loss_q = criterion_quality(quality_preds, quality_labels)
        print(f"Quality loss: {loss_q.item()}")
        
        print("Verification successful!")

    except Exception as e:
        print(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dataset_and_model()
