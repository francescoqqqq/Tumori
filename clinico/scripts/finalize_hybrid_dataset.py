import os
import shutil
import json
import random
import csv

def setup_nnunet_hybrid(base_data_dir="data", nnunet_raw_dir="nnUNet_raw", task_id=500, task_name="Hybrid_BraTS"):
    """
    Combines Synthetic and Real data into nnU-Net format and creates splits.
    Moves files to save space (instead of copying).
    """
    
    # Define paths
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    target_base = os.path.join(nnunet_raw_dir, dataset_name)
    target_imagesTr = os.path.join(target_base, "imagesTr")
    target_labelsTr = os.path.join(target_base, "labelsTr")
    
    os.makedirs(target_imagesTr, exist_ok=True)
    os.makedirs(target_labelsTr, exist_ok=True)
    
    # --- 1. Gather and Move Files ---
    
    # List of all cases (id, type)
    all_cases = []
    
    # Process Synthetic
    synth_dir = os.path.join(base_data_dir, "synthetic")
    if os.path.exists(synth_dir):
        print("Moving Synthetic data...")
        # Read metadata if exists, else default to 'frankenstein'
        synth_meta = {}
        csv_path = os.path.join(synth_dir, 'synthetic_metadata.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    synth_meta[row['id']] = row['category']
        
        # Move files
        for f in os.listdir(os.path.join(synth_dir, "images")):
            shutil.move(os.path.join(synth_dir, "images", f), os.path.join(target_imagesTr, f))
        for f in os.listdir(os.path.join(synth_dir, "labels")):
            shutil.move(os.path.join(synth_dir, "labels", f), os.path.join(target_labelsTr, f))
            # Add to list (remove .nii.gz)
            case_id = f.replace(".nii.gz", "")
            # Default to frankenstein if not in metadata
            cat = synth_meta.get(case_id, 'frankenstein')
            all_cases.append({'id': case_id, 'source': 'synthetic', 'cat': cat})
            
    # Process Real
    real_dir = os.path.join(base_data_dir, "brats_real")
    if os.path.exists(real_dir):
        print("Moving Real BraTS data...")
        for f in os.listdir(os.path.join(real_dir, "images")):
            shutil.move(os.path.join(real_dir, "images", f), os.path.join(target_imagesTr, f))
        for f in os.listdir(os.path.join(real_dir, "labels")):
            shutil.move(os.path.join(real_dir, "labels", f), os.path.join(target_labelsTr, f))
            case_id = f.replace(".nii.gz", "")
            all_cases.append({'id': case_id, 'source': 'real', 'cat': 'real'})

    print(f"Total cases assembled: {len(all_cases)}")
    
    # --- 2. Create Splits ---
    # Goal: 140 Total. Test=10, Val=30, Train=100.
    # Test set must be representative:
    # - 3 Real
    # - 7 Synthetic (2 small, 2 medium, 1 large, 2 complex)
    
    test_set = []
    val_set = []
    train_set = []
    
    # Group by category
    groups = {}
    for c in all_cases:
        key = c['cat']
        if key not in groups: groups[key] = []
        groups[key].append(c['id'])
        
    # Shuffle all groups
    for k in groups: random.shuffle(groups[k])
    
    # Select Test Set (Fixed strategy)
    # Real: 3
    if len(groups.get('real', [])) >= 3:
        test_set.extend(groups['real'][:3])
        groups['real'] = groups['real'][3:]
    else:
        # Fallback if few real cases
        test_set.extend(groups.get('real', []))
        groups['real'] = []
    
    # Synthetic (Frankenstein)
    # Take 7 random synthetic cases
    synth_keys = [k for k in groups.keys() if k != 'real']
    all_synth = []
    for k in synth_keys:
        all_synth.extend(groups[k])
    
    random.shuffle(all_synth)
    test_set.extend(all_synth[:7])
    
    # Remaining go to pool
    remaining_synth = all_synth[7:]
    
    # Remaining for Train/Val (random split of what's left)
    remaining = []
    remaining.extend(groups.get('real', []))
    remaining.extend(remaining_synth)
    random.shuffle(remaining)
    
    val_set = remaining[:30]
    train_set = remaining[30:]
    
    print(f"Split counts: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # --- 3. Create dataset.json ---
    json_dict = {
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR"
        },
        "labels": {
            "background": 0,
            "necrotic": 1,
            "edema": 2,
            "enhancing": 3
        },
        "numTraining": len(all_cases),
        "file_ending": ".nii.gz",
        "name": task_name,
        "description": "Hybrid Synthetic-Real Dataset for Loss Function Benchmarking",
        # Note: nnU-Net v2 usually expects splits in a separate splits_final.json file,
        # but listing all here is standard for the dataset.json.
        # We will save the split info to a separate CSV for the user to use during training command.
    }
    
    with open(os.path.join(target_base, "dataset.json"), "w") as f:
        json.dump(json_dict, f, indent=4)
        
    # Save Split Info for reference
    split_info = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }
    with open("hybrid_splits.json", "w") as f:
        json.dump(split_info, f, indent=4)
        
    print(f"Dataset ready at: {target_base}")
    print("Splits saved to hybrid_splits.json")

if __name__ == "__main__":
    setup_nnunet_hybrid()

