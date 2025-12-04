import os
import json
import glob

def sanitize_dataset():
    base = "nnUNet_raw/Dataset500_Hybrid_BraTS"
    images_dir = os.path.join(base, "imagesTr")
    labels_dir = os.path.join(base, "labelsTr")
    
    # Get all labels
    labels = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]
    valid_cases = []
    
    print(f"Scanning {len(labels)} potential cases...")
    
    for label in labels:
        case_id = label.replace(".nii.gz", "")
        
        # Check if all 4 images exist
        is_complete = True
        missing = []
        for i in range(4):
            img_path = os.path.join(images_dir, f"{case_id}_{i:04d}.nii.gz")
            if not os.path.exists(img_path):
                is_complete = False
                missing.append(f"{case_id}_{i:04d}.nii.gz")
        
        if is_complete:
            valid_cases.append(case_id)
        else:
            print(f"Removing incomplete case {case_id}. Missing: {missing}")
            # Delete incomplete files to clean up
            os.remove(os.path.join(labels_dir, label))
            for i in range(4):
                p = os.path.join(images_dir, f"{case_id}_{i:04d}.nii.gz")
                if os.path.exists(p):
                    os.remove(p)

    # Check for orphan images (images without label)
    all_images = glob.glob(os.path.join(images_dir, "*.nii.gz"))
    for img_path in all_images:
        filename = os.path.basename(img_path)
        # BraTS_Real_XXX_0000.nii.gz -> BraTS_Real_XXX
        # Remove last _XXXX.nii.gz
        parts = filename.split("_")
        case_id = "_".join(parts[:-1]) # This handles IDs with underscores
        
        if case_id not in valid_cases:
            print(f"Removing orphan image: {filename}")
            os.remove(img_path)
            
    print(f"Sanitization complete. Valid cases: {len(valid_cases)}")
    
    # Update dataset.json
    json_path = os.path.join(base, "dataset.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    old_count = data['numTraining']
    data['numTraining'] = len(valid_cases)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Updated dataset.json: {old_count} -> {len(valid_cases)}")

if __name__ == "__main__":
    sanitize_dataset()

