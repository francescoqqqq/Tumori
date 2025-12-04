import synapseclient
import zipfile
import os
import shutil

def download_stratified_subset(synapse_id, target_dir_images, target_dir_labels, total_cases=40, step=31):
    syn = synapseclient.Synapse()
    try:
        syn.login()
    except Exception:
        print("Login failed. Trying with prompt...")
        try:
            token = input("Please enter your Synapse Auth Token: ")
            syn.login(authToken=token)
        except Exception as e:
            print(f"Login still failed: {e}")
            return

    print(f"Downloading BraTS ZIP: {synapse_id}...")
    file_entity = syn.get(synapse_id, downloadLocation=".")
    zip_path = file_entity.path
    
    print("Scanning ZIP contents...")
    extracted_count = 0
    
    # Mappings for BraTS suffixes to nnU-Net integers
    # t1n -> 0000, t1c -> 0001, t2w -> 0002, t2f -> 0003
    # seg -> label file
    suffix_map = {
        't1n': '0000',
        't1c': '0001',
        't2w': '0002', 
        't2f': '0003'
    }

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get list of files, sorted
        all_files = sorted(zf.namelist())
        
        # Identify unique patients (directories)
        # Assuming structure: Parent/PatientID/Files
        patient_folders = set()
        for f in all_files:
            parts = f.split('/')
            for p in parts:
                if "BraTS-GLI" in p and not p.endswith('.zip'):
                    patient_folders.add(p)
                    
        sorted_patients = sorted(list(patient_folders))
        print(f"Found {len(sorted_patients)} total patients in ZIP.")
        
        # Select indices
        selected_patients = []
        for i in range(0, len(sorted_patients), step):
            if len(selected_patients) < total_cases:
                selected_patients.append(sorted_patients[i])
            else:
                break
                
        print(f"Selected {len(selected_patients)} patients for extraction (Step={step}).")
        
        # Extract and Rename
        for idx, patient_id in enumerate(selected_patients):
            # New ID for nnU-Net: BraTS_Real_001, etc.
            new_id = f"BraTS_Real_{idx+1:03d}"
            print(f"Processing {patient_id} -> {new_id} ({idx+1}/{len(selected_patients)})")
            
            # Find files for this patient
            patient_files = [f for f in all_files if patient_id in f and f.endswith('.nii.gz')]
            
            for file_path in patient_files:
                filename = os.path.basename(file_path)
                
                # Determine modality
                dest_path = None
                
                if 'seg' in filename:
                    # Label file
                    dest_path = os.path.join(target_dir_labels, f"{new_id}.nii.gz")
                else:
                    # Image file
                    for suffix, code in suffix_map.items():
                        if suffix in filename:
                            dest_path = os.path.join(target_dir_images, f"{new_id}_{code}.nii.gz")
                            break
                
                if dest_path:
                    # Extract to temp, then move/rename
                    source = zf.open(file_path)
                    with open(dest_path, "wb") as target:
                        shutil.copyfileobj(source, target)

    # Clean up
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("Removed ZIP file.")

if __name__ == "__main__":
    # Create directories
    os.makedirs("data/brats_real/images", exist_ok=True)
    os.makedirs("data/brats_real/labels", exist_ok=True)
    
    # BraTS 2023 Training Data ID
    download_stratified_subset("syn51514132", "data/brats_real/images", "data/brats_real/labels")

