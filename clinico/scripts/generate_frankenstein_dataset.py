import os
import numpy as np
import nibabel as nib
import random
from tqdm import tqdm

def get_complete_cases(images_dir, labels_dir):
    """
    Returns a list of Case IDs that have ALL 4 modalities AND a label file.
    """
    valid_ids = []
    
    # Check all potential label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]
    
    for lf in label_files:
        case_id = lf.replace(".nii.gz", "")
        
        # Check if all 4 images exist
        is_complete = True
        for i in range(4):
            img_name = f"{case_id}_{i:04d}.nii.gz"
            if not os.path.exists(os.path.join(images_dir, img_name)):
                is_complete = False
                break
        
        if is_complete:
            valid_ids.append(case_id)
            
    return sorted(valid_ids)

def load_case(case_path_prefix):
    """Loads a BraTS case (4 modalities + seg)."""
    images = []
    for i in range(4):
        img_path = f"{case_path_prefix}_{i:04d}.nii.gz"
        img = nib.load(img_path).get_fdata()
        images.append(img)
    
    label_path = case_path_prefix.replace("images", "labels") + ".nii.gz"
    seg = nib.load(label_path).get_fdata()
    
    return np.stack(images, axis=0), seg, nib.load(label_path).affine

def extract_tumor(images, seg):
    tumor_coords = np.argwhere(seg > 0)
    if len(tumor_coords) == 0:
        return None, None
        
    min_z, min_y, min_x = tumor_coords.min(axis=0)
    max_z, max_y, max_x = tumor_coords.max(axis=0)
    
    pad = 2
    min_z = max(0, min_z - pad); min_y = max(0, min_y - pad); min_x = max(0, min_x - pad)
    max_z = min(seg.shape[0], max_z + pad); max_y = min(seg.shape[1], max_y + pad); max_x = min(seg.shape[2], max_x + pad)
    
    slices = (slice(min_z, max_z), slice(min_y, max_y), slice(min_x, max_x))
    return images[:, slices[0], slices[1], slices[2]], seg[slices[0], slices[1], slices[2]]

def insert_tumor(host_images, host_seg, donor_tumor_imgs, donor_tumor_seg, affine, output_prefix):
    c, d, h, w = host_images.shape
    td, th, tw = donor_tumor_seg.shape
    
    brain_mask = (host_images[0] > 0.01)
    
    for _ in range(20):
        if d - td <= 0 or h - th <= 0 or w - tw <= 0: return False

        cz = random.randint(td//2, d - td//2)
        cy = random.randint(th//2, h - th//2)
        cx = random.randint(tw//2, w - tw//2)
        
        z_slice = slice(cz - td//2, cz - td//2 + td)
        y_slice = slice(cy - th//2, cy - th//2 + th)
        x_slice = slice(cx - tw//2, cx - tw//2 + tw)
        
        region_brain = brain_mask[z_slice, y_slice, x_slice]
        region_seg = host_seg[z_slice, y_slice, x_slice]
        
        if np.mean(region_brain) > 0.5 and np.sum(region_seg) == 0:
            tumor_mask = (donor_tumor_seg > 0)
            
            for i in range(4):
                host_slice = host_images[i, z_slice, y_slice, x_slice]
                tumor_slice = donor_tumor_imgs[i]
                host_slice[tumor_mask] = tumor_slice[tumor_mask]
                host_images[i, z_slice, y_slice, x_slice] = host_slice
            
            host_seg_slice = host_seg[z_slice, y_slice, x_slice]
            host_seg_slice[tumor_mask] = donor_tumor_seg[tumor_mask]
            host_seg[z_slice, y_slice, x_slice] = host_seg_slice
            
            for i in range(4):
                nib.save(nib.Nifti1Image(host_images[i], affine), f"{output_prefix}_{i:04d}.nii.gz")
            
            label_out = output_prefix.replace("images", "labels") + ".nii.gz"
            nib.save(nib.Nifti1Image(host_seg, affine), label_out)
            return True
    return False

def main():
    real_images_dir = "data/brats_real/images"
    real_labels_dir = "data/brats_real/labels"
    synth_base_dir = "data/synthetic"
    
    os.makedirs(os.path.join(synth_base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(synth_base_dir, "labels"), exist_ok=True)
    
    # 1. Identify COMPLETE cases only
    real_ids = get_complete_cases(real_images_dir, real_labels_dir)
    print(f"Found {len(real_ids)} COMPLETE real cases for Frankenstein generation.")
    
    if len(real_ids) < 2:
        print("Need at least 2 complete cases to mix! Aborting.")
        return

    # 2. Generate Synthetic Cases
    num_synthetic = 100
    print(f"Generating {num_synthetic} synthetic cases...")
    
    # Pre-load data to speed up? No, memory might be issue. Load on demand.
    
    count = 0
    pbar = tqdm(total=num_synthetic)
    
    while count < num_synthetic:
        synth_id = f"BraTS_Synth_{count+1:03d}"
        
        host_id = random.choice(real_ids)
        donor_id = random.choice(real_ids)
        
        try:
            host_prefix = os.path.join(real_images_dir, host_id)
            host_imgs, host_seg, affine = load_case(host_prefix)
            
            donor_prefix = os.path.join(real_images_dir, donor_id)
            donor_imgs, donor_seg, _ = load_case(donor_prefix)
            
            tumor_imgs, tumor_seg = extract_tumor(donor_imgs, donor_seg)
            
            if tumor_imgs is None:
                continue
                
            if random.random() > 0.5:
                tumor_imgs = np.flip(tumor_imgs, axis=2)
                tumor_seg = np.flip(tumor_seg, axis=2)
                
            out_prefix = os.path.join(synth_base_dir, "images", synth_id)
            success = insert_tumor(host_imgs, host_seg, tumor_imgs, tumor_seg, affine, out_prefix)
            
            if success:
                count += 1
                pbar.update(1)
                
        except Exception as e:
            print(f"Error processing {host_id}/{donor_id}: {e}")
            continue
            
    pbar.close()
    print("Frankenstein generation complete.")

if __name__ == "__main__":
    main()
