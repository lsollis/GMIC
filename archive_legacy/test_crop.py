# Create corrected_test_crop.py
import sys
import os
sys.path.insert(0, '/workspace')

from src.cropping.crop_mammogram import crop_mammogram_one_image_short_path
import pandas as pd

# Load your CSV
df = pd.read_csv('/workspace/data/gmic_format_xai.csv')
print(f"CSV has {len(df)} rows")

# Test cropping with correct scan object structure
for i, row in df.head(3).iterrows():
    file_path = row['file_path']
    
    if not os.path.exists(file_path):
        print(f"MISSING: {file_path}")
        continue
        
    print(f"\nTesting: {file_path}")
    
    # Create proper scan object that the cropping function expects
    image_id = f"{row['patient_id']}_{row['exam_id']}_{row['laterality']}_{row['view']}"
    scan = {
        'short_file_path': image_id,
        'horizontal_flip': row.get('horizontal_flip', 'NO'),
        'side': row['laterality'],  # L or R
        'full_view': f"{row['laterality']}-{row['view']}",  # L-CC, R-MLO, etc.
        'view': row['view']  # CC or MLO
    }
    
    print(f"Scan object: {scan}")
    
    try:
        # Create staging directory and copy the file there with expected name
        staging_dir = "/tmp/test_staging"
        output_dir = "/tmp/test_output"
        os.makedirs(staging_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the image to staging with expected filename
        import shutil
        staged_path = os.path.join(staging_dir, f"{image_id}.png")
        shutil.copy2(file_path, staged_path)
        print(f"Staged: {file_path} -> {staged_path}")
        
        # Now try cropping
        result = crop_mammogram_one_image_short_path(
            scan=scan,  # Pass the scan dict, not just image_id
            input_data_folder=staging_dir,
            output_data_folder=output_dir,
            num_iterations=100,
            buffer_size=50
        )
        print(f"SUCCESS! Result: {result}")
        
        # Check if output file was created
        output_path = os.path.join(output_dir, f"{image_id}.png")
        if os.path.exists(output_path):
            print(f"Cropped image saved: {output_path}")
        break
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()