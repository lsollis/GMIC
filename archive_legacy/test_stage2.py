# Create test_stage2.py
import sys
import os
sys.path.insert(0, '/workspace')

from src.utilities import pickling
from src.utilities import data_handling
from src.optimal_centers.get_optimal_centers import get_optimal_centers

# Load your cropped exam list
cropped_exam_list_path = "/workspace/outputs/cropped_exam_list.pkl"
data_prefix = "/workspace/data/processed/cropped_images"

print("Loading cropped exam list...")
cropped_exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
print(f"Loaded {len(cropped_exam_list)} exams")

print("Unpacking to image list...")
data_list = data_handling.unpack_exam_into_images(cropped_exam_list, cropped=True)
print(f"Unpacked to {len(data_list)} images")

# Check first image structure
if data_list:
    sample = data_list[0]
    print(f"Sample image keys: {list(sample.keys())}")
    
    # Check required keys for center extraction
    required_keys = ["short_file_path", "full_view", "view", "horizontal_flip", "rightmost_points", "bottommost_points"]
    missing = [k for k in required_keys if k not in sample]
    if missing:
        print(f"PROBLEM: Missing required keys: {missing}")
    else:
        print("All required keys present")
        
        # Try center extraction on one image
        print("Testing center extraction...")
        try:
            centers = get_optimal_centers(
                data_list=data_list[:1],  # Just test first image
                data_prefix=data_prefix,
                num_processes=1
            )
            print(f"Success! Centers: {centers}")
        except Exception as e:
            print(f"Center extraction failed: {e}")
            import traceback
            traceback.print_exc()