# ============================================================================
# smart_data_loader.py - Final unified data loader
# ============================================================================

import os
import torch
import numpy as np
import pandas as pd
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# GMIC imports
from src.data_loading import loading
from src.utilities import pickling, data_handling
from src.constants import VIEWS


class GMICDataLoader:
    """
    Smart GMIC data loader that handles:
    1. CSV and PKL input formats
    2. Automatic preprocessing cache detection
    3. Train/val/test splits
    4. Backward compatibility with existing code
    """

    def __init__(self, data_path, image_path, batch_size=4, random_seed=42,
                 use_predefined_splits=True, val_split=0.2, test_split=0.1,
                 input_format="auto", enable_preprocessing=True,
                 output_dir="/workspace/processed_data", num_processes=4,
                 force_preprocessing=True, cache_validation=True):
        """
        Initialize smart GMIC data loader
        """
        self.data_path = data_path
        self.image_path = image_path
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.use_predefined_splits = use_predefined_splits
        self.val_split = val_split
        self.test_split = test_split
        self.input_format = input_format
        self.enable_preprocessing = enable_preprocessing
        self.output_dir = output_dir
        self.num_processes = num_processes
        self.force_preprocessing = force_preprocessing
        self.cache_validation = cache_validation

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # --- Robust format detection & guards ---
        if self.input_format == "auto":
            if str(self.data_path).endswith(".csv"):
                self.input_format = "csv"
            elif str(self.data_path).endswith(".pkl"):
                # processed PKL by default
                self.input_format = "pkl"
            else:
                raise ValueError(f"Cannot auto-detect format for {self.data_path}")

        # If the path is a PKL, never preprocess & always treat as PKL
        if str(self.data_path).endswith(".pkl"):
            if self.input_format != "pkl":
                print(f"⚠️ input_format='{self.input_format}' overridden to 'pkl' based on file extension.")
            self.input_format = "pkl"
            if self.enable_preprocessing:
                print("⚠️ enable_preprocessing=True with .pkl → disabling preprocessing.")
            self.enable_preprocessing = False
            # If the PKL is in our output tree, default images to cropped_images
            processed_dir = os.path.join(self.output_dir, "cropped_images")
            if os.path.isdir(processed_dir):
                self.image_path = processed_dir

        print(f"🔍 Detected format: {self.input_format}")
        print(f"📁 Data path: {self.data_path}")
        print(f"🖼️  Image path: {self.image_path}")
        print(f"⚙️  Preprocessing: {self.enable_preprocessing}")

        # Initialize based on format and preprocessing needs
        if self.enable_preprocessing:
            self._initialize_with_preprocessing()
        else:
            self._initialize_without_preprocessing()

    def _initialize_with_preprocessing(self):
        """Initialize with complete GMIC preprocessing pipeline"""
        print("🔄 Complete GMIC preprocessing pipeline enabled")

        # Expected cache artifacts
        cache_path = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cache_info_path = os.path.join(self.output_dir, "preprocessing_cache_info.json")
        cropped_dir = os.path.join(self.output_dir, "cropped_images")

        # 1) Try to use cache (unless force_preprocessing=True)
        if not self.force_preprocessing and os.path.exists(cache_path) and os.path.exists(cache_info_path):
            if self._validate_preprocessing_cache(cache_info_path):
                print("📋 Loading from preprocessing cache...")
                self.exam_list = pickling.unpickle_from_file(cache_path)

                # 🔒 Lock this loader instance to the processed artifacts
                self.data_path = cache_path
                if os.path.isdir(cropped_dir):
                    self.image_path = cropped_dir
                self.input_format = "pkl"
                print(f"[DATALOADER] Locked to processed cache: data_path={self.data_path}, image_path={self.image_path}")

                self._create_data_splits()
                return

        # 2) Build cache from source
        print("🚀 Starting complete preprocessing pipeline...")

        if self.input_format == "csv":
            # Start from CSV with raw image paths
            self.df = pd.read_csv(self.data_path)

            # Convert CSV to initial exam list format for cropping
            initial_exam_list = self._convert_csv_to_initial_format()

            # Stage 1: Crop mammograms
            cropped_exam_list = self._stage1_crop_mammograms(initial_exam_list)

            # Stage 2: Extract optimal centers
            final_exam_list = self._stage2_extract_centers(cropped_exam_list)

            self.exam_list = final_exam_list

        elif self.input_format == "raw_pkl":
            # Start from raw PKL (assumes initial exam list format)
            initial_exam_list = pickling.unpickle_from_file(self.data_path)

            # Stage 1: Crop mammograms
            cropped_exam_list = self._stage1_crop_mammograms(initial_exam_list)

            # Stage 2: Extract optimal centers
            final_exam_list = self._stage2_extract_centers(cropped_exam_list)

            self.exam_list = final_exam_list

        else:
            raise ValueError(f"Preprocessing not supported for format: {self.input_format}")

        # 3) Cache the processed data
        self._save_preprocessing_cache()

        # 🔒 Lock this loader instance to the processed artifacts
        self.data_path = cache_path
        if os.path.isdir(cropped_dir):
            self.image_path = cropped_dir
        self.input_format = "pkl"
        print(f"[DATALOADER] Locked to processed cache: data_path={self.data_path}, image_path={self.image_path}")

        # 4) Create splits
        self._create_data_splits()

        print("✅ Complete preprocessing pipeline finished!")

    def _initialize_without_preprocessing(self):
        """Initialize without preprocessing - use existing data"""
        print("📂 Using existing processed/ready data")

        if self.input_format == "csv":
            self.df = pd.read_csv(self.data_path)
            self.exam_list = self._convert_csv_to_gmic_format()
        elif self.input_format == "pkl":
            self.exam_list = pickling.unpickle_from_file(self.data_path)
            self.df = None
            # If this PKL lives in our processed tree, prefer the cropped images dir
            processed_dir = os.path.join(self.output_dir, "cropped_images")
            if os.path.isdir(processed_dir):
                self.image_path = processed_dir
                print(f"[DATALOADER] image_path set to processed dir: {self.image_path}")
        else:
            raise ValueError(f"Unknown input_format for no-preproc path: {self.input_format}")

        # Create splits
        self._create_data_splits()

    def _convert_csv_to_gmic_format(self):
        """Convert CSV format to GMIC exam list format"""
        exam_list = []

        grouped = self.df.groupby(['patient_id', 'exam_id'])
        for (patient_id, exam_id), group in grouped:
            exam = {
                'patient_id': patient_id,
                'exam_id': exam_id,
                'horizontal_flip': group.iloc[0].get('horizontal_flip', 'NO'),
                'cancer_label': {},
                'best_center': {},
                'file_paths': {}  # NEW: Store full file paths
            }

            # Initialize view lists
            for view in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                exam[view] = []
                exam['best_center'][view] = []
                exam['file_paths'][view] = []  # NEW: Store full paths per view

            # Set cancer labels
            exam_level_label = group.iloc[0]['exam_level_label']
            exam['cancer_label'] = {
                'benign': 1 if exam_level_label == 0 else 0,
                'malignant': 1 if exam_level_label == 1 else 0,
                'left_benign': 0, 'right_benign': 0,
                'left_malignant': 0, 'right_malignant': 0,
                'unknown': 0
            }

            # Process images
            for _, row in group.iterrows():
                laterality = row['laterality']
                view = row['view']
                full_view = f"{laterality}-{view}"

                image_id = f"{patient_id}_{exam_id}_{laterality}_{view}"
                exam[full_view].append(image_id)
                exam['best_center'][full_view].append((128, 128))  # Default center

                # NEW: Store the full file path from CSV
                exam['file_paths'][full_view].append(row['file_path'])

                # Update cancer labels
                view_label = row['view_level_label']
                if laterality == 'L':
                    if view_label == 1:
                        exam['cancer_label']['left_malignant'] = 1
                    else:
                        exam['cancer_label']['left_benign'] = 1
                else:
                    if view_label == 1:
                        exam['cancer_label']['right_malignant'] = 1
                    else:
                        exam['cancer_label']['right_benign'] = 1

            # Store split info
            if 'split_group' in group.columns:
                exam['split_group'] = group.iloc[0]['split_group']

            exam_list.append(exam)

        return exam_list

    def _create_data_splits(self):
        """Create train/validation/test splits"""

        # Convert exams to individual images
        all_data = self._unpack_exam_into_images(self.exam_list)

        # Use predefined splits if available
        if self.use_predefined_splits and self._has_predefined_splits():
            self._use_predefined_splits(all_data)
        else:
            self._create_automatic_splits(all_data)

    def _has_predefined_splits(self):
        """Check if predefined splits are available"""
        if self.input_format == "csv" and self.df is not None and 'split_group' in self.df.columns:
            return True
        elif any('split_group' in exam for exam in self.exam_list):
            return True
        return False

    def _use_predefined_splits(self, all_data):
        """Use predefined splits from data"""
        split_mapping = {'train': 'train', 'dev': 'val', 'val': 'val', 'test': 'test'}

        self.train_data = []
        self.val_data = []
        self.test_data = []

        for datum in all_data:
            # Determine split group
            if 'split_group' in datum:
                split_group = datum['split_group']
            elif self.input_format == "csv" and self.df is not None:
                split_group = 'train'  # Default
            else:
                split_group = 'train'

            mapped_split = split_mapping.get(split_group.lower(), 'train')

            if mapped_split == 'train':
                self.train_data.append(datum)
            elif mapped_split == 'val':
                self.val_data.append(datum)
            elif mapped_split == 'test':
                self.test_data.append(datum)

        print(f"✅ Using predefined splits:")
        print(f"  Train: {len(self.train_data)} images")
        print(f"  Val: {len(self.val_data)} images")
        print(f"  Test: {len(self.test_data)} images")

    def _create_automatic_splits(self, all_data):
        """Create automatic train/val/test splits"""

        # Get unique exam IDs for splitting
        if self.input_format == "csv":
            unique_exam_ids = list(set([f"{data['patient_id']}_{data['exam_id']}" for data in all_data]))
        else:
            unique_exam_ids = list(set([data['exam_id'] for data in all_data]))

        # Split by exam to avoid data leakage
        if self.test_split > 0:
            train_val_ids, test_ids = train_test_split(
                unique_exam_ids, test_size=self.test_split, random_state=self.random_seed
            )
        else:
            train_val_ids = unique_exam_ids
            test_ids = []

        if self.val_split > 0 and len(train_val_ids) > 1:
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=self.val_split / (1 - self.test_split),
                random_state=self.random_seed + 1
            )
        else:
            train_ids = train_val_ids
            val_ids = []

        # Create data lists
        if self.input_format == "csv":
            self.train_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in train_ids]
            self.val_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in val_ids]
            self.test_data = [d for d in all_data if f"{d['patient_id']}_{d['exam_id']}" in test_ids]
        else:
            self.train_data = [d for d in all_data if d['exam_id'] in train_ids]
            self.val_data = [d for d in all_data if d['exam_id'] in val_ids]
            self.test_data = [d for d in all_data if d['exam_id'] in test_ids]

        print(f"✅ Created automatic splits:")
        print(f"  Train: {len(self.train_data)} images from {len(train_ids)} exams")
        print(f"  Val: {len(self.val_data)} images from {len(val_ids)} exams")
        print(f"  Test: {len(self.test_data)} images from {len(test_ids)} exams")

    def _unpack_exam_into_images(self, exam_list):
        """Convert exam list to individual image entries"""
        data_list = []

        for exam_idx, exam in enumerate(exam_list):
            for view in VIEWS.LIST:
                if view in exam and exam[view]:
                    for img_idx, image_identifier in enumerate(exam[view]):

                        datum = {
                            'exam_id': exam_idx,
                            'image_id': image_identifier,
                            'short_file_path': image_identifier,
                            'view': view,
                            'horizontal_flip': exam['horizontal_flip'],
                            'best_center': exam['best_center'][view][img_idx] if 'best_center' in exam and view in exam['best_center'] else None,
                            'cancer_label': exam['cancer_label'],
                        }

                        # NEW: Add the full file path from CSV
                        if 'file_paths' in exam and view in exam['file_paths'] and img_idx < len(exam['file_paths'][view]):
                            datum['full_file_path'] = exam['file_paths'][view][img_idx]
                        else:
                            datum['full_file_path'] = None

                        # Extract labels
                        if 'cancer_label' in exam:
                            datum['exam_level_label'] = exam['cancer_label'].get('malignant', 0)
                            datum['view_level_label'] = self._extract_view_level_label(exam['cancer_label'], view)
                        else:
                            datum['exam_level_label'] = 0
                            datum['view_level_label'] = 0

                        # Add CSV-specific info if available
                        if 'patient_id' in exam:
                            datum['patient_id'] = exam['patient_id']

                        # Add split group if available
                        if 'split_group' in exam:
                            datum['split_group'] = exam['split_group']

                        data_list.append(datum)

        return data_list

    def _extract_view_level_label(self, cancer_label, view):
        side_code = view.split('-')[0].upper()          # "L" or "R"
        side = 'left' if side_code == 'L' else 'right'  # map to keys you actually use

        if cancer_label.get(f'{side}_malignant', 0) == 1:
            return 1
        if cancer_label.get(f'{side}_benign', 0) == 1:
            return 0
        return 0

    def _validate_preprocessing_cache(self, cache_info_path):
        """Validate preprocessing cache is still valid"""
        if not self.cache_validation:
            return True

        try:
            with open(cache_info_path, 'r') as f:
                cache_info = json.load(f)

            # Check if input data has changed
            current_hash = self._get_input_data_hash()
            if cache_info.get('input_hash') != current_hash:
                print("⚠️  Input data changed, cache invalid")
                return False

            print("✅ Preprocessing cache is valid")
            return True

        except Exception as e:
            print(f"⚠️  Cache validation failed: {e}")
            return False

    def _get_input_data_hash(self):
        """Get hash of input data for cache validation"""
        hasher = hashlib.md5()
        with open(self.data_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _save_preprocessing_cache(self):
        """Save preprocessing cache with metadata"""
        cache_path = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cache_info_path = os.path.join(self.output_dir, "preprocessing_cache_info.json")

        # Save processed exam list
        pickling.pickle_to_file(cache_path, self.exam_list)

        # Save cache metadata
        cache_info = {
            'input_hash': self._get_input_data_hash(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_exams': len(self.exam_list)
        }

        with open(cache_info_path, 'w') as f:
            json.dump(cache_info, f, indent=2)

        print(f"💾 Cached processed data: {cache_path}")

    def _convert_csv_to_initial_format(self):
        """Convert CSV to initial exam list format suitable for cropping pipeline"""
        exam_list = []

        # Group by patient and exam
        grouped = self.df.groupby(['patient_id', 'exam_id'])

        for (patient_id, exam_id), group in grouped:
            exam = {
                'patient_id': patient_id,
                'exam_id': exam_id,
                'horizontal_flip': group.iloc[0].get('horizontal_flip', 'NO'),
                'cancer_label': {},
                'original_file_paths': {}  # Store original file paths from CSV
            }

            # Initialize view lists
            for view in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                exam[view] = []
                exam['original_file_paths'][view] = []

            # Set cancer labels
            exam_level_label = group.iloc[0]['exam_level_label']
            exam['cancer_label'] = {
                'benign': 1 if exam_level_label == 0 else 0,
                'malignant': 1 if exam_level_label == 1 else 0,
                'left_benign': 0, 'right_benign': 0,
                'left_malignant': 0, 'right_malignant': 0,
                'unknown': 0
            }

            # Process each image
            for _, row in group.iterrows():
                laterality = row['laterality']
                view = row['view']
                full_view = f"{laterality}-{view}"

                # Create image identifier
                image_id = f"{patient_id}_{exam_id}_{laterality}_{view}"
                exam[full_view].append(image_id)

                # Store original file path from CSV for copying
                exam['original_file_paths'][full_view].append(row['file_path'])

                # Update view-level cancer labels
                view_label = row['view_level_label']
                if laterality == 'L':
                    if view_label == 1:
                        exam['cancer_label']['left_malignant'] = 1
                    else:
                        exam['cancer_label']['left_benign'] = 1
                else:
                    if view_label == 1:
                        exam['cancer_label']['right_malignant'] = 1
                    else:
                        exam['cancer_label']['right_benign'] = 1

            # Add split info if available
            if 'split_group' in group.columns:
                exam['split_group'] = group.iloc[0]['split_group']

            exam_list.append(exam)

        #exam_list = exam_list[:50]  # Limit to first 50 exams for initial testing
        print(f"Converted CSV to initial format: {len(exam_list)} exams")
        return exam_list

    def _assert_cropped_schema(self, exam_list):
        """Minimum fields needed by center extraction; do NOT require 'window_location'."""
        required_keys = [
            "short_file_path", "full_view", "view", "horizontal_flip",
            "rightmost_points", "bottommost_points"
        ]
        data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
        if not data_list:
            raise RuntimeError("Cropped exam list unpacked to 0 images.")
        missing = [k for k in required_keys if k not in data_list[0]]
        if missing:
            raise RuntimeError(f"Cropped exam list missing keys: {missing}. "
                               f"Did the crop step actually run, or did you reuse an initial/stale PKL?")

    def _debug_cropped_schema(self, exam_list):
        """Debug what schema we actually have vs what we expect"""
        print("🔍 Debugging cropped exam list schema...")

        if not exam_list:
            print("❌ Exam list is empty!")
            return

        # Check exam-level structure
        sample_exam = exam_list[0]
        print(f"📋 Sample exam keys: {list(sample_exam.keys())}")

        # Unpack to image level
        try:
            data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
            if not data_list:
                print("❌ No images found after unpacking!")
                return

            sample_image = data_list[0]
            print(f"🖼️  Sample image keys: {list(sample_image.keys())}")

            required_keys = [
                "short_file_path", "full_view", "view", "horizontal_flip",
                "rightmost_points", "bottommost_points"
            ]

            missing_keys = [k for k in required_keys if k not in sample_image]
            present_keys = [k for k in required_keys if k in sample_image]

            print(f"✅ Present keys: {present_keys}")
            print(f"❌ Missing keys: {missing_keys}")

            for key in present_keys[:3]:
                print(f"   {key}: {sample_image[key]}")

        except Exception as e:
            print(f"❌ Error during unpacking: {e}")
            import traceback
            traceback.print_exc()

    def _prepare_staging_area(self, initial_exam_list, staging_dir):
        """Prepare staging area by copying files with expected naming"""
        import shutil

        for exam in initial_exam_list:
            for view in ['L-CC', 'L-MLO', 'R-CC', 'R-MLO']:
                if view in exam and exam[view] and 'original_file_paths' in exam:
                    for i, (image_id, original_path) in enumerate(zip(exam[view], exam['original_file_paths'][view])):
                        expected_filename = f"{image_id}.png"
                        staging_path = os.path.join(staging_dir, expected_filename)
                        if os.path.exists(original_path):
                            shutil.copy2(original_path, staging_path)
                            print(f"Staged: {original_path} -> {staging_path}")

    def _stage1_crop_mammograms(self, initial_exam_list):
        print("Stage 1: Cropping mammograms...")

        import shutil

        staging_dir = os.path.join(self.output_dir, "staging")
        output_data_folder = os.path.join(self.output_dir, "cropped_images")
        initial_exam_list_path = os.path.join(self.output_dir, "initial_exam_list.pkl")
        cropped_exam_list_path = os.path.join(self.output_dir, "cropped_exam_list.pkl")

        # (Re)create folders
        os.makedirs(staging_dir, exist_ok=True)
        if os.path.exists(output_data_folder):
            if self.force_preprocessing:
                print("🔄 force_preprocessing=True: removing existing cropped_images folder...")
                shutil.rmtree(output_data_folder)
                os.makedirs(output_data_folder, exist_ok=True)
            else:
                print("📁 Using existing cropped_images folder. Set force_preprocessing=True to regenerate.")
                if os.path.exists(cropped_exam_list_path):
                    print("📋 Loading existing cropped exam list...")
                    try:
                        cropped_exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
                        self._assert_cropped_schema(cropped_exam_list)
                        print(f"✅ Using cached cropped data: {len(cropped_exam_list)} exams")
                        return cropped_exam_list
                    except Exception as e:
                        print(f"⚠️  Cached cropped data invalid: {e}")
                        print("🔄 Will regenerate...")
                        shutil.rmtree(output_data_folder)
                        os.makedirs(output_data_folder, exist_ok=True)
                else:
                    print("🔄 No cached exam list found, will regenerate...")
                    shutil.rmtree(output_data_folder)
                    os.makedirs(output_data_folder, exist_ok=True)
        else:
            os.makedirs(output_data_folder, exist_ok=True)

        # Stage files and write initial list
        self._prepare_staging_area(initial_exam_list, staging_dir)
        pickling.pickle_to_file(initial_exam_list_path, initial_exam_list)

        try:
            from src.cropping.crop_mammogram import crop_mammogram

            print(f"  Input folder: {staging_dir}")
            print(f"  Output folder: {output_data_folder}")
            print(f"  Processing {len(initial_exam_list)} exams with {self.num_processes} processes")

            crop_mammogram(
                input_data_folder=staging_dir,
                exam_list_path=initial_exam_list_path,
                cropped_exam_list_path=cropped_exam_list_path,
                output_data_folder=output_data_folder,
                num_processes=self.num_processes,
                num_iterations=100,
                buffer_size=50,
            )

            cropped_exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
            self._assert_cropped_schema(cropped_exam_list)
            print(f"Stage 1 complete: {len(cropped_exam_list)} exams cropped")
            return cropped_exam_list

        except Exception as e:
            print(f"Cropping failed: {e}")
            print("   Using original exam list (no centers will be possible).")
            return initial_exam_list

    def _stage2_extract_centers(self, cropped_exam_list):
        """Stage 2: Extract optimal centers using GMIC's helpers."""
        print("🎯 Stage 2: Extracting optimal centers...")

        # Paths for traceability/debug
        cropped_exam_list_path = os.path.join(self.output_dir, "cropped_exam_list.pkl")
        data_prefix = os.path.join(self.output_dir, "cropped_images")
        output_exam_list_path = os.path.join(self.output_dir, "final_exam_list.pkl")

        # Save the cropped list we just got from Stage 1
        pickling.pickle_to_file(cropped_exam_list_path, cropped_exam_list)

        try:
            data_list = data_handling.unpack_exam_into_images(cropped_exam_list, cropped=True)

            if data_list:
                sample = data_list[0]
                for k in ["short_file_path", "full_view", "view", "horizontal_flip"]:
                    assert k in sample, f"Missing key {k} in data_list item"

            print(f"  Cropped data: {data_prefix}")
            print(f"  Processing {len(cropped_exam_list)} exams, {len(data_list)} images, with {self.num_processes} processes")

            from src.optimal_centers.get_optimal_centers import get_optimal_centers

            centers = get_optimal_centers(
                data_list=data_list,
                data_prefix=data_prefix,
                num_processes=int(self.num_processes),
            )

            data_handling.add_metadata(cropped_exam_list, "best_center", centers)

            pickling.pickle_to_file(output_exam_list_path, cropped_exam_list)
            print(f"✅ Stage 2 complete: {len(cropped_exam_list)} exams; centers added for {len(centers)} images.")
            return cropped_exam_list

        except Exception as e:
            print(f"Center extraction failed: {e}")
            print("   Using cropped exam list without centers...")
            return cropped_exam_list

    # ---------------------------
    # Batching / Image IO
    # ---------------------------

    def _resolve_image_path(self, datum):
        """Return an existing file path for the datum image or None if not found."""
        candidates = []

        # 1) Explicit full path (CSV mode)
        p = datum.get("full_file_path")
        if p:
            candidates.append(p)

        # 2) Current image_path root (raw or processed)
        if self.image_path:
            candidates.append(os.path.join(self.image_path, f"{datum['short_file_path']}.png"))

        # 3) Processed default (<output_dir>/cropped_images)
        candidates.append(os.path.join(self.output_dir, "cropped_images", f"{datum['short_file_path']}.png"))

        # 4) Staging fallback (debug)
        candidates.append(os.path.join(self.output_dir, "staging", f"{datum['short_file_path']}.png"))

        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def get_data_for_split(self, split='train'):
        """Get data for specific split"""
        if split == 'train':
            return self.train_data
        elif split == 'val' or split == 'validation':
            return self.val_data
        elif split == 'test':
            return self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")

    def get_batch_iterator(self, split='train', shuffle=True):
        """Get batch iterator for specified split"""
        data_list = self.get_data_for_split(split)

        if shuffle and split == 'train':
            np.random.shuffle(data_list)

        for i in range(0, len(data_list), self.batch_size):
            batch_data = data_list[i:i + self.batch_size]
            batch_images = []
            batch_labels = []
            batch_metadata = []

            for datum in batch_data:
                try:
                    image_path = self._resolve_image_path(datum)
                    if not image_path:
                        print(f"❌ Image not found for {datum['short_file_path']} (view={datum['view']}). "
                              f"Tried roots: [{self.image_path}], "
                              f"{os.path.join(self.output_dir, 'cropped_images')}, "
                              f"{os.path.join(self.output_dir, 'staging')}")
                        continue

                    # Load and process image
                    loaded_image = loading.load_image(
                        image_path=image_path,
                        view=datum['view'],
                        horizontal_flip=datum['horizontal_flip']
                    )

                    # Process with best center if available
                    if datum['best_center'] is not None:
                        processed_image = loading.process_image(
                            image=loaded_image,
                            view=datum['view'],
                            best_center=datum['best_center']
                        )
                    else:
                        processed_image = loaded_image.copy()
                        loading.standard_normalize_single_image(processed_image)

                    # Convert to tensor format
                    processed_image = np.expand_dims(np.expand_dims(processed_image, 0), 0)
                    batch_images.append(processed_image)

                    # Create labels
                    view_level_label = datum['view_level_label']
                    if view_level_label == 1:
                        benign_label = 0
                        malignant_label = 1
                    else:
                        benign_label = 1
                        malignant_label = 0

                    batch_labels.append([benign_label, malignant_label])

                    # Metadata
                    metadata = {
                        'image_id': datum['image_id'],
                        'exam_id': datum['exam_id'],
                        'view': datum['view'],
                        'split': split,
                        'exam_level_label': datum['exam_level_label'],
                        'view_level_label': datum['view_level_label'],
                        'image_path': image_path
                    }

                    if 'patient_id' in datum:
                        metadata['patient_id'] = datum['patient_id']

                    batch_metadata.append(metadata)

                except Exception as e:
                    print(f"❌ Error processing {datum.get('image_id', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # Yield batch
            if batch_images:
                batch_tensor = torch.FloatTensor(np.concatenate(batch_images, axis=0))
                label_tensor = torch.FloatTensor(batch_labels)
                yield batch_tensor, label_tensor, batch_metadata

    def get_split_info(self):
        """Get information about data splits"""
        info = {
            'train_size': len(self.train_data),
            'val_size': len(self.val_data),
            'test_size': len(self.test_data),
            'total_size': len(self.train_data) + len(self.val_data) + len(self.test_data),
            'input_format': self.input_format,
            'preprocessing_enabled': self.enable_preprocessing
        }

        if self.input_format == "csv" and hasattr(self, 'train_data'):
            train_patients = set([d.get('patient_id') for d in self.train_data if 'patient_id' in d])
            val_patients = set([d.get('patient_id') for d in self.val_data if 'patient_id' in d])
            test_patients = set([d.get('patient_id') for d in self.test_data if 'patient_id' in d])

            info.update({
                'train_patients': len(train_patients),
                'val_patients': len(val_patients),
                'test_patients': len(test_patients),
                'total_patients': len(train_patients | val_patients | test_patients)
            })

        return info

    def get_class_distribution(self, split='train'):
        """Get class distribution for a split"""
        data = self.get_data_for_split(split)

        exam_level_counts = defaultdict(int)
        view_level_counts = defaultdict(int)

        for datum in data:
            exam_level_counts[datum['exam_level_label']] += 1
            view_level_counts[datum['view_level_label']] += 1

        return {
            'exam_level': dict(exam_level_counts),
            'view_level': dict(view_level_counts),
            'total': len(data)
        }

    def save_pkl_format(self, output_path):
        """Save exam list in PKL format"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True
        )
        pickling.pickle_to_file(output_path, self.exam_list)
        print(f"Saved exam list to: {output_path}")

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 60)
        print("GMIC DATA LOADER SUMMARY")
        print("=" * 60)

        split_info = self.get_split_info()
        print(f"Input format: {self.input_format.upper()}")
        print(f"Preprocessing enabled: {self.enable_preprocessing}")
        print(f"Total images: {split_info['total_size']}")
        print(f"Total exams: {len(self.exam_list)}")

        if 'total_patients' in split_info:
            print(f"Total patients: {split_info['total_patients']}")

        print(f"\nData Splits:")
        print(f"  Train: {split_info['train_size']} images")
        if 'train_patients' in split_info:
            print(f"    from {split_info['train_patients']} patients")
        print(f"  Val:   {split_info['val_size']} images")
        if 'val_patients' in split_info:
            print(f"    from {split_info['val_patients']} patients")
        print(f"  Test:  {split_info['test_size']} images")
        if 'test_patients' in split_info:
            print(f"    from {split_info['test_patients']} patients")

        # Class distributions
        for split in ['train', 'val', 'test']:
            if len(self.get_data_for_split(split)) > 0:
                dist = self.get_class_distribution(split)
                print(f"\n{split.capitalize()} Class Distribution:")
                print(f"  Exam level - Normal: {dist['exam_level'].get(0, 0)}, Cancer: {dist['exam_level'].get(1, 0)}")
                print(f"  View level - Normal: {dist['view_level'].get(0, 0)}, Cancer: {dist['view_level'].get(1, 0)}")

    def __len__(self):
        """Return size of training split"""
        return len(self.train_data)