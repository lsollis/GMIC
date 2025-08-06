# ============================================================================
# local_train_gmic.py - Standalone training script
# ============================================================================

#!/usr/bin/env python3
"""
Standalone GMIC training script using existing model components.
For local development and testing before federated deployment.
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add current directory to path to import bc_executor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bc_executor import GMICDataLoader
from src.modeling import gmic
from src.constants import PERCENT_T_DICT

def define_parser():
    parser = argparse.ArgumentParser(description='Train GMIC model locally using existing components')
    parser.add_argument("--exam_list_path", type=str, default="/workspace/data/bc_data/data.pkl")
    parser.add_argument("--image_path", type=str, default="/workspace/data/bc_data/cropped_images")
    parser.add_argument("--segmentation_path", type=str, default="/workspace/data/bc_data/segmentation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_path", type=str, default="/workspace/data/gmic_model.pth")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate existing model")
    return parser.parse_args()

def main():
    args = define_parser()
    
    print(f"Using device: {args.device}")
    print(f"GMIC dataset path: {args.exam_list_path}")
    
    # Set up GMIC model using your existing implementation
    gmic_parameters = {
        "device_type": "gpu" if "cuda" in args.device else "cpu",
        "gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "percent_t": PERCENT_T_DICT["1"]
    }
    
    # Create GMIC model
    model = gmic.GMIC(gmic_parameters)
    model.to(args.device)
    
    # Create data loader using existing components
    data_loader = GMICDataLoader(
        exam_list_path=args.exam_list_path,
        image_path=args.image_path,
        segmentation_path=args.segmentation_path,
        batch_size=args.batch_size
    )
    
    if args.evaluate_only:
        print("Loading existing GMIC model for evaluation...")
        model.load_state_dict(torch.load(args.model_path))
        
        # Evaluate model
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader.get_batch_iterator():
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Evaluation completed. Average loss: {avg_loss:.4f}")
        
    else:
        # Training mode
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        print(f"Starting GMIC training for {args.epochs} epochs...")
        print(f"Total samples: {len(data_loader)}")
        
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(data_loader.get_batch_iterator()):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass through GMIC
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save trained model
        torch.save(model.state_dict(), args.model_path)
        print(f"GMIC model saved to: {args.model_path}")
        
        print("GMIC training completed!")

if __name__ == "__main__":
    main()