# ============================================================================
# local_train_gmic.py - Standalone training script with flexible data loader
# ============================================================================

#!/usr/bin/env python3
"""
Standalone GMIC training script using flexible data loader for CSV/PKL input.
For local development and testing before federated deployment.
"""

import argparse
import sys
import os
import copy
import random
import uuid
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import GMICDataLoader  # Use the flexible data loader
from src.modeling import gmic
from src.constants import PERCENT_T_DICT

def define_parser():
    parser = argparse.ArgumentParser(description='Train GMIC model locally using flexible data loader')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- data / IO ---
    parser.add_argument("--data_path", type=str, default="/workspace/data/gmic_format_xai.csv")
    parser.add_argument("--image_path", type=str, default="/workspace/data/processed/cropped_images")
    parser.add_argument("--input_format", type=str, default="auto", choices=["auto", "csv", "pkl"])
    parser.add_argument("--device_type", type=str, default="gpu")
    parser.add_argument("--device", type=str, default=device)
    parser.add_argument("--gpu_number", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="/workspace/outputs")
    parser.add_argument("--model_path", type=str, default="/workspace/models/gmic.pth")

    # --- dataset split ---
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--use_predefined_splits", action="store_true")
    parser.add_argument("--seed", type=int, default=42)            

    # --- train / eval mode ---
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--test_mode", action="store_true")

    # --- optimization ---
    parser.add_argument("--lr_heads", type=float, default=1e-4)    
    parser.add_argument("--lr_backbone", type=float, default=1e-5) 
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=4)         

    # --- model hyperparams (GMIC) ---
    parser.add_argument("--percent_t", type=str, default="1")
    parser.add_argument("--K", type=int, default=6)                  
    parser.add_argument("--cam_h", type=int, default=46)
    parser.add_argument("--cam_w", type=int, default=30)
    parser.add_argument("--post_dim", type=int, default=256)        
    parser.add_argument("--num_classes", type=int, default=2)       
    parser.add_argument("--use_v1_global", action="store_true")     
    parser.add_argument("--lambda_l1", type=float, default=1e-5)    

    # NEW: crop shape (used by RetrieveROIModule / _retrieve_crop)
    parser.add_argument("--crop_h", type=int, default=256)
    parser.add_argument("--crop_w", type=int, default=256)

    # --- freezing options ---
    parser.add_argument("--freeze_all_backbones", action="store_true")                             # NEW
    parser.add_argument("--unfreeze_global_last", action="store_true")                             # NEW
    parser.add_argument("--unfreeze_local_last", action="store_true")                              # NEW

    # --- pretrained / checkpoints ---
    parser.add_argument("--pretrained_model_index", type=str, default="ensemble",
                        help="one of {1,2,3,4,5,ensemble} if supported by your GMIC build")
    parser.add_argument("--load_checkpoint", type=str, default="",  
                        help="path to .pth to load weights instead of model_index")

    # --- random search ---
    parser.add_argument("--search", action="store_true")            
    parser.add_argument("--search_trials", type=int, default=20)    
    parser.add_argument("--search_max_epochs", type=int, default=25)
    return parser.parse_args()

def build_gmic_from_args(args):
    """
    Construct GMIC strictly from CLI args; no hard-coded defaults here.
    Anything not passed will fall back to GMIC's internal defaults.
    """
    params = {}

    # Device hint (derived from CLI)
    params["device_type"] = args.device_type
    params["gpu_number"] = args.gpu_number

    # IO / model-related knobs from CLI
    params["image_path"] = args.image_path
    params["num_classes"] = args.num_classes
    params["use_v1_global"] = bool(args.use_v1_global)

    # Optional structured params only if provided
    if hasattr(args, "cam_h") and hasattr(args, "cam_w"):
        params["cam_size"] = (args.cam_h, args.cam_w)

    if hasattr(args, "K"):
        params["K"] = args.K

    if hasattr(args, "post_dim"):
        params["post_processing_dim"] = args.post_dim

    if hasattr(args, "percent_t"):
        # Validate key and map via constants—still no literal values here
        if args.percent_t not in PERCENT_T_DICT:
            raise ValueError(
                f"percent_t '{args.percent_t}' not in PERCENT_T_DICT keys {list(PERCENT_T_DICT.keys())}"
            )
        params["percent_t"] = PERCENT_T_DICT[args.percent_t]

    if hasattr(args, "lambda_l1"):
        params["lambda_l1"] = args.lambda_l1

    if hasattr(args, "crop_h") and hasattr(args, "crop_w"):
        params["crop_shape"] = (args.crop_h, args.crop_w)

    # Construct with ONLY what came from CLI
    return gmic.GMIC(params)

def load_state_dict_forgiving(model, ckpt_path, device="cpu"):
    sd = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
        print("Loaded checkpoint with strict=True")
    except RuntimeError:
        print("Strict load failed; retrying with strict=False")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print("[load] Missing keys:", missing)
        if unexpected:
            print("[load] Unexpected keys:", unexpected)

def load_pretrained_if_requested(model, args):
    if args.load_checkpoint and os.path.isfile(args.load_checkpoint):
        print(f"Loading checkpoint: {args.load_checkpoint}")
        load_state_dict_forgiving(model, args.load_checkpoint, device=args.device)
        return

    if hasattr(model, "load_pretrained_by_index"):
        print(f"Loading pretrained model by index: {args.pretrained_model_index}")
        # This call is model-dependent; keep as-is if present
        model.load_pretrained_by_index(args.pretrained_model_index)

def configure_optimizers(model, args):
    # Identify parameter groups. Adjust these names to match your GMIC class.
    global_backbone = getattr(model, "global_backbone", None)
    local_backbone  = getattr(model, "local_backbone", None)
    heads = []

    for name in ["global_1x1_and_post", "mil_attention_block", "classifier_heads"]:
        m = getattr(model, name, None)
        if m is not None:
            heads.append({"params": [p for p in m.parameters() if p.requires_grad], "lr": args.lr_heads})

    backbone_params = []
    if global_backbone is not None:
        backbone_params += [p for p in global_backbone.parameters() if p.requires_grad]
    if local_backbone is not None:
        backbone_params += [p for p in local_backbone.parameters() if p.requires_grad]

    param_groups = []

    # Add backbone params if they exist
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.lr_backbone})

    # Add head params if they exist
    if heads:
        param_groups += heads

    # 🔒 Fallback: if nothing was added above, just use *all* model params
    if not param_groups:
        param_groups = [{"params": model.parameters(), "lr": args.lr_heads}]

    optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.2, patience=2, verbose=True
    )
    early_stopper = EarlyStopper(patience=args.patience)
    return optimizer, scheduler, early_stopper

def apply_freezing_plan(model, args):
    # Example patterns—adapt names to your model
    if getattr(args, "freeze_all_backbones", False):
        # If model.backbones might be a dict, this works:
        backbones = getattr(model, "backbones", None) or getattr(model, "gb", None)
        if backbones is not None:
            set_requires_grad(backbones, False)
        else:
            # fallback: name-based freeze
            for name, p in model.named_parameters():
                if "backbone" in name or "encoder" in name:
                    p.requires_grad = False

    if getattr(args, "freeze_transformer", False):
        if hasattr(model, "transformer"):
            set_requires_grad(model.transformer, False)
        else:
            for name, p in model.named_parameters():
                if "transformer" in name:
                    p.requires_grad = False

def evaluate_model(model, data_loader, criterion, device, split='val'):
    """Evaluate model on specified split"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets, metadata in data_loader.get_batch_iterator(split, shuffle=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            lg, tg = align_logits_targets(logits, targets)
            loss = criterion(lg, tg)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            if probs.ndim == 2 and probs.shape[1] > 1:
                all_predictions.extend(probs[:, 1])
                all_targets.extend(targets[:, 1].cpu().numpy())
            else:
                all_predictions.extend(probs.reshape(-1))
                all_targets.extend(targets.cpu().numpy().reshape(-1))
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
    predicted_labels = (all_predictions > 0.5).astype(int)
    accuracy = 100 * accuracy_score(all_targets, predicted_labels)
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        "auc": auc,
        "accuracy": accuracy,
        "loss": avg_loss,
        "total_samples": len(all_targets)
    }

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_requires_grad(obj, requires_grad: bool):
    # If it's a dict of modules/params, recurse on values
    if isinstance(obj, dict):
        for v in obj.values():
            set_requires_grad(v, requires_grad)
        return

    # If it's a list/tuple of modules/params, recurse
    if isinstance(obj, (list, tuple)):
        for v in obj:
            set_requires_grad(v, requires_grad)
        return

    # If it's a torch.nn.Module, toggle all its parameters
    if hasattr(obj, "parameters") and callable(getattr(obj, "parameters")):
        for p in obj.parameters():
            p.requires_grad = requires_grad
        return

    # If it's already an iterable of parameters
    try:
        for p in obj:
            p.requires_grad = requires_grad
        return
    except TypeError:
        raise TypeError(
            f"Unsupported type for set_requires_grad: {type(obj)} "
            "Expected nn.Module, dict/list/tuple of modules or params, or iterable of params."
        )

def align_logits_targets(logits, targets):
    # ensure float
    targets = targets.float()
    # If logits are (N,1) but targets are (N,2), take positive column
    if logits.dim() == 2 and logits.size(1) == 1 and targets.dim() == 2 and targets.size(1) == 2:
        targets = targets[:, 1:2]  # keep 2D shape (N,1)
    return logits, targets

class EarlyStopper:
    def __init__(self, patience=4):
        self.best = -float("inf")
        self.count = 0
        self.patience = patience
        self.best_state = None

    def step(self, metric, model):
        if metric > self.best:
            self.best = metric
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.count += 1
            return False

    def should_stop(self):
        return self.count >= self.patience
    
def _train_single_run(base_args) -> dict:
    """
    Runs a short training using the existing pipeline and returns
    {"val_auc": float, "best_path": str, "trial_args": Namespace}
    """
    # Build model & data like in main(), but using base_args
    model = build_gmic_from_args(base_args)
    model.to(base_args.device)
    load_pretrained_if_requested(model, base_args)
    apply_freezing_plan(model, base_args)

    data_loader = GMICDataLoader(
        data_path=base_args.data_path,
        image_path=base_args.image_path,
        input_format=base_args.input_format,
        batch_size=base_args.batch_size,
        use_predefined_splits=base_args.use_predefined_splits,
        val_split=base_args.val_split,
        test_split=base_args.test_split
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler, early_stopper = configure_optimizers(model, base_args)

    # short training loop (uses search_max_epochs)
    best_val_auc = -float("inf")
    model.train()
    for epoch in range(base_args.search_max_epochs):
        total_loss, num_batches = 0.0, 0
        all_predictions, all_targets = [], []

        for batch_idx, (inputs, targets, meta) in enumerate(data_loader.get_batch_iterator('train')):
            inputs = inputs.to(base_args.device)
            targets = targets.to(base_args.device)

            optimizer.zero_grad()
            logits = model(inputs)
            logits, targets_aligned = align_logits_targets(logits, targets)
            loss = criterion(logits, targets_aligned)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            if probs.ndim == 2 and probs.shape[1] > 1:
                pos_probs = probs[:, 1]
                pos_targets = targets[:, 1].detach().cpu().numpy()
            else:
                pos_probs = probs.reshape(-1)
                pos_targets = targets.detach().cpu().numpy().reshape(-1)

            all_predictions.extend(pos_probs)
            all_targets.extend(pos_targets)

        # end epoch metrics (optional)
        # train_auc = roc_auc_score(...) if you want to log

        # validation
        if len(data_loader.get_data_for_split('val')) > 0:
            val_metrics = evaluate_model(model, data_loader, criterion, base_args.device, split='val')
            val_auc = val_metrics['auc']
            scheduler.step(val_auc)

            improved = early_stopper.step(val_auc, model)
            if improved:
                best_val_auc = val_auc

            if early_stopper.should_stop():
                break

            model.train()

    # at the end of this run, save best and return metrics
    uid = uuid.uuid4().hex[:8]
    best_path = os.path.join(base_args.out_dir, f"gmic_best_search_{uid}.pth")
    if early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)
        torch.save(model.state_dict(), best_path)

    return {"val_auc": best_val_auc, "best_path": best_path, "trial_args": base_args}


def _sample_hparams(base_args):
    args = copy.deepcopy(base_args)

    # helper: log-uniform jitter around a base value
    def jitter_lr(lr, low=0.3, high=3.0):
        return float(lr * (10 ** random.uniform(np.log10(low), np.log10(high))))

    args.lr_backbone = jitter_lr(base_args.lr_backbone)
    args.lr_heads    = jitter_lr(base_args.lr_heads)

    # weight decay: log-uniform over a safe band
    wd_low, wd_high = 1e-6, 3e-4
    args.weight_decay = float(10 ** random.uniform(np.log10(wd_low), np.log10(wd_high)))

    # percent_t: categorical over available keys
    pt_keys = list(PERCENT_T_DICT.keys())
    args.percent_t = random.choice(pt_keys)

    # K and post_dim (optional; keep ranges tight to avoid big VRAM swings)
    low_K = max(1, int(0.75 * base_args.K))
    high_K = int(1.25 * base_args.K)
    args.K = random.randint(low_K, max(low_K, high_K))
    args.post_dim = int(base_args.post_dim * random.choice([1.0, 1.5]))  # gentle bump

    # Freezing policy: choose one of a few curated strategies
    policies = [
        dict(freeze_all_backbones=True,  unfreeze_global_last=False, unfreeze_local_last=False),  # F0
        dict(freeze_all_backbones=False, unfreeze_global_last=True,  unfreeze_local_last=False),  # F1
        dict(freeze_all_backbones=False, unfreeze_global_last=False, unfreeze_local_last=True),   # F2
        dict(freeze_all_backbones=False, unfreeze_global_last=True,  unfreeze_local_last=True),   # F3
    ]
    pol = random.choice(policies)
    args.freeze_all_backbones = pol["freeze_all_backbones"]
    args.unfreeze_global_last = pol["unfreeze_global_last"]
    args.unfreeze_local_last  = pol["unfreeze_local_last"]

    # Patience tweak for short runs
    args.patience = max(2, int(base_args.patience + random.choice([-1, 0, 1])))

    return args

def main():

    args = define_parser()

    if args.percent_t not in PERCENT_T_DICT:
        raise ValueError(
            f"percent_t '{args.percent_t}' not in PERCENT_T_DICT keys {list(PERCENT_T_DICT.keys())}"
        )
    
    # Seed & ensure output dir
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Using device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Input format: {args.input_format}")
    
    # Build model from args (no hard-coded params)
    model = build_gmic_from_args(args)
    model.to(args.device)

    # ---- Load pretrained weights if requested ----
    load_pretrained_if_requested(model, args)

    # ---- Apply freezing/unfreezing plan ----
    apply_freezing_plan(model, args)
    
    # Create flexible data loader
    data_loader = GMICDataLoader(
        data_path=args.data_path,
        image_path=args.image_path,
        input_format=args.input_format,
        batch_size=args.batch_size,
        use_predefined_splits=args.use_predefined_splits,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    # Print dataset summary
    data_loader.print_summary()
    
    criterion = nn.BCEWithLogitsLoss()

    # If random search is requested, run it and return
    if args.search:
        if args.val_split <= 0.0:
            raise ValueError("Random search requires a nonzero validation split (set --val_split > 0)")
        print(f"\n🔎 Random search: trials={args.search_trials}, max_epochs_per_trial={args.search_max_epochs}")
        set_seed(args.seed)  # reproducibility across trials

        best = {"val_auc": -float("inf"), "best_path": None, "trial_args": None}
        for t in range(args.search_trials):
            trial_args = _sample_hparams(args)
            print(f"\n— Trial {t+1}/{args.search_trials} —")
            print(
                f"lr_backbone={trial_args.lr_backbone:.2e}, "
                f"lr_heads={trial_args.lr_heads:.2e}, "
                f"weight_decay={trial_args.weight_decay:.2e}, "
                f"K={trial_args.K}, post_dim={trial_args.post_dim}, "
                f"percent_t={trial_args.percent_t}, patience={trial_args.patience}, "
                f"freeze_all={trial_args.freeze_all_backbones}, "
                f"unfreeze_global_last={trial_args.unfreeze_global_last}, "
                f"unfreeze_local_last={trial_args.unfreeze_local_last}"
            )
            result = _train_single_run(trial_args)
            print(f"   trial_val_auc={result['val_auc']:.4f}, best_saved={result['best_path']}")

            if result["val_auc"] > best["val_auc"]:
                best = result

        print("\n=== RANDOM SEARCH RESULT ===")
        print(f"Best val AUC: {best['val_auc']:.4f}")
        print(f"Best checkpoint: {best['best_path']}")
        print("Best hyperparams:")
        ta = best["trial_args"]
        print(f"  lr_backbone={ta.lr_backbone:.2e}")
        print(f"  lr_heads   ={ta.lr_heads:.2e}")
        print(f"  K          ={ta.K}")
        print(f"  post_dim   ={ta.post_dim}")
        print(f"  percent_t  ={ta.percent_t}")
        print(f"  patience   ={ta.patience}")
        return  # exits main() after search
    
    if args.evaluate_only:
        print("Loading existing GMIC model for evaluation...")
        load_state_dict_forgiving(model, args.model_path, device=args.device)
        
        # Evaluate on validation set
        print("\n--- Validation Results ---")
        val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
        print(f"Validation - AUC: {val_metrics['auc']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.2f}%, "
              f"Loss: {val_metrics['loss']:.4f}")
        print(f"  [debug] val_samples={val_metrics.get('total_samples', 'N/A')}")

        # Evaluate on test set if available
        if len(data_loader.get_data_for_split('test')) > 0:
            print("\n--- Test Results ---")
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            print(f"Test - AUC: {test_metrics['auc']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.2f}%, "
                  f"Loss: {test_metrics['loss']:.4f}")
            print(f"  [debug] test_samples={test_metrics.get('total_samples', 'N/A')}")

    elif args.test_mode:
        print("Loading existing GMIC model for test evaluation...")
        load_state_dict_forgiving(model, args.model_path, device=args.device)

        # Evaluate only on test set
        if len(data_loader.get_data_for_split('test')) > 0:
            print("\n--- Final Test Results ---")
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            print(f"🧪 Final Test - AUC: {test_metrics['auc']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.2f}%, "
                  f"Loss: {test_metrics['loss']:.4f}")
            print(f"  [debug] test_samples={test_metrics.get('total_samples', 'N/A')}")
        else:
            print("No test data available!")
        
    else:
        # Training mode
        # Grouped optimizer using your lr_heads / lr_backbone
        optimizer, scheduler, early_stopper = configure_optimizers(model, args)

        print(f"\nStarting GMIC training for {args.epochs} epochs...")
        print(f"Training samples: {len(data_loader.get_data_for_split('train'))}")

        print(f"Optimizer groups: lr_backbone={args.lr_backbone:.2e}, lr_heads={args.lr_heads:.2e}, "
            f"weight_decay={args.weight_decay:.2e}")
        print(f"Freezing: freeze_all_backbones={args.freeze_all_backbones}, "
            f"unfreeze_global_last={args.unfreeze_global_last}, "
            f"unfreeze_local_last={args.unfreeze_local_last}")
        
        model.train()
        
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            all_predictions = []
            all_targets = []
            
            # Training loop
            for batch_idx, (inputs, targets, metadata) in enumerate(data_loader.get_batch_iterator('train')):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass through GMIC
                logits = model(inputs)
                logits, targets_aligned = align_logits_targets(logits, targets)
                loss = criterion(logits, targets_aligned)

                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Collect statistics
                total_loss += loss.item()
                num_batches += 1
                
                # Store predictions for training metrics
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                if probs.ndim == 2 and probs.shape[1] > 1:
                    pos_probs = probs[:, 1]
                    pos_targets = targets[:, 1].detach().cpu().numpy()
                else:
                    pos_probs = probs.reshape(-1)
                    pos_targets = targets.detach().cpu().numpy().reshape(-1)

                all_predictions.extend(pos_probs)
                all_targets.extend(pos_targets)
                
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Calculate training metrics
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            train_auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
            predicted_labels = (all_predictions > 0.5).astype(int)
            train_accuracy = 100 * accuracy_score(all_targets, predicted_labels)
            avg_loss = total_loss / max(num_batches, 1)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, "
                  f"Train AUC: {train_auc:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            
            # Validation evaluation
            if len(data_loader.get_data_for_split('val')) > 0:
                val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
                print(f"           - Val AUC: {val_metrics['auc']:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
                
                # Sanity-check split size
                print(f"  [debug] val_samples={val_metrics.get('total_samples', 'N/A')}")

                # ---- NEW: scheduler + early stopping use val_metrics['auc'] ----
                val_auc = val_metrics['auc']                 # <- this is what we pass around
                scheduler.step(val_auc)

                improved = early_stopper.step(val_auc, model)
                if improved:
                    best_model_path = os.path.join(args.out_dir, "gmic_best.pth")
                    print(f"           - ✅ New best validation AUC {val_auc:.4f}! Saved to: {best_model_path}")

                if early_stopper.should_stop():
                    print("           - ⏹ Early stopping triggered.")
                    break

                model.train()

        if early_stopper.best_state is not None:
            model.load_state_dict(early_stopper.best_state)
            best_path = os.path.join(args.out_dir, "gmic_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Loaded best weights and saved to {best_path}")

        last_path = os.path.join(args.out_dir, "gmic_last.pth")
        torch.save(model.state_dict(), last_path)
        print(f"Saved last weights to {last_path}")

        
        # Final evaluation on all splits
        print("\n" + "="*50)
        print("FINAL EVALUATION RESULTS")
        print("="*50)
        
        model.eval()
        
        # Training set evaluation
        train_metrics = evaluate_model(model, data_loader, criterion, args.device, split='train')
        print(f"Final Train - AUC: {train_metrics['auc']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.2f}%")
        
        # Validation set evaluation
        if len(data_loader.get_data_for_split('val')) > 0:
            val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
            print(f"Final Val   - AUC: {val_metrics['auc']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  [debug] val_samples={val_metrics.get('total_samples', 'N/A')}")

        # Test set evaluation
        if len(data_loader.get_data_for_split('test')) > 0:
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            print(f"Final Test  - AUC: {test_metrics['auc']:.4f}, "
                  f"Accuracy: {test_metrics['accuracy']:.2f}%")
        
        print("\nGMIC training completed!")

if __name__ == "__main__":
    main()