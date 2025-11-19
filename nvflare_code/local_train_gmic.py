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
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import GMICDataLoader  # Use the flexible data loader
from src.constants import PERCENT_T_DICT
from nvflare_code.training_core import (
    build_gmic_from_args,
    load_state_dict_forgiving,
    load_pretrained_if_requested,
    configure_optimizers,
    apply_freezing_plan,
    evaluate_model,
    set_seed,
    EarlyStopper,
    _train_single_run,
    _sample_hparams,
)

def define_parser():
    parser = argparse.ArgumentParser(description='Train GMIC model locally using flexible data loader')

    # --- data / IO ---
    parser.add_argument("--data_path", type=str, default="/workspace/data/gmic_format_xai.csv")
    parser.add_argument("--image_path", type=str, default="/workspace/data/processed/cropped_images")
    parser.add_argument("--input_format", type=str, default="auto", choices=["auto", "csv", "pkl"])
    parser.add_argument("--gpu_number", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="/workspace/outputs")
    parser.add_argument("--model_path", type=str, default="/workspace/models/sample_model_5.p")

    # --- dataset split ---
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--use_predefined_splits", action="store_true")
    parser.add_argument("--seed", type=int, default=42)            

    # --- preprocessing control ---
    parser.add_argument("--enable_preprocessing", dest="enable_preprocessing", action="store_true",
                        help="Run full preprocessing pipeline (staging, cropping, centers) if source CSV/PKL provided.")
    parser.add_argument("--no-enable_preprocessing", dest="enable_preprocessing", action="store_false",
                        help="Skip preprocessing; assume data_path already points to processed PKL and images.")
    parser.set_defaults(enable_preprocessing=True)
    # Deprecated flags removed: force_preprocessing now handled automatically (non-destructive)
    parser.add_argument("--cache_validation", action="store_true",
                        help="Validate preprocessing cache against input hash before reuse.")
    # Stage resume is automatic based on progress manifest; explicit resume / force flags removed.
    parser.add_argument("--show_preprocess_status", action="store_true", help="Print preprocessing manifest status and exit.")

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
    # logging
    parser.add_argument("--log_file", type=str, default="", help="Optional log file path to append progress")
    parser.add_argument("--search_log_batch_interval", type=int, default=0, help="If >0, log every N batches during random search trials")
    parser.add_argument("--train_log_batch_interval", type=int, default=5, help="If >0, log every N batches during full training")
    parser.add_argument("--log_lr_each_epoch", action="store_true", help="Log LR snapshot at epoch start/end for full training")
    # multi-GPU
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated GPU ids for training (e.g. '0,1'). For search we use only the first.")
    parser.add_argument("--debug_devices", action="store_true", help="Print parameter device summary and first-batch input device(s) then continue.")
    parser.add_argument("--per_file_logging", action="store_true", help="Log per-file staging and cropping events.")
    # TensorBoard
    parser.add_argument("--tb_log_dir", type=str, default="", help="If set, enable TensorBoard logging to this directory.")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard even if tb_log_dir is set.")
    # Performance / memory optimization
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision (torch.cuda.amp) during training.")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="Accumulate gradients over this many batches before optimizer step.")
    parser.add_argument("--memory_efficient", action="store_true", help="Call torch.cuda.empty_cache() each batch step to reduce fragmentation (slower).")
    return parser.parse_args()

## (Core training utilities moved to nvflare_code.training_core)

def main():
    args = define_parser()

    # Parse GPU list early
    gpu_ids = []
    if args.gpus:
        try:
            gpu_ids = [int(x) for x in args.gpus.split(',') if x.strip() != ""]
        except ValueError:
            raise ValueError(f"Invalid --gpus format: {args.gpus}")
    
    # Always set args.device with proper fallback logic
    if torch.cuda.is_available() and gpu_ids:
        primary_id = gpu_ids[0]
        args.device = torch.device(f"cuda:{primary_id}")
    elif torch.cuda.is_available():
        # CUDA available but no --gpus specified, use --gpu_number
        args.device = torch.device(f"cuda:{args.gpu_number}")
    else:
        # No CUDA available, fall back to CPU
        args.device = torch.device("cpu")
        gpu_ids = []

    if args.percent_t not in PERCENT_T_DICT:
        raise ValueError(
            f"percent_t '{args.percent_t}' not in PERCENT_T_DICT keys {list(PERCENT_T_DICT.keys())}"
        )
    
    # Seed & ensure output dir
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    def write_log(msg: str):
        stamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{stamp}] {msg}"
        print(line)
        if args.log_file:
            try:
                os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
                with open(args.log_file, 'a') as f:
                    f.write(line + '\n')
            except Exception as e:
                print(f"[log-error] {e}")

    write_log(f"Using device: {args.device}")
    write_log(f"Data path: {args.data_path}")
    write_log(f"Input format: {args.input_format}")
    # Flags removed: force_preprocessing / resume_centers / force_center_extraction; behavior is now manifest-driven.
    # TensorBoard (optional) - create EARLY so the directory appears immediately even during long preprocessing
    tb_writer = None
    if args.tb_log_dir and not args.disable_tensorboard:
        from nvflare_code.training_core import create_tb_writer
        tb_writer = create_tb_writer(args.tb_log_dir, enable=True)
        if tb_writer:
            write_log(f"[tensorboard] logging to {args.tb_log_dir}")
        else:
            write_log(f"[tensorboard] failed to initialize writer for {args.tb_log_dir}")

    # Create flexible data loader ONCE (reused for random search trials too)
    data_loader = GMICDataLoader(
        data_path=args.data_path,
        image_path=args.image_path,
        input_format=args.input_format,
        batch_size=args.batch_size,
        use_predefined_splits=args.use_predefined_splits,
        val_split=args.val_split,
        test_split=args.test_split,
        log_file=args.log_file,
        enable_preprocessing=args.enable_preprocessing,
        output_dir=args.out_dir,
        per_file_logging=args.per_file_logging,
    )

    if args.show_preprocess_status:
        data_loader.print_preprocessing_status()
        ok, msg = data_loader.validate_auto_resume()
        print(f"Validation: ok={ok} msg={msg}")
        return
    
    # Print dataset summary
    data_loader.print_summary()
    
    # Using CrossEntropyLoss: model outputs logits (B, num_classes), targets are integer class indices
    criterion = nn.CrossEntropyLoss()


    # If random search is requested, run it and return
    if args.search:
        if args.val_split <= 0.0:
            raise ValueError("Random search requires a nonzero validation split (set --val_split > 0)")
        write_log(f"🔎 Random search: trials={args.search_trials}, max_epochs_per_trial={args.search_max_epochs}")
        set_seed(args.seed)  # reproducibility across trials

        best = {"val_auc": -float("inf"), "best_path": None, "trial_args": None}
        for t in range(args.search_trials):
            trial_args = _sample_hparams(args)
            write_log(f"— Trial {t+1}/{args.search_trials} —")
            write_log(
                " | ".join([
                    f"lr_backbone={trial_args.lr_backbone:.2e}",
                    f"lr_heads={trial_args.lr_heads:.2e}",
                    f"weight_decay={trial_args.weight_decay:.2e}",
                    f"K={trial_args.K}",
                    f"post_dim={trial_args.post_dim}",
                    f"percent_t={trial_args.percent_t}",
                    f"patience={trial_args.patience}",
                    f"freeze_all={trial_args.freeze_all_backbones}",
                    f"unfreeze_global_last={trial_args.unfreeze_global_last}",
                    f"unfreeze_local_last={trial_args.unfreeze_local_last}"
                ])
            )
            # Reuse the single data_loader built once above; only model/hparams vary per trial
            result = _train_single_run(trial_args, data_loader, log_fn=write_log, tb_writer=tb_writer, trial_index=t)
            write_log(f"   trial_val_auc={result['val_auc']:.4f}, best_saved={result['best_path']}")

            # TensorBoard per-trial logging
            if tb_writer:
                step = t + 1  # trial index as step
                prefix = f"search/trial_{step}"
                tb_writer.add_scalar(f"{prefix}/val_auc", result["val_auc"], step)
                # Log hyperparameters as scalars (single-point each)
                tb_writer.add_scalar(f"{prefix}/lr_backbone", trial_args.lr_backbone, step)
                tb_writer.add_scalar(f"{prefix}/lr_heads", trial_args.lr_heads, step)
                tb_writer.add_scalar(f"{prefix}/weight_decay", trial_args.weight_decay, step)
                tb_writer.add_scalar(f"{prefix}/K", trial_args.K, step)
                tb_writer.add_scalar(f"{prefix}/post_dim", trial_args.post_dim, step)
                tb_writer.add_scalar(f"{prefix}/percent_t", float(trial_args.percent_t), step)
                tb_writer.add_scalar(f"{prefix}/patience", trial_args.patience, step)
                tb_writer.add_scalar(f"{prefix}/freeze_all_backbones", 1.0 if trial_args.freeze_all_backbones else 0.0, step)
                tb_writer.add_scalar(f"{prefix}/unfreeze_global_last", 1.0 if trial_args.unfreeze_global_last else 0.0, step)
                tb_writer.add_scalar(f"{prefix}/unfreeze_local_last", 1.0 if trial_args.unfreeze_local_last else 0.0, step)

            if result["val_auc"] > best["val_auc"]:
                best = result
                if tb_writer:
                    tb_writer.add_scalar("search/best_val_auc", best["val_auc"], t + 1)

        write_log("=== RANDOM SEARCH RESULT ===")
        write_log(f"Best val AUC: {best['val_auc']:.4f}")
        write_log(f"Best checkpoint: {best['best_path']}")
        write_log("Best hyperparams:")
        ta = best["trial_args"]
        write_log(f"  lr_backbone={ta.lr_backbone:.2e}")
        write_log(f"  lr_heads   ={ta.lr_heads:.2e}")
        write_log(f"  K          ={ta.K}")
        write_log(f"  post_dim   ={ta.post_dim}")
        write_log(f"  percent_t  ={ta.percent_t}")
        write_log(f"  patience   ={ta.patience}")
        if tb_writer and best["trial_args"] is not None:
            # Log a summary of best hyperparameters (use step = total trials)
            final_step = args.search_trials
            tb_writer.add_scalar("search/final_best_val_auc", best["val_auc"], final_step)
            tb_writer.add_text("search/best_checkpoint", str(best["best_path"]), final_step)
            tb_writer.add_text(
                "search/best_hparams",
                "\n".join([
                    f"lr_backbone={ta.lr_backbone:.2e}",
                    f"lr_heads={ta.lr_heads:.2e}",
                    f"weight_decay={ta.weight_decay:.2e}",
                    f"K={ta.K}",
                    f"post_dim={ta.post_dim}",
                    f"percent_t={ta.percent_t}",
                    f"patience={ta.patience}",
                    f"freeze_all_backbones={ta.freeze_all_backbones}",
                    f"unfreeze_global_last={ta.unfreeze_global_last}",
                    f"unfreeze_local_last={ta.unfreeze_local_last}",
                ])
            )
            tb_writer.flush()
            tb_writer.close()
        return  # exits main() after search
    
    # Build model from args (no hard-coded params) ONLY if not doing search
    model = build_gmic_from_args(args)
    model.to(args.device)

    # ---- Load pretrained weights if requested ----
    load_pretrained_if_requested(model, args)

    # ---- Apply freezing/unfreezing plan ----
    apply_freezing_plan(model, args)

    # Wrap with DataParallel ONLY for full training (not search/eval) and if >1 GPU ids
    using_multi_gpu = (len(gpu_ids) > 1) and (not args.search) and (not args.evaluate_only) and (not args.test_mode)
    if using_multi_gpu:
        write_log(f"Enabling DataParallel on GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    # Device diagnostics (after potential DataParallel wrap)
    if args.debug_devices:
        from nvflare_code.training_core import summarize_parameter_devices
        summary = summarize_parameter_devices(model)
        write_log(f"[devices] param_devices={summary['unique_devices']} counts={summary['device_counts']}")
        for ex in summary['examples']:
            write_log(f"[devices] example {ex}")
    
    if args.evaluate_only:
        write_log("Loading existing GMIC model for evaluation...")
        load_state_dict_forgiving(model, args.model_path, device=args.device)

        # Evaluate on validation set
        write_log("--- Validation Results ---")
        val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
        write_log(f"Validation - AUC: {val_metrics['auc']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%, Loss: {val_metrics['loss']:.4f}")
        write_log(f"[debug] val_samples={val_metrics.get('total_samples', 'N/A')}")

        # Evaluate on test set if available
        if len(data_loader.get_data_for_split('test')) > 0:
            write_log("--- Test Results ---")
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            write_log(f"Test - AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}")
            write_log(f"[debug] test_samples={test_metrics.get('total_samples', 'N/A')}")

    elif args.test_mode:
        write_log("Loading existing GMIC model for test evaluation...")
        load_state_dict_forgiving(model, args.model_path, device=args.device)

        # Evaluate only on test set
        if len(data_loader.get_data_for_split('test')) > 0:
            write_log("--- Final Test Results ---")
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            write_log(f"Final Test - AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.2f}%, Loss: {test_metrics['loss']:.4f}")
            write_log(f"[debug] test_samples={test_metrics.get('total_samples', 'N/A')}")
        else:
            write_log("No test data available!")
        
    else:
        # Training mode
        optimizer, scheduler, early_stopper = configure_optimizers(model, args)
        write_log(f"Starting GMIC training for {args.epochs} epochs")
        write_log(f"Training samples: {len(data_loader.get_data_for_split('train'))}")
        write_log(f"Optimizer groups: lr_backbone={args.lr_backbone:.2e}, lr_heads={args.lr_heads:.2e}, weight_decay={args.weight_decay:.2e}")
        write_log(f"Freezing: freeze_all_backbones={args.freeze_all_backbones}, unfreeze_global_last={args.unfreeze_global_last}, unfreeze_local_last={args.unfreeze_local_last}")
        write_log(f"Perf flags: use_amp={args.use_amp} grad_accumulation={args.grad_accumulation} memory_efficient={args.memory_efficient}")

        # AMP GradScaler (only if CUDA + requested)
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())
        grad_accum = max(1, int(getattr(args, 'grad_accumulation', 1)))

        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            num_batches = 0
            all_predictions = []
            all_targets = []

            # Ensure gradients are cleared at epoch start
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, (inputs, targets, metadata) in enumerate(data_loader.get_batch_iterator('train')):
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)
                with torch.cuda.amp.autocast(enabled=args.use_amp and torch.cuda.is_available()):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                # Scale loss for accumulation to keep global batch comparable
                eff_loss = loss / grad_accum
                if args.use_amp and torch.cuda.is_available():
                    scaler.scale(eff_loss).backward()
                else:
                    eff_loss.backward()

                do_step = ((batch_idx + 1) % grad_accum == 0)
                if do_step:
                    if args.use_amp and torch.cuda.is_available():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    # Prepare for next accumulation cycle
                    optimizer.zero_grad(set_to_none=True)
                    if args.memory_efficient and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if batch_idx == 0:  # per-epoch snapshots & diagnostics
                    lrs = [pg['lr'] for pg in optimizer.param_groups]
                    write_log(
                        "Epoch {} LRs {}".format(
                            epoch + 1,
                            ",".join(f"{lr:.2e}" for lr in lrs)
                        )
                    )
                    if args.debug_devices:
                        from nvflare_code.training_core import first_batch_input_device_str, summarize_parameter_devices
                        inp_dev = first_batch_input_device_str(inputs)
                        write_log(f"[devices] first_batch_input_devices={inp_dev}")
                        # Only log param summary once overall (epoch==0)
                        if epoch == 0:
                            summary = summarize_parameter_devices(model)
                            write_log(f"[devices] param_devices={summary['unique_devices']} counts={summary['device_counts']}")

                total_loss += loss.item()
                num_batches += 1

                probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
                if probs.ndim == 2 and probs.shape[1] > 1:
                    pos_probs = probs[:, 1]
                else:
                    pos_probs = probs.reshape(-1)
                pos_targets = targets.detach().cpu().numpy().reshape(-1)
                all_predictions.extend(pos_probs)
                all_targets.extend(pos_targets)

                interval = getattr(args, 'train_log_batch_interval', 0)
                if interval and batch_idx % interval == 0:
                    write_log(f"Epoch {epoch + 1} Batch {batch_idx} Loss {loss.item():.4f}")

            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            train_auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
            predicted_labels = (all_predictions > 0.5).astype(int)
            train_accuracy = 100 * accuracy_score(all_targets, predicted_labels)
            avg_loss = total_loss / max(num_batches, 1)

            # Final optimizer step if there is a remainder accumulation not yet stepped
            if (num_batches % grad_accum) != 0:
                if args.use_amp and torch.cuda.is_available():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if args.memory_efficient and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            lrs_end = [pg['lr'] for pg in optimizer.param_groups]
            write_log(
                f"Epoch {epoch + 1} Train Loss {avg_loss:.4f} Train AUC {train_auc:.4f} Train Acc {train_accuracy:.2f}% LRs_end "
                + ",".join(f"{lr:.2e}" for lr in lrs_end)
            )
            if tb_writer:
                tb_writer.add_scalar("train/loss", avg_loss, epoch + 1)
                tb_writer.add_scalar("train/auc", train_auc, epoch + 1)
                tb_writer.add_scalar("train/accuracy", train_accuracy, epoch + 1)
                for i, lr in enumerate(lrs_end):
                    tb_writer.add_scalar(f"lr/group_{i}", lr, epoch + 1)

            if len(data_loader.get_data_for_split('val')) > 0:
                val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
                write_log(
                    f"Epoch {epoch + 1} Val AUC {val_metrics['auc']:.4f} Val Acc {val_metrics['accuracy']:.2f}% Val Loss {val_metrics['loss']:.4f}"
                )
                if tb_writer:
                    tb_writer.add_scalar("val/auc", val_metrics['auc'], epoch + 1)
                    tb_writer.add_scalar("val/loss", val_metrics['loss'], epoch + 1)
                    tb_writer.add_scalar("val/accuracy", val_metrics['accuracy'], epoch + 1)
                write_log(f"[debug] val_samples={val_metrics.get('total_samples', 'N/A')}")
                val_auc = val_metrics['auc']
                scheduler.step(val_auc)
                improved = early_stopper.step(val_auc, model)
                if improved:
                    best_model_path = os.path.join(args.out_dir, "gmic_best.pth")
                    write_log(f"New best val AUC {val_auc:.4f} saved {best_model_path}")
                if early_stopper.should_stop():
                    write_log("Early stopping triggered")
                    break
                model.train()

        # Post-training save & evaluation (only if we actually trained)
        if early_stopper.best_state is not None:
            model.load_state_dict(early_stopper.best_state)
            best_path = os.path.join(args.out_dir, "gmic_best.pth")
            underlying = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(underlying.state_dict(), best_path)
            write_log(f"Loaded best weights and saved to {best_path}")

        last_path = os.path.join(args.out_dir, "gmic_last.pth")
        underlying = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(underlying.state_dict(), last_path)
        write_log(f"Saved last weights to {last_path}")

        write_log("FINAL EVALUATION RESULTS")
        model.eval()
        train_metrics = evaluate_model(model, data_loader, criterion, args.device, split='train')
        write_log(f"Final Train - AUC {train_metrics['auc']:.4f} Acc {train_metrics['accuracy']:.2f}%")
        if len(data_loader.get_data_for_split('val')) > 0:
            val_metrics = evaluate_model(model, data_loader, criterion, args.device, split='val')
            write_log(f"Final Val   - AUC {val_metrics['auc']:.4f} Acc {val_metrics['accuracy']:.2f}%")
            write_log(f"[debug] val_samples={val_metrics.get('total_samples', 'N/A')}")
        if len(data_loader.get_data_for_split('test')) > 0:
            test_metrics = evaluate_model(model, data_loader, criterion, args.device, split='test')
            write_log(f"Final Test  - AUC {test_metrics['auc']:.4f} Acc {test_metrics['accuracy']:.2f}%")
        if tb_writer:
            tb_writer.flush()
        write_log("GMIC training completed")
        if tb_writer:
            tb_writer.close()

if __name__ == "__main__":
    main()