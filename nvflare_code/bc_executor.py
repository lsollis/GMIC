# ============================================================================
# bc_executor.py - NVFLARE Executor with flexible CSV/PKL input
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import json

from nvflare.app_common.utils.fl_model_utils import FLModel, FLModelUtils, ParamsType
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager
from nvflare.app_common.utils.fl_model_utils import FLModelUtils, FLModel, ParamsType
from nvflare.apis.dxo import MetaKey
from nvflare.apis.shareable import Shareable, ReturnCode
from nvflare.app_common.app_constant import AppConstants
import numpy as np
import torch

# Import the smart data loader (corrected import)
from nvflare_code.data_loader import GMICDataLoader  # This should be the final data loader class
from src.modeling import gmic
from src.constants import PERCENT_T_DICT

class GMICFederatedExecutor(Executor):
    """Corrected NVFLARE Executor with smart preprocessing and comprehensive data loading"""
    
    def __init__(
        self,
        epochs: int = 5,
        lr: float = 1e-4,
        batch_size: int = 4,
        data_path: str = "/workspace/data/gmic_format_xai.csv",
        image_path: str = "/workspace/data/XAI_output",
        model_path: str = "/workspace/models/sample_model_1.p",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        input_format: str = "auto",
        enable_preprocessing: bool = True,  # NEW: Enable preprocessing
        force_preprocessing: bool = True,   # NEW: Force preprocessing
        cache_validation: bool = True,       # NEW: Cache validation
        use_predefined_splits: bool = True,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,               # NEW: Random seed
        output_dir: str = "/workspace/processed_data",  # NEW: Output directory
        num_processes: int = 4,              # NEW: Number of processes
        pre_train_task_name: str = AppConstants.TASK_GET_WEIGHTS,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        validate_task_name: str = AppConstants.TASK_VALIDATION,
        test_task_name: str = "test",
        exclude_vars=None,
        gmic_parameters: dict | None = None,
        loss: str = "bce_with_logits",
        optimizer: dict | None = None,
        task_names: dict | None = None
    ):
        """
        GMIC Federated Executor with all required parameters
        
        Args:
            epochs: Local training epochs per round
            lr: Learning rate for GMIC training
            batch_size: Batch size for mammography images
            data_path: Path to CSV or PKL file with mammography data
            image_path: Base path for images (raw or processed)
            model_path: Path to save/load model (.pth extension)
            device: Training device (cuda/cpu)
            input_format: "auto", "csv", "pkl", or "raw_pkl"
            enable_preprocessing: If True, enable preprocessing pipeline
            force_preprocessing: If True, always run preprocessing
            cache_validation: If True, validate cache integrity
            use_predefined_splits: Use split_group column from data
            val_split: Fraction for validation if creating new splits
            test_split: Fraction for testing if creating new splits
            random_seed: Random seed for reproducible splits
            output_dir: Directory to save processed data
            num_processes: Number of processes for preprocessing
        """
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.data_path = data_path
        self.image_path = image_path
        self.model_path = model_path
        self.device = device
        self.input_format = input_format
        self.enable_preprocessing = enable_preprocessing
        self.force_preprocessing = force_preprocessing
        self.cache_validation = cache_validation
        self.use_predefined_splits = use_predefined_splits
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.output_dir = output_dir
        self.num_processes = num_processes
        
        # Task names
        self.pre_train_task_name = pre_train_task_name
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name
        self.validate_task_name = validate_task_name
        self.test_task_name = test_task_name
        self.exclude_vars = exclude_vars
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.data_loader = None
        self.persistence_manager = None
        self.best_metrics = {"val_auc": 0.0, "test_auc": 0.0}
        
        # Metrics tracking
        self.training_history = []
        self.validation_history = []
        self.test_history = []

        # GMIC parameters
        self.gmic_parameters_cfg = gmic_parameters or {}
        self.loss_name = loss
        self.optimizer_cfg = optimizer or {"name": "adam", "lr": self.lr, "weight_decay": 0.0}
        self.task_names = task_names or {}
        # optionally let JSON override task strings
        self.pre_train_task_name = self.task_names.get("pre_train", self.pre_train_task_name)
        self.train_task_name = self.task_names.get("train", self.train_task_name)
        self.validate_task_name = self.task_names.get("validate", self.validate_task_name)
        self.test_task_name = self.task_names.get("test", self.test_task_name)
        self.submit_model_task_name = self.task_names.get("submit_model", self.submit_model_task_name)
        
        
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle NVFLARE events"""
        if event_type == EventType.START_RUN:
            self.initialize()
        elif event_type == EventType.END_RUN:
            self._save_final_results(fl_ctx)
    
    def initialize(self):
        """Initialize data loader, model, optimizer, and bookkeeping.
        Backward-compatible: will use JSON-driven overrides if present, otherwise defaults.
        """
        import math

        # ----------------------------
        # 1) Determine effective data paths (use cache when present)
        # ----------------------------
        processed_pkl = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cropped_dir = os.path.join(self.output_dir, "cropped_images")

        if os.path.isfile(processed_pkl) and os.path.isdir(cropped_dir):
            effective_data_path = processed_pkl
            effective_image_path = cropped_dir
            effective_input_format = "pkl"
            effective_enable_preprocessing = False
            print(f"[EXEC] Using cached processed data: {processed_pkl}")
        else:
            effective_data_path = self.data_path
            effective_image_path = self.image_path
            effective_input_format = self.input_format
            effective_enable_preprocessing = self.enable_preprocessing
            print(f"[EXEC] Using source data: {self.data_path} (enable_preprocessing={self.enable_preprocessing})")

        # ----------------------------
        # 2) Build data loader with EFFECTIVE values
        # ----------------------------
        self.data_loader = GMICDataLoader(
            data_path=effective_data_path,
            image_path=effective_image_path,
            batch_size=self.batch_size,
            random_seed=self.random_seed,
            use_predefined_splits=self.use_predefined_splits,
            val_split=self.val_split,
            test_split=self.test_split,
            input_format=effective_input_format,
            enable_preprocessing=effective_enable_preprocessing,
            output_dir=self.output_dir,
            num_processes=self.num_processes,
            force_preprocessing=self.force_preprocessing,
            cache_validation=self.cache_validation,
        )

        self.data_loader.print_summary()

        # ----------------------------
        # 3) Compute iterations per phase (avoid zeros)
        # ----------------------------
        split_info = self.data_loader.get_split_info()
        self._n_train_iterations = max(1, math.ceil(split_info["train_size"] / max(self.batch_size, 1)))
        self._n_val_iterations   = max(1, math.ceil(split_info["val_size"]   / max(self.batch_size, 1)))
        self._n_test_iterations  = max(1, math.ceil(split_info["test_size"]  / max(self.batch_size, 1)))

        # ----------------------------
        # 4) Build GMIC parameters from defaults + optional JSON overrides
        # ----------------------------
        def _to_tuple(x):
            return tuple(x) if isinstance(x, (list,)) else x

        gmic_defaults = {
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
            "gpu_number": 0,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "image_path": effective_image_path,     # use EFFECTIVE path
            "cam_size": (46, 30),
            "K": 6,
            "crop_shape": (256, 256),
            "post_processing_dim": 256,
            "num_classes": 2,
            "use_v1_global": False,
            "percent_t": PERCENT_T_DICT["1"],
        }

        # Pull overrides if __init__ set them; otherwise empty dict
        overrides = dict(getattr(self, "gmic_parameters_cfg", {}) or {})

        # Normalize list->tuple fields from JSON
        for k in ["max_crop_noise", "cam_size", "crop_shape"]:
            if k in overrides:
                overrides[k] = _to_tuple(overrides[k])

        # Support "percent_t": "auto:1" form
        pt = overrides.get("percent_t")
        if isinstance(pt, str) and pt.startswith("auto:"):
            key = pt.split(":", 1)[1]
            overrides["percent_t"] = PERCENT_T_DICT.get(key, PERCENT_T_DICT["1"])

        self.gmic_parameters = {**gmic_defaults, **overrides}

        # ----------------------------
        # 5) Create model & move to device
        # ----------------------------
        self.model = gmic.GMIC(self.gmic_parameters)
        self.model.to(self.device)

        # ----------------------------
        # 6) Loss & Optimizer (honor JSON if present; otherwise keep current defaults)
        # ----------------------------
        loss_name = (getattr(self, "loss_name", "bce_with_logits") or "bce_with_logits").lower()
        if loss_name in ["bce_with_logits", "bce_logits"]:
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name in ["bce"]:
            self.criterion = nn.BCELoss()
        elif loss_name in ["cross_entropy", "ce"]:
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        opt_cfg = {**{"name": "adam", "lr": self.lr, "weight_decay": 0.0}, **(getattr(self, "optimizer_cfg", {}) or {})}
        name = (opt_cfg.get("name") or "adam").lower()
        lr = float(opt_cfg.get("lr", self.lr))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        if name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            momentum = float(opt_cfg.get("momentum", 0.9))
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        # ----------------------------
        # 7) If CSV input, save a PKL snapshot for reproducibility
        # ----------------------------
        if self.data_loader.input_format == "csv":
            pkl_output_path = os.path.join(os.path.dirname(self.model_path), "converted_data.pkl")
            try:
                self.data_loader.save_pkl_format(pkl_output_path)
            except Exception as e:
                print(f"[WARN] Failed to save PKL snapshot: {e!r}")

        print(f"Initialized GMIC executor with {self.data_loader.input_format.upper()} data: {split_info}")
    
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        try:
            # --- round headers ---
            current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0))
            total_rounds = int(shareable.get_header(AppConstants.NUM_ROUNDS, 1))
            self.log_info(fl_ctx, f"Starting round {current_round}/{total_rounds} - Train+Validate+Test")

            # --- receive global model (support both FLModel and legacy raw weights) ---
            fl_model_in = FLModelUtils.from_shareable(shareable)
            if fl_model_in and fl_model_in.params:
                safe_params = self._ensure_torch_state(fl_model_in.params, fl_ctx)
                self.model.load_state_dict(safe_params, strict=False)
                self.log_info(fl_ctx, "Loaded global model from server (FLModel; numpy→torch as needed)")
            elif AppConstants.MODEL_WEIGHTS in shareable:
                legacy_params = shareable[AppConstants.MODEL_WEIGHTS]
                safe_params = self._ensure_torch_state(legacy_params, fl_ctx)
                self.model.load_state_dict(safe_params, strict=False)
                self.log_info(fl_ctx, "Loaded global model from server (legacy MODEL_WEIGHTS; numpy→torch as needed)")

            # --- TRAIN ---
            self.log_info(fl_ctx, "Phase 1: Training...")
            train_metrics = self._local_train(fl_ctx, abort_signal, shareable)
            num_examples = int(train_metrics.get("train_samples", self.batch_size))
            if num_examples <= 0:
                # ensure a positive weight so FedAvg doesn’t drop this update
                num_examples = max(1, self.batch_size)

            if not abort_signal.triggered:
                self._save_local_model(fl_ctx, shareable)

            # --- VAL ---
            self.log_info(fl_ctx, "Phase 2: Validation...")
            val_metrics = self._evaluate_model(fl_ctx, split="val")

            # --- TEST (final round only) ---
            if current_round == (total_rounds - 1):
                self.log_info(fl_ctx, "Phase 3: Testing...")
                test_metrics = self._evaluate_model(fl_ctx, split="test")
            else:
                self.log_info(fl_ctx, "Phase 3: Skipping test (not final round)")
                test_metrics = {}

            # --- package update for FedAvg (DXO/FLModel) ---
            # send numpy arrays (server-side aggregation is perfectly happy with numpy)
            updated_weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
            # Combine metrics and ensure all are numeric
            combined_metrics = {**train_metrics, **val_metrics, **test_metrics, "round": int(current_round)}

            # Safety filter: only include numeric values
            numeric_metrics = {}
            for key, value in combined_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_metrics[key] = float(value) if not isinstance(value, int) else int(value)
                else:
                    self.log_warning(fl_ctx, f"Skipping non-numeric metric: {key} = {value}")

            fl_model_out = FLModel(
                params=updated_weights,
                params_type=ParamsType.FULL,
                metrics=numeric_metrics,
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: num_examples, "num_examples": num_examples},
            )
            reply = FLModelUtils.to_shareable(fl_model_out)
            reply.set_return_code(ReturnCode.OK)

            # summary log
            self.log_info(
                fl_ctx,
                f"Round {current_round} done - "
                f"train_loss={train_metrics.get('train_loss', 'N/A')} "
                f"val_acc={val_metrics.get('val_accuracy','N/A')} "
                + (f"test_acc={test_metrics.get('test_accuracy','N/A')}" if test_metrics else "")
            )
            return reply

        except Exception as e:
            self.log_exception(fl_ctx, f"Error in all-in-one task: {e}")
            # signal failure explicitly so the server ignores this result instead of trying to parse as DXO
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _ensure_torch_state(self, params: dict, fl_ctx=None) -> dict:
        """Convert incoming FLModel.params (possibly numpy) into a proper torch state_dict."""
        ref = self.model.state_dict()
        device = next(self.model.parameters()).device
        out = {}

        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                t = v.to(device)
            elif isinstance(v, np.ndarray):
                # match dtype to reference param if available
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.from_numpy(v).to(device=device, dtype=dtype)
            elif np.isscalar(v):
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.tensor(v, device=device, dtype=dtype)
            else:
                if fl_ctx:
                    self.log_warning(fl_ctx, f"Skipping param {k} of unsupported type {type(v)}")
                continue

            # common BN buffer gotcha
            if "num_batches_tracked" in k:
                t = t.to(torch.long)
            out[k] = t
        return out
    
    def _handle_train_task(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Handle training task"""
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.WEIGHTS:
            return make_reply(ReturnCode.BAD_TASK_DATA)
        
        # Load global weights
        torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
        self.model.load_state_dict(torch_weights)
        
        # Perform local training
        train_metrics = self._local_train(fl_ctx, abort_signal, shareable)
        
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        
        # Save local model
        self._save_local_model(fl_ctx, shareable)
        
        # Return updated weights with metrics
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={
                "train_loss": train_metrics["avg_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_auc": train_metrics["auc"],
                "train_samples": train_metrics["num_samples"],
                MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_train_iterations
            }
        )
        shareable = dxo.to_shareable()
        shareable.set_header(AppConstants.MODEL_OWNER, fl_ctx.get_identity_name())
        return shareable
    
    def _handle_validate_task(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Handle validation task"""
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHTS:
            torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
            self.model.load_state_dict(torch_weights)
        
        # Validate on validation split
        val_metrics = self._evaluate_model(fl_ctx, split='val')
        self.validation_history.append(val_metrics)
        
        # Update best validation metrics
        if val_metrics["auc"] > self.best_metrics["val_auc"]:
            self.best_metrics["val_auc"] = val_metrics["auc"]
            self._save_best_model(fl_ctx, val_metrics, shareable, model_type="best_val")
        
        # Return validation metrics
        dxo = DXO(
            data_kind=DataKind.METRICS,
            data={
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"]
            }
        )
        return dxo.to_shareable()
    
    def _handle_test_task(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        """Handle test task - final evaluation on held-out test set"""
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHTS:
            torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
            self.model.load_state_dict(torch_weights)
        
        # Evaluate on test split
        test_metrics = self._evaluate_model(fl_ctx, split='test')
        self.test_history.append(test_metrics)
        
        # Update best test metrics and save best test model
        if test_metrics["auc"] > self.best_metrics["test_auc"]:
            self.best_metrics["test_auc"] = test_metrics["auc"]
            self._save_best_model(fl_ctx, test_metrics, shareable, model_type="best_test")
        
        self.log_info(fl_ctx, 
            f"🧪 Test Results - AUC: {test_metrics['auc']:.4f}, "
            f"Accuracy: {test_metrics['accuracy']:.2f}%, "
            f"F1: {test_metrics['f1']:.4f}")
        
        # Return test metrics
        dxo = DXO(
            data_kind=DataKind.METRICS,
            data={
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"]
            }
        )
        return dxo.to_shareable()
    
    def _get_model_weights(self) -> Shareable:
        """Get current GMIC model weights"""
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_train_iterations}
        )
        return outgoing_dxo.to_shareable()
    
    def _local_train(self, fl_ctx, abort_signal, shareable: Shareable):
        """Local training on training split"""
        
        self.model.train()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.epochs):
            self.log_info(fl_ctx, f"Training epoch {epoch + 1}/{self.epochs}")
            
            for batch_idx, (inputs, targets, metadata) in enumerate(self.data_loader.get_batch_iterator('train')):
                if abort_signal.triggered:
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Collect metrics
                total_loss += loss.item()
                num_samples += inputs.size(0)
                
                # Store predictions for metrics calculation
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                all_predictions.extend(predictions[:, 1])  # Malignant probability
                all_targets.extend(targets[:, 1].cpu().numpy())  # Malignant label
                
                if batch_idx % 5 == 0:
                    self.log_info(fl_ctx, 
                        f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate training metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
        predicted_labels = (all_predictions > 0.5).astype(int)
        accuracy = 100 * accuracy_score(all_targets, predicted_labels)
        avg_loss = total_loss / max(num_samples, 1)
        
        train_metrics = {
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "train_auc": float(auc),
            "train_samples": int(num_samples),
            "epochs_completed": int(self.epochs)
        }
        
        self.training_history.append(train_metrics)
        self.log_info(fl_ctx, f"Training completed - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Accuracy: {accuracy:.2f}%")
        
        return train_metrics
    
    def _evaluate_model(self, fl_ctx: FLContext, split='val'):
        """Evaluate model on specified split (val or test)"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets, metadata in self.data_loader.get_batch_iterator(split, shuffle=False):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                all_predictions.extend(predictions[:, 1])  # Malignant probability
                all_targets.extend(targets[:, 1].cpu().numpy())  # Malignant label
        
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Core metrics
        auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
        predicted_labels = (all_predictions > 0.5).astype(int)
        accuracy = 100 * accuracy_score(all_targets, predicted_labels)
        avg_loss = total_loss / max(num_batches, 1)
        
        # Additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, predicted_labels, average='binary', zero_division=0
        )
        
        # ✅ FIXED:
        metrics = {
            f"{split}_auc": float(auc),
            f"{split}_accuracy": float(accuracy),
            f"{split}_loss": float(avg_loss),
            f"{split}_precision": float(precision),
            f"{split}_recall": float(recall),
            f"{split}_f1": float(f1),
            f"{split}_samples": int(len(all_targets)),
            # Removed "split": split
        }
        
        self.log_info(fl_ctx, 
            f"{split.capitalize()} evaluation - AUC: {auc:.4f}, "
            f"Accuracy: {accuracy:.2f}%, F1: {f1:.4f}")
        
        return metrics
    
    def _save_local_model(self, fl_ctx: FLContext, shareable: Shareable):
        """Save current model state"""
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM)
        )
        models_dir = os.path.join(run_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save to persistent location
        persistent_dir = "/workspace/gmic_results"
        os.makedirs(persistent_dir, exist_ok=True)
        
        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        
        # Save model with round info
        model_path = f"{persistent_dir}/{client_name}_gmic_model_round_{current_round}.pth"
        torch.save(self.model.state_dict(), model_path)
        
        self.log_info(fl_ctx, f"Model saved: {model_path}")
    
    def _save_best_model(self, fl_ctx, metrics, shareable: Shareable, model_type="best"):
        """Save best performing model"""
        persistent_dir = "/workspace/gmic_results"
        os.makedirs(persistent_dir, exist_ok=True)
        
        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        
        # Save best model
        model_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save metrics
        metrics_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "client": client_name,
                "round": current_round,
                "model_type": model_type,
                **metrics
            }, f, indent=2)
        
        self.log_info(fl_ctx, f"✅ {model_type} model saved with {metrics['split']} AUC={metrics['auc']:.4f}")
    
    def _load_local_model(self, fl_ctx: FLContext):
        """Load previously saved GMIC model"""
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM)
        )
        models_dir = os.path.join(run_dir, "models")
        if not os.path.exists(models_dir):
            return None
        
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(self.model_path),
            default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self.exclude_vars)
        return ml
    
    def _save_final_results(self, fl_ctx: FLContext):
        """Save final training results and metrics summary"""
        persistent_dir = "/workspace/gmic_results"
        os.makedirs(persistent_dir, exist_ok=True)
        
        client_name = fl_ctx.get_identity_name()
        
        # Save complete training history
        results = {
            "client_name": client_name,
            "training_history": self.training_history,
            "validation_history": self.validation_history,
            "test_history": self.test_history,
            "best_metrics": self.best_metrics,
            "data_splits": self.data_loader.get_split_info(),
            "class_distributions": {
                "train": self.data_loader.get_class_distribution("train"),
                "val": self.data_loader.get_class_distribution("val"),
                "test": self.data_loader.get_class_distribution("test")
            }
        }
        
        results_path = f"{persistent_dir}/{client_name}_final_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.log_info(fl_ctx, f"Final results saved to: {results_path}")
        
        # Print summary
        self.log_info(fl_ctx, "Training Complete!")
        self.log_info(fl_ctx, f"Best Validation AUC: {self.best_metrics['val_auc']:.4f}")
        self.log_info(fl_ctx, f"Best Test AUC: {self.best_metrics['test_auc']:.4f}")