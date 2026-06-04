# ============================================================================
# bc_executor.py - NVFLARE Executor with flexible CSV/PKL input
# ============================================================================

import os
import csv
import copy
import json
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

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

from train.training_core import create_tb_writer, first_batch_input_device_str, summarize_parameter_devices
from data_loader.data_loader import GMICDataLoader  # This should be the final data loader class
from constants.constants import PERCENT_T_DICT
from train.training_core import (
    build_gmic_from_args,
    load_pretrained_if_requested,
    configure_optimizers,
    apply_freezing_plan,
    evaluate_model,
    set_seed,
    EarlyStopper,
)
from fl_utils import (
    bn_state_keys,
    load_global_into,
    proximal_penalty,
    clone_detached_state,
    tolerant_load_pretrained,
    gmic_malignant_loss,
    gmic_focal_loss,
    malignant_score,
    gmic_outputs,
)

# Allowed federated methods (selected via the client job config "method" arg).
VALID_METHODS = ("local", "fedavg", "fedprox", "fedbn", "ditto", "ditto_modulewise")

# Allowed loss values. "gmic"/"gmic_bce" -> native GMIC deep-supervised 3-head malignant
# BCE + L1 (criterion unused; computed in _compute_task_loss). "focal" -> same 3-head
# structure with focal BCE (criterion unused). "cross_entropy"/"ce" -> legacy CrossEntropyLoss.
# Plain "bce" is NOT valid.
VALID_LOSSES = ("gmic", "gmic_bce", "focal", "cross_entropy", "ce")


def resolve_criterion(loss_name: str):
    """Map a config `loss` string to a criterion (or None for the GMIC/focal paths).

    Returns None for gmic/gmic_bce/focal (loss computed in _compute_task_loss), an
    nn.CrossEntropyLoss for cross_entropy/ce, and raises ValueError on anything else
    so an unknown loss fails loudly instead of silently routing to a default."""
    name = (loss_name or "gmic").lower()
    if name in ("gmic", "gmic_bce", "focal"):
        return None
    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"Unknown loss '{name}'. Valid: {VALID_LOSSES} "
        f"('gmic'/'gmic_bce' = native GMIC malignant loss; 'focal' = focal BCE variant; "
        f"'cross_entropy'/'ce' = legacy). Plain 'bce' is not supported."
    )


class GMICFederatedExecutor(Executor):
    """NVFLARE Executor mirroring local_train_gmic features (preprocessing, freezing, grouped LR, logging)."""

    def __init__(
            self,
            epochs: int = 50,
            patience: int = 4,
            lr_heads: float = 1e-4,
            lr_backbone: float = 1e-5,
            weight_decay: float = 1e-5,
            batch_size: int = 128,
            data_path: str = "/workspace/data/gmic_format_xai.csv",
            image_path: str = "/workspace/data/XAI_output",
            model_path: str = "/workspace/models/sample_model_5.p",
            device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
            input_format: str = "csv",
            enable_preprocessing: bool = True,
            force_preprocessing: bool = True,
            cache_validation: bool = True,
            use_predefined_splits: bool = True,
            val_split: float = 0.15,
            test_split: float = 0.15,
            random_seed: int = 42,
            output_dir: str = "/workspace/data/processed",
            results_dir: str = "/workspace/gmic_results_centralized",
            out_dir: str | None = None,
            num_processes: int = 4,
            log_file: str = "/workspace/outputs_centralized/executor.log",
            file_integrity_check: bool = True,
            fail_on_integrity_error: bool = False,
            # Freezing options
            freeze_all_backbones: bool = False,
            unfreeze_global_last: bool = False,
            unfreeze_local_last: bool = False,
            # Percent_t key
            percent_t_key: str = "1",
            percent_t: str | None = None,
            # ROI / architecture params
            K: int = 6,
            cam_h: int = 46,
            cam_w: int = 30,
            crop_h: int = 256,
            crop_w: int = 256,
            post_dim: int = 256,
            num_classes: int = 2,
            use_v1_global: bool = False,
            lambda_l1: float = 1e-5,
            # Optional pretrained / checkpoint
            pretrained_model_index: str = "ensemble",
            load_checkpoint: str = "",
            # Logging / TensorBoard
            train_log_batch_interval: int = 5,
            log_lr_each_epoch: bool = False,
            tb_log_dir: str | None = None,
            disable_tensorboard: bool = False,
            per_file_logging: bool = False,
            tolerant_missing_metadata: bool = False,
            gpus: str = "",
            debug_devices: bool = False,
            # Multi-GPU & memory optimization
            use_amp: bool = False,
            grad_accumulation: int = 1,
            memory_efficient: bool = False,
            pre_train_task_name: str = AppConstants.TASK_GET_WEIGHTS,
            train_task_name: str = AppConstants.TASK_TRAIN,
            submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
            validate_task_name: str = AppConstants.TASK_VALIDATION,
            test_task_name: str = "test",
            exclude_vars=None,
            gmic_parameters: dict | None = None,
            loss: str = "cross_entropy",
            optimizer: dict | None = None,
            task_names: dict | None = None,
            # ---- Federated method selection (single switch across 6 regimes) ----
            method: str = "fedavg",
            fedprox_mu: float = 0.01,
            ditto_lambda: float = 0.1,
            lambda_global: float = 1.0,
            lambda_local: float = 0.1,
            lambda_fusion: float | None = None,   # ties to lambda_local if None
            use_fedbn: bool = False,
            local_epochs: int | None = None,      # alias/override for `epochs`
            pretrained_weights_path: str = "",    # falls back to model_path if empty
            pos_weight: float = 3.0,              # malignant-class weight for GMIC BCE
            # ---- Imbalance / overfitting levers (all default to current/off behavior) ----
            use_balanced_sampler: bool = False,   # view-level class-balanced batch sampling
            focal_gamma: float = 2.0,             # used when loss=="focal"
            focal_alpha: float = 0.25,            # used when loss=="focal"
            lr_scheduler: str | None = None,      # None/"plateau"(default)/"cosine"/"none"
            scheduler_patience: int = 2,          # ReduceLROnPlateau patience
            min_lr: float = 0.0,                  # scheduler floor
            cosine_t_max: int = 0,                # cosine period (0 -> epochs)
            decision_threshold: float = 0.5,      # operating point for acc/sens/spec/precision
            eval_threshold_sweep=None,            # sweep thresholds (default [.3,.4,.5,.6,.7]); on
            freeze_backbone_epochs: int = 0,      # keep backbones frozen for first N epochs
            optimizer_name: str = "adam",         # "adam" (default, unchanged) or "adamw"
        ):
            """GMIC Federated Executor (feature parity with local trainer)."""
            super().__init__()

            # Local logger (safe even if stdout is broken)
            self._logger = logging.getLogger(self.__class__.__name__)

            # Allow legacy alias 'out_dir' via kwargs fallback (NVFLARE may pass unexpected)
            # Results directory (allow alias out_dir)
            chosen_results = out_dir if out_dir else results_dir
            self.results_dir = chosen_results or "/workspace/gmic_results_centralized"
            self.epochs = epochs
            self.lr_heads = lr_heads
            self.lr_backbone = lr_backbone
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.data_path = data_path
            self.image_path = image_path
            self.model_path = model_path
            self.device = device
            self.input_format = input_format
            self.enable_preprocessing = enable_preprocessing
            self.cache_validation = cache_validation
            self.force_preprocessing = force_preprocessing
            self.use_predefined_splits = use_predefined_splits
            self.val_split = val_split
            self.test_split = test_split
            self.random_seed = random_seed
            self.output_dir = output_dir
            self.log_file = log_file
            self.file_integrity_check = file_integrity_check
            self.fail_on_integrity_error = fail_on_integrity_error
            self.num_processes = num_processes
            self.freeze_all_backbones = freeze_all_backbones
            self.unfreeze_global_last = unfreeze_global_last
            self.unfreeze_local_last = unfreeze_local_last
            # Allow legacy percent_t alias if provided
            self.percent_t_key = percent_t if percent_t is not None else percent_t_key
            self.K = K
            self.cam_h = cam_h
            self.cam_w = cam_w
            self.crop_h = crop_h
            self.crop_w = crop_w
            self.post_dim = post_dim
            self.num_classes = num_classes
            self.use_v1_global = use_v1_global
            self.lambda_l1 = lambda_l1
            self.pretrained_model_index = pretrained_model_index
            self.load_checkpoint = load_checkpoint
            self.train_log_batch_interval = train_log_batch_interval
            self.log_lr_each_epoch = log_lr_each_epoch
            self.tb_log_dir = tb_log_dir
            self.disable_tensorboard = disable_tensorboard
            self.per_file_logging = per_file_logging
            self.tolerant_missing_metadata = tolerant_missing_metadata
            self.gpus = gpus
            self.debug_devices = debug_devices
            self.use_amp = use_amp
            self.grad_accumulation = max(1, grad_accumulation)
            self.memory_efficient = memory_efficient
            self._scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if torch.cuda.is_available() else None
            self.pre_train_task_name = pre_train_task_name
            self.train_task_name = train_task_name
            self.submit_model_task_name = submit_model_task_name
            self.validate_task_name = validate_task_name
            self.test_task_name = test_task_name
            self.exclude_vars = exclude_vars
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.data_loader = None
            self.persistence_manager = None
            self.best_metrics = {"val_auc": 0.0, "test_auc": 0.0}
            self.training_history = []
            self.validation_history = []
            self.test_history = []
            self.gmic_parameters_cfg = gmic_parameters or {}
            self.loss_name = loss
            self.optimizer_cfg = optimizer or {"name": "adam", "weight_decay": self.weight_decay}
            self.task_names = task_names or {}
            self.pre_train_task_name = self.task_names.get("pre_train", self.pre_train_task_name)
            self.train_task_name = self.task_names.get("train", self.train_task_name)
            self.validate_task_name = self.task_names.get("validate", self.validate_task_name)
            self.test_task_name = self.task_names.get("test", self.test_task_name)
            self.submit_model_task_name = self.task_names.get("submit_model", self.submit_model_task_name)
            self.patience = patience

            # ---- Federated method configuration ----
            self.method = (method or "fedavg").lower()
            if self.method not in VALID_METHODS:
                self._logger.warning(
                    "[EXEC][WARN] Unknown method '%s'; falling back to 'fedavg'. Valid: %s",
                    self.method, VALID_METHODS,
                )
                self.method = "fedavg"
            self.fedprox_mu = float(fedprox_mu)
            self.ditto_lambda = float(ditto_lambda)
            self.lambda_global = float(lambda_global)
            self.lambda_local = float(lambda_local)
            self.lambda_fusion = float(lambda_fusion) if lambda_fusion is not None else float(lambda_local)
            self.lam_dict = {
                "global": self.lambda_global,
                "local": self.lambda_local,
                "fusion": self.lambda_fusion,
            }
            self.use_fedbn = bool(use_fedbn)
            self.pos_weight = float(pos_weight)
            # ---- Imbalance / overfitting levers ----
            self.use_balanced_sampler = bool(use_balanced_sampler)
            self.focal_gamma = float(focal_gamma)
            self.focal_alpha = float(focal_alpha)
            self.lr_scheduler_name = lr_scheduler
            self.scheduler_patience = int(scheduler_patience)
            self.min_lr = float(min_lr)
            self.cosine_t_max = int(cosine_t_max)
            self.decision_threshold = float(decision_threshold)
            self.eval_threshold_sweep = (list(eval_threshold_sweep)
                                         if eval_threshold_sweep else [0.3, 0.4, 0.5, 0.6, 0.7])
            self.freeze_backbone_epochs = int(freeze_backbone_epochs)
            self.optimizer_name = (optimizer_name or "adam").lower()
            # Guardrail: balanced sampler + a large pos_weight double-counts the imbalance
            # correction (sampler evens the batch mix AND the loss up-weights positives).
            if self.use_balanced_sampler and self.pos_weight > 2.0:
                self._logger.warning(
                    "[EXEC][WARN] use_balanced_sampler=True AND pos_weight=%.2f (>2.0): these "
                    "STACK the imbalance correction and may over-correct toward positives. "
                    "Consider pos_weight~1.0 when the balanced sampler is on.", self.pos_weight
                )
            self.pretrained_weights_path = pretrained_weights_path or ""
            if local_epochs is not None:
                self.epochs = int(local_epochs)
            # Ditto personal model `v` and its optimizer persist across rounds.
            self._v_model = None
            self._v_optimizer = None
            self._global_ref = None
            self._prox_ref = None
            self._prox_lambda = None
            self._opt_args = None
            self._pretrained_report = None

            # Use output_dir as the canonical results_dir for model/metrics artifacts
            self.results_dir = self.output_dir


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
        # If this executor instance was recreated for a later round/end_run, recover persisted best metrics
        try:
            self._recover_persisted_best_metrics()
        except Exception as e:
            self._logger.warning("[EXEC][WARN] Failed to recover persisted best metrics: %s", e, exc_info=True)

        # 1. Effective data paths
        set_seed(self.random_seed)
        processed_pkl = os.path.join(self.output_dir, "processed_exam_list.pkl")
        cropped_dir = os.path.join(self.output_dir, "cropped_images")
        
        # **NEW: Force preprocessing if flag is set**
        # Must clear the FULL preprocessing state so the loader rebuilds from scratch.
        # In particular cropped_exam_list.pkl (the Stage-1 exam list) must go too: if it
        # survives while cropped_images/ is deleted, the loader sees "Stage 1 done" and
        # resumes at "Stage2 only" -> center extraction then fails on the missing PNGs and
        # silently trains WITHOUT best-centers (corrupting the local-module ROI patches).
        if self.force_preprocessing:
            self._logger.warning("[EXEC] force_preprocessing=True: Clearing cached preprocessed data")
            import shutil
            stale_files = [
                processed_pkl,
                os.path.join(self.output_dir, "cropped_exam_list.pkl"),         # Stage-1 list
                os.path.join(self.output_dir, "preprocessing_cache_info.json"),  # cache validator
                os.path.join(self.output_dir, "preprocessing_progress.json"),    # resume manifest
            ]
            for f in stale_files:
                if os.path.isfile(f):
                    os.remove(f)
                    self._logger.info("[EXEC] Deleted: %s", f)
            if os.path.isdir(cropped_dir):
                shutil.rmtree(cropped_dir, ignore_errors=True)
                self._logger.info("[EXEC] Deleted: %s", cropped_dir)
        
        # Now check for cache (will be empty if force_preprocessing=True)
        if os.path.isfile(processed_pkl) and os.path.isdir(cropped_dir):
            effective_data_path = processed_pkl
            effective_image_path = cropped_dir
            effective_input_format = "pkl"
            effective_enable_preprocessing = False
            self._logger.info("[EXEC] Using cached processed data: %s", processed_pkl)
        else:
            effective_data_path = self.data_path
            effective_image_path = self.image_path
            effective_input_format = self.input_format
            effective_enable_preprocessing = self.enable_preprocessing
            self._logger.info(
                "[EXEC] Using source data: %s (enable_preprocessing=%s)",
                self.data_path, self.enable_preprocessing
            )

        # 2. Data loader
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
            cache_validation=self.cache_validation,
            log_file=self.log_file,
            per_file_logging=self.per_file_logging,
            tolerant_missing_metadata=self.tolerant_missing_metadata,
            file_integrity_check=self.file_integrity_check,
            fail_on_integrity_error=self.fail_on_integrity_error,
            use_balanced_sampler=self.use_balanced_sampler,
        )
        self.data_loader.print_summary()

        # 3. Iterations per split
        split_info = self.data_loader.get_split_info()
        self._n_train_iterations = max(1, math.ceil(split_info["train_size"] / max(self.batch_size, 1)))
        self._n_val_iterations = max(1, math.ceil(split_info["val_size"] / max(self.batch_size, 1)))
        self._n_test_iterations = max(1, math.ceil(split_info["test_size"] / max(self.batch_size, 1)))

        # 4. GMIC parameters
        def _to_tuple(x):
            return tuple(x) if isinstance(x, (list,)) else x

        gmic_defaults = {
            "device_type": "gpu" if "cuda" in str(self.device) else "cpu",
            "gpu_number": 0,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "image_path": effective_image_path,
            "cam_size": (self.cam_h, self.cam_w),
            "K": self.K,
            "crop_shape": (self.crop_h, self.crop_w),
            "post_processing_dim": self.post_dim,
            "num_classes": self.num_classes,
            "use_v1_global": self.use_v1_global,
            "percent_t": PERCENT_T_DICT.get(self.percent_t_key, PERCENT_T_DICT["1"]),
            "lambda_l1": self.lambda_l1,
        }
        overrides = dict(getattr(self, "gmic_parameters_cfg", {}) or {})
        for k in ["max_crop_noise", "cam_size", "crop_shape"]:
            if k in overrides:
                overrides[k] = _to_tuple(overrides[k])
        pt = overrides.get("percent_t")
        if isinstance(pt, str) and pt.startswith("auto:"):
            key = pt.split(":", 1)[1]
            overrides["percent_t"] = PERCENT_T_DICT.get(key, PERCENT_T_DICT["1"])
        self.gmic_parameters = {**gmic_defaults, **overrides}
        # C1: single source of truth for percent_t. The model is built from
        # percent_t_key (-> PERCENT_T_DICT). Fail loud if a gmic_parameters.percent_t
        # override disagrees, so a config sweep can't silently use a different threshold.
        _pt_key_val = PERCENT_T_DICT.get(self.percent_t_key, PERCENT_T_DICT["1"])
        _pt_override = overrides.get("percent_t")
        if isinstance(_pt_override, (int, float)) and abs(float(_pt_override) - float(_pt_key_val)) > 1e-9:
            raise ValueError(
                f"percent_t mismatch: gmic_parameters.percent_t={_pt_override} but "
                f"percent_t_key='{self.percent_t_key}' -> {_pt_key_val}. Use ONE source "
                f"(prefer percent_t_key; remove percent_t from gmic_parameters)."
            )

        # 5. Model build
        class _Args: pass
        a = _Args()
        a.device_type = gmic_defaults["device_type"]
        a.gpu_number = 0
        a.image_path = self.gmic_parameters.get("image_path")
        a.num_classes = self.gmic_parameters.get("num_classes", 2)
        a.cam_h, a.cam_w = self.gmic_parameters.get("cam_size", (self.cam_h, self.cam_w))
        a.K = self.gmic_parameters.get("K", self.K)
        a.post_dim = self.gmic_parameters.get("post_processing_dim", self.post_dim)
        a.percent_t = self.percent_t_key
        a.lambda_l1 = self.gmic_parameters.get("lambda_l1", self.lambda_l1)
        a.crop_h, a.crop_w = self.gmic_parameters.get("crop_shape", (self.crop_h, self.crop_w))
        a.use_v1_global = self.gmic_parameters.get("use_v1_global", self.use_v1_global)

        self.model = build_gmic_from_args(a)
        # Helper: underlying model reference (for DP awareness)
        self._underlying = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if "cuda" in str(self.device):
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                self._logger.warning(
                    "[EXEC] CUDA '%s' requested but no driver found — falling back to CPU", self.device
                )
                self.device = torch.device("cpu")
        self.model.to(self.device)

        """
        # Optional pretrained load (if no explicit checkpoint provided)
        if (not self.load_checkpoint) and getattr(self, 'pretrained_model_index', None):
            try:
                load_pretrained_if_requested(self.model, type('obj', (), {
                    'pretrained_model_index': self.pretrained_model_index,
                    'model_path': self.model_path,
                    'device': self.device,
                }))
                self._logger.info("[EXEC] Loaded pretrained model index %s", self.pretrained_model_index)
            except Exception as e:
                self._logger.warning("[EXEC][WARN] pretrained load failed: %s", e, exc_info=True)

        if self.load_checkpoint and os.path.isfile(self.load_checkpoint):
            try:
                state = torch.load(self.load_checkpoint, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                self._logger.info("[EXEC] Loaded checkpoint: %s", self.load_checkpoint)
            except Exception as e:
                self._logger.warning("[EXEC][WARN] Failed to load checkpoint %s: %s", self.load_checkpoint, e, exc_info=True)
        """

        # 5b. Always initialize from the pretrained GMIC checkpoint (decision: fine-tune,
        # not train-from-scratch). The released checkpoint keys match this model 1:1
        # except two benign items (missing '_device_ref' buffer, unexpected
        # 'shared_rep_filter.weight'), so a clean load reports matched=258/259.
        # This makes "init from pretrained" robust and self-contained — independent of
        # whether the server persistor seeds round-0 with the same checkpoint — and is
        # what makes the round-0 baseline meaningful and `method=local` start pretrained.
        ckpt = self.pretrained_weights_path or self.model_path
        self._pretrained_report = None
        if ckpt and os.path.isfile(ckpt):
            # R4: raise_on_partial=True and NO broad except -> a partial backbone load
            # (or a corrupt checkpoint) aborts initialize() loudly. A half-loaded model
            # would make the cold-start baseline silently meaningless.
            self._pretrained_report = tolerant_load_pretrained(
                self._underlying, ckpt, device=self.device,
                log_fn=lambda m: self._logger.info(m), raise_on_partial=True,
            )
            self._logger.info("[EXEC] Pretrained init from %s: %s", ckpt, self._pretrained_report)
        else:
            self._logger.warning(
                "[EXEC][WARN] pretrained checkpoint not found at '%s' — model starts from "
                "random init and the round-0 baseline will NOT be meaningful. "
                "Set pretrained_weights_path (or model_path) on the restricted machine.", ckpt
            )

        # 6. Freezing
        class _FreezeArgs: pass
        fa = _FreezeArgs()
        fa.freeze_all_backbones = self.freeze_all_backbones
        fa.unfreeze_global_last = self.unfreeze_global_last
        fa.unfreeze_local_last = self.unfreeze_local_last
        apply_freezing_plan(self.model, fa)

        # 7. Loss & optimizer. resolve_criterion validates the loss name and fails loudly on
        # an unknown value (same pattern as the `method` validation). Returns None for the GMIC
        # path (loss computed in _compute_task_loss); criterion is unused there.
        self.criterion = resolve_criterion(getattr(self, "loss_name", "gmic"))

        class _OptArgs: pass
        oa = _OptArgs()
        oa.lr_heads = self.lr_heads
        oa.lr_backbone = self.lr_backbone
        oa.weight_decay = self.weight_decay
        oa.patience = self.patience
        oa.optimizer_name = self.optimizer_name
        # Scheduler selection (None -> historical ReduceLROnPlateau; preserves current behavior)
        oa.lr_scheduler = self.lr_scheduler_name
        oa.scheduler_patience = self.scheduler_patience
        oa.min_lr = self.min_lr
        oa.cosine_t_max = self.cosine_t_max
        oa.epochs = self.epochs
        self.optimizer, self.scheduler, self.early_stopper = configure_optimizers(self.model, oa)
        # Stash optimizer args so the Ditto personal model `v` can build its own optimizer.
        self._opt_args = oa

        # 8. TensorBoard
        self.tb_writer = None
        if self.tb_log_dir and not self.disable_tensorboard:
            try:
                self.tb_writer = create_tb_writer(self.tb_log_dir, enable=True)
                if self.tb_writer:
                    self._logger.info("[EXEC][tensorboard] logging to %s", self.tb_log_dir)
            except Exception as e:
                self._logger.warning("[EXEC][tensorboard][WARN] %s", e, exc_info=True)

        # 9. Multi-GPU
        gpu_ids = []
        if self.gpus:
            try:
                gpu_ids = [int(x) for x in self.gpus.split(',') if x.strip()]
            except ValueError:
                self._logger.warning("[EXEC][WARN] Invalid gpus string: %s", self.gpus)

        if torch.cuda.is_available() and len(gpu_ids) > 0:
            # Ensure primary device matches first GPU id
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
            self.model.to(self.device)
            if len(gpu_ids) > 1:
                self._logger.info("[EXEC] Enabling DataParallel on GPUs: %s", gpu_ids)
                self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

        # 10. Optional PKL snapshot
        if self.data_loader.input_format == "csv":
            pkl_output_path = os.path.join(os.path.dirname(self.model_path), "converted_data.pkl")
            try:
                self.data_loader.save_pkl_format(pkl_output_path)
            except Exception as e:
                self._logger.warning("[WARN] Failed to save PKL snapshot: %r", e, exc_info=True)

        self._logger.info(
            "Initialized GMIC executor with %s data: %s",
            self.data_loader.input_format.upper(), split_info
        )


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:

        try:
            # --- round headers ---
            current_round = int(shareable.get_header(AppConstants.CURRENT_ROUND, 0))
            total_rounds = int(shareable.get_header(AppConstants.NUM_ROUNDS, 1))
            self.log_info(fl_ctx, f"Starting round {current_round}/{total_rounds} - Train+Validate+Test")

            # Per-run config echo so sweep results can be matched to the intended config from
            # the log. Show only the lambda(s) that are ACTIVE for the selected method.
            if self.method == "ditto_modulewise":
                lam_desc = (f"lambda_global={self.lambda_global} lambda_local={self.lambda_local} "
                            f"lambda_fusion={self.lambda_fusion}")
            elif self.method == "ditto":
                lam_desc = f"ditto_lambda={self.ditto_lambda}"
            elif self.method == "fedprox":
                lam_desc = f"fedprox_mu={self.fedprox_mu}"
            else:
                lam_desc = "(no lambda for this method)"
            self.log_info(
                fl_ctx,
                f"[run-config] method={self.method} use_fedbn={self.use_fedbn} {lam_desc} "
                f"loss={self.loss_name} pos_weight={self.pos_weight} epochs={self.epochs} "
                f"lr_heads={self.lr_heads} balanced_sampler={self.use_balanced_sampler} "
                f"lr_scheduler={self.lr_scheduler_name} decision_threshold={self.decision_threshold} "
                f"eval_threshold_sweep={self.eval_threshold_sweep} "
                f"optimizer={self.optimizer_name} weight_decay={self.weight_decay} "
                f"freeze_backbone_epochs={self.freeze_backbone_epochs}"
                + (f" focal(gamma={self.focal_gamma},alpha={self.focal_alpha})" if self.loss_name == "focal" else "")
            )

            # --- receive global model (support both FLModel and legacy raw weights) ---
            incoming = None
            fl_model_in = FLModelUtils.from_shareable(shareable)
            if fl_model_in and fl_model_in.params:
                incoming = self._ensure_torch_state(fl_model_in.params, fl_ctx)
            elif AppConstants.MODEL_WEIGHTS in shareable:
                incoming = self._ensure_torch_state(shareable[AppConstants.MODEL_WEIGHTS], fl_ctx)

            # --- round-0 pretrained baseline (BEFORE consuming the global) ---
            # initialize() pretrained-loads the model, so right now `self.model` holds the
            # cold-start weights. Dump its val/test predictions once so every method is
            # comparable to the "no local adaptation at round 0" reference row.
            if current_round == 0:
                try:
                    self._dump_baseline(fl_ctx)
                except Exception as e:
                    self.log_warning(fl_ctx, f"[baseline] dump failed: {e}")

            # --- method-aware consumption of the global model ---
            # local: skip load | fedbn/use_fedbn: skip BN keys | else: full load.
            # Also captures the frozen global ref (FedProx/Ditto) and lazily inits Ditto v.
            self._consume_global(incoming, fl_ctx)

            # === NUCLEAR OPTION: FORCE RANDOM INIT FOR THIS RUN ===
            # self._reset_model_weights()
            # self.log_info(fl_ctx, "[RANDOM_INIT] Forced re-randomization of all model weights at start of execute().")

            # --- TRAIN (with early stopping) ---
            self.log_info(fl_ctx, "Phase 1: Training (early stopping enabled)...")
            train_metrics = self._local_train(fl_ctx, abort_signal, shareable)

            # C2: the aggregation contribution is THIS round's FINAL post-epoch local weights
            # (textbook FedAvg/FedProx/FedBN; for Ditto, the global-tracked w). Capture them
            # BEFORE loading any best-val checkpoint, so the sent w is never a stale best-state.
            final_state = {k: v.detach().cpu().clone() for k, v in self._underlying.state_dict().items()}

            # Deployed/evaluated model = best-val OF THIS ROUND. Load it into the global-tracked
            # model for eval/dumps only (non-Ditto; Ditto deploys v). The SENT w stays = final_state.
            if (self.method not in ("ditto", "ditto_modulewise")
                    and getattr(self, 'early_stopper', None)
                    and getattr(self.early_stopper, 'best_state', None)):
                try:
                    self._underlying.load_state_dict(self.early_stopper.best_state, strict=False)
                    self.log_info(fl_ctx, f"Deployed/eval = best-val-of-round (val_auc={self.early_stopper.best:.4f}); sent w = final-epoch")
                    self._save_best_model(fl_ctx, {"auc": float(self.early_stopper.best)}, shareable, model_type="best_val_round")
                except Exception as e:
                    self.log_warning(fl_ctx, f"Failed to load best-val weights for eval: {e}")

            # --- Ditto personal pass: train v with task_loss(v) + proximal(v, w_ref) ---
            # (v persists across rounds; only the global-tracked model w is sent/aggregated.)
            if self.method in ("ditto", "ditto_modulewise"):
                self._train_personal_model(fl_ctx, abort_signal)

            # Deployed/evaluated model: personal v for Ditto-family, else the global-tracked model.
            deployed = self._v_model if self.method in ("ditto", "ditto_modulewise") else self.model

            num_examples = int(train_metrics.get("train_samples", self.batch_size))
            if num_examples <= 0:
                # ensure a positive weight so FedAvg doesn’t drop this update
                num_examples = max(1, self.batch_size)

            if not abort_signal.triggered:
                self._save_local_model(fl_ctx, shareable, model=deployed)

            # --- VAL (on the deployed model: v for Ditto-family, else w) ---
            self.log_info(fl_ctx, "Phase 2: Validation...")
            val_core = self._evaluate_model(fl_ctx, split="val", model=deployed)
            # Prefix validation metrics so server selector finds 'val_auc'
            val_metrics = {f"val_{k}": v for k, v in val_core.items() if k in ("auc", "accuracy", "loss", "samples")}

            # Per-site validation metric for the deployed model (offline ranking / fairness).
            self.log_info(
                fl_ctx,
                f"[per-site-metric] site={fl_ctx.get_identity_name()} round={current_round} "
                f"method={self.method} deployed_val_auc={val_core.get('auc', float('nan')):.4f}"
            )

            # Update best tracking
            if val_core.get('auc', -1) > self.best_metrics.get('val_auc', -1):
                self.best_metrics['val_auc'] = val_core['auc']
                try:
                    self._save_best_model(fl_ctx, {"auc": val_core['auc']}, shareable, model_type="best_val_overall", model=deployed)
                except Exception:
                    pass

            # --- TEST (final round only) ---
            if current_round == (total_rounds - 1):
                self.log_info(fl_ctx, "Phase 3: Testing...")
                test_core = self._evaluate_model(fl_ctx, split="test", model=deployed)
                test_metrics = {f"test_{k}": v for k, v in test_core.items() if k in ("auc", "accuracy", "loss", "samples")}
                if test_core.get('auc', -1) > self.best_metrics.get('test_auc', -1):
                    self.best_metrics['test_auc'] = test_core['auc']
                    try:
                        self._save_best_model(fl_ctx, test_core, shareable, model_type="best_test_overall", model=deployed)
                    except Exception:
                        pass
                # End-of-run prediction dump for offline AUC / DeLong CIs / cross-site sigma(AUC).
                for _split in ("val", "test"):
                    try:
                        self._dump_predictions(fl_ctx, split=_split, model=deployed,
                                               round_idx=current_round, method_tag=self.method)
                    except Exception as e:
                        self.log_warning(fl_ctx, f"[predictions] dump ({_split}) failed: {e}")
            else:
                self.log_info(fl_ctx, "Phase 3: Skipping test (not final round)")
                test_metrics = {}

            # --- package update for FedAvg (DXO/FLModel) ---
            # C2: send the FINAL-epoch local weights (captured pre-reload), NOT the best-val
            # checkpoint -- standard FedAvg/FedProx/FedBN expect the current round's local update.
            updated_weights = {k: v.detach().cpu().numpy() for k, v in final_state.items()}

            combined_metrics = {**train_metrics, **val_metrics, **test_metrics, "round": int(current_round)}
            if 'val_auc' not in combined_metrics and 'val_auc' in val_metrics:
                combined_metrics['val_auc'] = val_metrics['val_auc']

            # Safety filter: only include numeric values
            numeric_metrics = {}
            for key, value in combined_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_metrics[key] = float(value) if not isinstance(value, int) else int(value)
                else:
                    self.log_warning(fl_ctx, f"Skipping non-numeric metric: {key} = {value}")

            # Attach validation metric in meta so IntimeModelSelector can read it
            val_auc_for_meta = numeric_metrics.get('val_auc') or numeric_metrics.get('best_val_auc')
            model_meta = {
                MetaKey.NUM_STEPS_CURRENT_ROUND: num_examples,
                "num_examples": num_examples,
            }
            if val_auc_for_meta is not None and not math.isnan(float(val_auc_for_meta)):
                model_meta['val_auc'] = float(val_auc_for_meta)
            else:
                self.log_warning(fl_ctx, "[metrics-debug] val_auc missing/NaN (degenerate split?); selector may warn")

            fl_model_out = FLModel(
                params=updated_weights,
                params_type=ParamsType.FULL,
                metrics=numeric_metrics,
                meta=model_meta,
            )

            self.log_info(fl_ctx, (
                f"[metrics-debug] meta keys: {list(model_meta.keys())} val_auc_meta={model_meta.get('val_auc')} "
                f"numeric_key_count={len(numeric_metrics)} has_val_auc={'val_auc' in numeric_metrics}"
            ))

            # Debug print of outgoing metrics formatting
            try:
                preview_items = []
                for i, k in enumerate(sorted(numeric_metrics.keys())):
                    if i >= 25:
                        preview_items.append(f"...+{len(numeric_metrics) - 25} more")
                        break
                    v = numeric_metrics[k]
                    if isinstance(v, float):
                        preview_items.append(f"{k}={v:.6g}")
                    else:
                        preview_items.append(f"{k}={v}")
                self.log_info(fl_ctx, f"[metrics-debug] outgoing metrics: {' | '.join(preview_items)}")
            except Exception as e:
                self.log_warning(fl_ctx, f"[metrics-debug] failed to format metrics preview: {e}")

            reply = FLModelUtils.to_shareable(fl_model_out)
            reply.set_return_code(ReturnCode.OK)

            self.log_info(
                fl_ctx,
                f"Round {current_round} done - "
                f"train_loss={train_metrics.get('train_loss', 'N/A')} "
                f"val_acc={val_metrics.get('val_accuracy', 'N/A')} "
                + (f"test_acc={test_metrics.get('test_accuracy', 'N/A')}" if test_metrics else "")
            )
            return reply

        except Exception as e:
            self.log_exception(fl_ctx, f"Error in all-in-one task: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


    def _ensure_torch_state(self, params: dict, fl_ctx=None) -> dict:
        """Convert incoming FLModel.params (possibly numpy) into a proper torch state_dict."""
        ref_model = self._underlying if hasattr(self, '_underlying') else self.model
        ref = ref_model.state_dict()
        device = next(ref_model.parameters()).device
        out = {}

        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                t = v.to(device)
            elif isinstance(v, np.ndarray):
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.from_numpy(v).to(device=device, dtype=dtype)
            elif np.isscalar(v):
                dtype = ref[k].dtype if k in ref else torch.float32
                t = torch.tensor(v, device=device, dtype=dtype)
            else:
                if fl_ctx:
                    self.log_warning(fl_ctx, f"Skipping param {k} of unsupported type {type(v)}")
                continue

            if "num_batches_tracked" in k:
                t = t.to(torch.long)
            out[k] = t
        return out


    def _get_model_weights(self) -> Shareable:
        """Get current GMIC model weights"""
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_train_iterations}
        )
        return outgoing_dxo.to_shareable()


    def _compute_task_loss(self, outputs, targets):
        """Task loss on the malignant head (native GMIC deep-supervised BCE + L1).

        `loss=gmic`/`gmic_bce`: BCEWithLogits(fusion)+BCE(global)+BCE(local)+lambda_l1*L1,
            malignant channel, shared pos_weight.
        `loss=focal`: same 3-head + L1 structure with FOCAL BCE (focal_gamma/focal_alpha).
        `loss=cross_entropy`/`ce`: legacy CrossEntropyLoss on the fusion logits.
        """
        name = (getattr(self, "loss_name", "gmic") or "gmic").lower()
        if name == "focal":
            return gmic_focal_loss(
                outputs, targets, lambda_l1=self.lambda_l1,
                gamma=self.focal_gamma, alpha=self.focal_alpha,
            )
        if name in ("gmic", "gmic_bce"):
            return gmic_malignant_loss(
                outputs, targets, lambda_l1=self.lambda_l1, pos_weight=self.pos_weight
            )
        fusion_logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        return self.criterion(fusion_logits, targets)


    def _apply_backbone_freeze(self, fl_ctx, frozen: bool):
        """Toggle requires_grad on the global+local backbones (freeze_backbone_epochs lever).

        Idempotent; logs only on a state change. Heads stay trainable throughout.
        """
        if getattr(self, "_backbone_frozen", None) == frozen:
            return
        mdl = self._underlying
        for attr in ("global_backbone", "local_backbone"):
            bb = getattr(mdl, attr, None)
            if bb is not None and hasattr(bb, "parameters"):
                for p in bb.parameters():
                    p.requires_grad = not frozen
        self._backbone_frozen = frozen
        self.log_info(fl_ctx, f"[freeze] backbones {'FROZEN' if frozen else 'UNFROZEN'} (heads always trainable)")

    def _scheduler_step(self, val_auc):
        """Step the LR scheduler, tolerating plateau (needs metric) vs cosine/step (no metric)."""
        sched = getattr(self, "scheduler", None)
        if sched is None:
            return
        try:
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_auc)
            else:
                sched.step()
        except Exception as e:
            self._logger.warning("[scheduler] step failed: %s", e)


    def _local_train(self, fl_ctx, abort_signal, shareable: Shareable):
        """Local training with per-epoch validation, scheduler stepping, and early stopping.

        Returns final aggregate training metrics plus best validation AUC observed.
        """
        has_val = len(self.data_loader.get_data_for_split('val')) > 0
        self.model.train()
        # C2: reset the early stopper each round so best-val selection is within-round
        # (best-state never crosses rounds, for either the sent or the deployed model).
        if getattr(self, 'early_stopper', None):
            self.early_stopper.reset()
        total_samples = 0
        epochs_run = 0
        round_best_val = -float('inf')
        last_epoch_loss = 0.0
        last_epoch_auc = 0.0
        last_epoch_acc = 0.0

        for epoch in range(self.epochs):
            if abort_signal.triggered:
                break
            epochs_run = epoch + 1
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_preds = []
            epoch_targets = []
            self.log_info(fl_ctx, f"Training epoch {epoch + 1}/{self.epochs}")

            # Backbone freeze lever: keep backbones frozen for the first N epochs, then unfreeze
            # once (anti-overfitting warmup). No-op when freeze_backbone_epochs==0 (default).
            if self.freeze_backbone_epochs > 0:
                self._apply_backbone_freeze(fl_ctx, frozen=(epoch < self.freeze_backbone_epochs))

            for batch_idx, (inputs, targets, metadata) in enumerate(self.data_loader.get_batch_iterator('train')):
                if abort_signal.triggered:
                    break
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                accumulate = self.grad_accumulation
                if batch_idx % accumulate == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                amp_device = "cuda" if "cuda" in str(self.device) else "cpu"
                with torch.autocast(device_type=amp_device, enabled=self.use_amp):
                    outputs = self.model(inputs)
                # Loss is computed OUTSIDE autocast, in fp32: F.binary_cross_entropy
                # (global/local heads, on already-sigmoided probabilities) is unsafe under
                # autocast and raises on GPU. The canonical AMP pattern autocasts only the
                # forward; BCEWithLogits/backward run in fp32.
                loss = self._compute_task_loss(outputs, targets) / accumulate
                # FedProx: add (mu/2)||w - w_global||^2 to the w-pass loss.
                # (No-op for fedavg/fedbn/local/ditto, where _prox_ref is None.)
                if getattr(self, "_prox_ref", None) is not None:
                    loss = loss + proximal_penalty(
                        self._underlying, self._prox_ref, self._prox_lambda
                    ) / accumulate

                if self.use_amp and self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    if (batch_idx + 1) % accumulate == 0:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                else:
                    loss.backward()
                    if (batch_idx + 1) % accumulate == 0:
                        self.optimizer.step()

                effective_loss = loss.item() * accumulate
                bsz = inputs.size(0)
                total_samples += bsz
                epoch_loss += effective_loss
                epoch_batches += 1

                if self.debug_devices and batch_idx == 0:
                    try:
                        inp_dev = first_batch_input_device_str(inputs)
                        summary = summarize_parameter_devices(self.model)
                        self.log_info(fl_ctx, f"[devices] first_batch_input_devices={inp_dev}")
                        self.log_info(fl_ctx, f"[devices] param_devices={summary['unique_devices']} counts={summary['device_counts']}")
                        for ex in summary['examples']:
                            self.log_info(fl_ctx, f"[devices] example {ex}")
                    except Exception as e:
                        self.log_warning(fl_ctx, f"[devices] diagnostics failed: {e}")

                # Malignant probability for batch AUC = sigmoid(fusion_logit)[:, 1]
                pos_probs = malignant_score(outputs).detach().cpu().numpy().reshape(-1)
                pos_targets = targets.detach().cpu().numpy().reshape(-1)

                epoch_preds.extend(pos_probs)
                epoch_targets.extend(pos_targets)

                if self.train_log_batch_interval and (batch_idx % self.train_log_batch_interval == 0):
                    self.log_info(fl_ctx, f"Epoch {epoch + 1} Batch {batch_idx} Loss {effective_loss:.4f}")

                if self.memory_efficient and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if epoch_batches == 0:
                break

            ep_preds_np = np.array(epoch_preds)
            ep_targets_np = np.array(epoch_targets)

            try:
                ep_auc = roc_auc_score(ep_targets_np, ep_preds_np) if len(np.unique(ep_targets_np)) > 1 else 0.0
            except Exception as e:
                self.log_warning(fl_ctx, f"roc_auc_score failed at epoch {epoch + 1}: {e}")
                ep_auc = 0.0

            ep_pred_labels = (ep_preds_np > 0.5).astype(int)
            ep_acc = 100 * accuracy_score(ep_targets_np, ep_pred_labels) if len(ep_targets_np) else 0.0
            ep_loss_avg = epoch_loss / max(epoch_batches, 1)

            if self.log_lr_each_epoch:
                lrs = [pg['lr'] for pg in self.optimizer.param_groups]
                self.log_info(fl_ctx, f"Epoch {epoch + 1} LRs {','.join(f'{lr:.2e}' for lr in lrs)}")

            last_epoch_loss = ep_loss_avg
            last_epoch_auc = ep_auc
            last_epoch_acc = ep_acc

            val_auc = None
            if has_val:
                val_metrics = self._evaluate_model(fl_ctx, split='val')
                val_auc = val_metrics['auc']
                # plateau scheduler steps on val_auc; cosine (and others) step without a metric;
                # None scheduler -> no-op. Handle all three.
                self._scheduler_step(val_auc)
                if self.early_stopper.step(val_auc, self.model):
                    round_best_val = val_auc
                if self.early_stopper.should_stop():
                    self.log_info(fl_ctx, f"Early stopping triggered at epoch {epoch + 1} (best_val_auc={self.early_stopper.best:.4f})")
                    break
                self.model.train()

            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': float(ep_loss_avg),
                'train_auc': float(ep_auc),
                'train_accuracy': float(ep_acc),
                'val_auc': float(val_auc) if val_auc is not None else None,
                'lr_groups': [pg['lr'] for pg in self.optimizer.param_groups],
            })

        best_val = self.early_stopper.best if (has_val and getattr(self.early_stopper, 'best_state', None)) else (
            round_best_val if round_best_val > -float('inf') else 0.0
        )
        early_stopped = int(epochs_run < self.epochs)

        train_metrics = {
            'train_samples': int(total_samples),
            'epochs_completed': int(epochs_run),
            'best_val_auc': float(best_val),
            'early_stopped': early_stopped,
            'patience': int(self.early_stopper.patience) if hasattr(self, 'early_stopper') else None,
            'train_loss': float(last_epoch_loss),
            'train_auc': float(last_epoch_auc),
            'train_accuracy': float(last_epoch_acc),
        }

        self.log_info(fl_ctx, f"Training finished (epochs_run={epochs_run}, best_val_auc={best_val:.4f})")
        return train_metrics


    def _evaluate_model(self, fl_ctx: FLContext, split='val', model=None):
        """Run evaluation and return base metric dict (auc, accuracy, loss, samples).

        `model` defaults to self.model (the global-tracked w); pass the deployed
        model (e.g. the Ditto personal model v) to evaluate it instead.
        """
        eval_model = model if model is not None else self.model
        core = evaluate_model(eval_model, self.data_loader, self.criterion, self.device,
                              split=split, decision_threshold=self.decision_threshold,
                              eval_threshold_sweep=self.eval_threshold_sweep)
        out = {
            "auc": float(core["auc"]),
            "accuracy": float(core["accuracy"]),
            "loss": float(core["loss"]),
            "samples": int(core.get("total_samples", 0)),
            # operating-point metrics at decision_threshold (AUC stays threshold-free)
            "sensitivity": float(core.get("sensitivity", float("nan"))),
            "specificity": float(core.get("specificity", float("nan"))),
            "precision": float(core.get("precision", float("nan"))),
        }
        self.log_info(
            fl_ctx,
            f"{split.capitalize()} evaluation - AUC {out['auc']:.4f} Acc {out['accuracy']:.2f}% "
            f"Loss {out['loss']:.4f} | @thr={self.decision_threshold:.2f} "
            f"Sens {out['sensitivity']:.4f} Spec {out['specificity']:.4f} Prec {out['precision']:.4f}"
        )
        # Additive operating-point sweep (greppable; denominators logged so small-n noise is
        # judgeable). Does NOT affect selection/early-stop/saved-best — logging only.
        n_pos = int(core.get("n_pos", 0))
        n_neg = int(core.get("n_neg", 0))
        for sw in core.get("sweep", []):
            self.log_info(
                fl_ctx,
                f"[threshold-sweep] split={split} thr={sw['threshold']:.2f} "
                f"sens={sw['sensitivity']:.4f} spec={sw['specificity']:.4f} prec={sw['precision']:.4f} "
                f"n_flagged={sw['n_flagged']} n_pos={n_pos} n_neg={n_neg}"
            )
        return out


    def _consume_global(self, incoming, fl_ctx: FLContext):
        """Method-aware load of the received global weights into the tracked model w.

          local             -> skip the load (each site trains independently from its
                               pretrained init; it still returns w so NVFLARE plumbing
                               is happy, but never consumes the aggregate).
          fedbn / use_fedbn -> load global but SKIP BatchNorm keys (local BN preserved).
          else              -> full load.

        Also captures the frozen global reference w_ref (FedProx/Ditto) and lazily
        initializes the Ditto personal model v exactly once (from the pretrained w).
        """
        method = self.method
        needs_ref = method in ("fedprox", "ditto", "ditto_modulewise")
        self._prox_ref = None
        self._prox_lambda = None

        if incoming is None:
            self.log_info(fl_ctx, "No global weights in shareable; keeping current model")
        elif method == "local":
            self.log_info(fl_ctx, "[method=local] skipping global load (independent local training)")
        else:
            skip = set()
            if method == "fedbn" or self.use_fedbn:
                skip = bn_state_keys(self._underlying)
            stats = load_global_into(self._underlying, incoming, skip_keys=skip)
            self.log_info(
                fl_ctx,
                f"[method={method}] consumed global: {stats} (bn_keys_skipped={len(skip)})"
            )

        # Frozen reference = the round's received global weights (current w state).
        if needs_ref and incoming is not None:
            self._global_ref = clone_detached_state(self._underlying)
            if method == "fedprox":
                self._prox_ref = self._global_ref
                self._prox_lambda = float(self.fedprox_mu)
        else:
            self._global_ref = None

        # Lazily create the Ditto personal model v ONCE, from the (pretrained) w.
        if method in ("ditto", "ditto_modulewise") and self._v_model is None:
            self._v_model = copy.deepcopy(self._underlying).to(self.device)
            try:
                opt, _, _ = configure_optimizers(self._v_model, self._opt_args)
            except Exception as e:
                self.log_warning(fl_ctx, f"[ditto] failed to build v optimizer via configure_optimizers: {e}")
                opt = torch.optim.Adam(self._v_model.parameters(), lr=self.lr_heads,
                                       weight_decay=self.weight_decay)
            self._v_optimizer = opt
            self.log_info(fl_ctx, "[ditto] initialized personal model v from pretrained init (persists across rounds)")


    def _train_personal_model(self, fl_ctx: FLContext, abort_signal: Signal):
        """Ditto personal pass: train v for self.epochs with task_loss(v) + proximal(v, w_ref).

        v and its optimizer are instance attributes that persist across rounds; v is
        never reset to the global. lambda is a scalar (ditto) or a per-group dict
        (ditto_modulewise: {global, local, fusion}). Runs in plain fp32 (no AMP) for
        simplicity/correctness — this is the deployed model, not the comms payload.
        """
        v = self._v_model
        opt = self._v_optimizer
        if v is None or opt is None or self._global_ref is None:
            self.log_warning(fl_ctx, "[ditto] personal model/ref not ready; skipping personal pass")
            return
        lam = self.ditto_lambda if self.method == "ditto" else self.lam_dict
        self.log_info(fl_ctx, f"[ditto] personal pass: epochs={self.epochs} lambda={lam}")
        v.train()
        for epoch in range(self.epochs):
            if abort_signal.triggered:
                break
            n_batches = 0
            for inputs, targets, metadata in self.data_loader.get_batch_iterator('train'):
                if abort_signal.triggered:
                    break
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                opt.zero_grad(set_to_none=True)
                outputs = v(inputs)
                loss = self._compute_task_loss(outputs, targets)
                loss = loss + proximal_penalty(v, self._global_ref, lam)
                loss.backward()
                opt.step()
                n_batches += 1
            self.log_info(fl_ctx, f"[ditto] personal epoch {epoch + 1}/{self.epochs} done ({n_batches} batches)")


    def _dump_baseline(self, fl_ctx: FLContext):
        """Round-0 cold-start reference: evaluate the pretrained weights on val+test and
        dump per-sample predictions tagged method=pretrained_baseline, round=0. This is
        the row every trained method is measured against and verifies the load+pipeline.
        """
        report = getattr(self, "_pretrained_report", None)
        if report is not None:
            self.log_info(fl_ctx, f"[baseline] pretrained load report: {report}")
        for split in ("val", "test"):
            if not self.data_loader.get_data_for_split(split):
                continue
            core = self._evaluate_model(fl_ctx, split=split, model=self.model)
            self.log_info(
                fl_ctx,
                f"[baseline] site={fl_ctx.get_identity_name()} method=pretrained_baseline "
                f"round=0 {split}_auc={core['auc']:.4f} {split}_acc={core['accuracy']:.2f} n={core['samples']}"
            )
            self._dump_predictions(fl_ctx, split=split, model=self.model,
                                   round_idx=0, method_tag="pretrained_baseline")


    def _dump_predictions(self, fl_ctx: FLContext, split, model, round_idx, method_tag):
        """Save per-sample (view-level) malignancy scores + labels + site_id to CSV so
        breast-level AUC, DeLong CIs and cross-site sigma(AUC) can be computed offline.
        """
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)
        client = fl_ctx.get_identity_name()
        eval_model = model if model is not None else self.model
        eval_model.eval()
        rows = []
        sal_maps = []    # malignant saliency channel per image, aligned with `rows`
        sal_paths = []   # same 'path' key as the CSV -> joins map<->prediction offline
        sal_labels = []
        with torch.no_grad():
            for inputs, targets, metas in self.data_loader.get_batch_iterator(split):
                inputs = inputs.to(self.device)
                outputs = eval_model(inputs)
                probs = malignant_score(outputs).detach().cpu().numpy().reshape(-1)
                tnp = targets.detach().cpu().numpy().reshape(-1)
                # Malignant saliency channel (already a sigmoid probability), (B, h, w)
                _, _, _, saliency = gmic_outputs(outputs)
                sal_np = saliency[:, 1].detach().cpu().numpy() if saliency is not None else None
                for i, meta in enumerate(metas):
                    rows.append({
                        "site_id": client,
                        "round": round_idx,
                        "method": method_tag,
                        "split": split,
                        "exam_id": meta.get("exam_id"),
                        "view": meta.get("view"),
                        "path": meta.get("path"),
                        "prob_malignant": float(probs[i]),
                        "label": int(tnp[i]),
                    })
                    if sal_np is not None:
                        sal_maps.append(sal_np[i].astype(np.float32))
                        sal_paths.append(meta.get("path"))
                        sal_labels.append(int(tnp[i]))
        out = os.path.join(
            persistent_dir, f"{client}_predictions_{method_tag}_round{round_idx}_{split}.csv"
        )
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "site_id", "round", "method", "split", "exam_id", "view",
                "path", "prob_malignant", "label"])
            writer.writeheader()
            writer.writerows(rows)
        self.log_info(fl_ctx, f"[predictions] wrote {len(rows)} rows -> {out}")

        # Malignant saliency maps for the cross-site heatmap comparison. Keyed by the
        # SAME 'path' identifier as the prediction CSV, so map<->prediction<->label<->
        # site<->method all join offline with no model needed. Values are post-sigmoid
        # probabilities in [0,1] (outputs[3][:,1]), never logits.
        if sal_maps:
            sal_out = os.path.join(
                persistent_dir,
                f"{client}_saliency_{method_tag}_round{round_idx}_{split}.npz",
            )
            np.savez_compressed(
                sal_out,
                saliency=np.stack(sal_maps, axis=0),   # (N, h, w) float32 in [0,1]
                path=np.array(sal_paths),              # join key (matches CSV 'path')
                label=np.array(sal_labels, dtype=np.int64),
                site_id=str(client), method=str(method_tag),
                split=str(split), round=int(round_idx),
            )
            self.log_info(fl_ctx, f"[saliency] wrote {len(sal_maps)} maps -> {sal_out}")


    def _save_local_model(self, fl_ctx: FLContext, shareable: Shareable, model=None):
        """Save current (deployed) model state"""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)

        save_model = model if model is not None else self.model
        model_path = f"{persistent_dir}/{client_name}_gmic_model_round_{current_round}.pth"
        torch.save(save_model.state_dict(), model_path)
        self.log_info(fl_ctx, f"Model saved: {model_path}")


    def _save_best_model(self, fl_ctx, metrics, shareable: Shareable, model_type="best", model=None):
        """Save best performing (deployed) model (expects metrics with keys including auc)."""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)

        save_model = model if model is not None else self.model
        model_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_model.pth"
        torch.save(save_model.state_dict(), model_path)

        metrics_path = f"{persistent_dir}/{client_name}_{model_type}_gmic_metrics.json"
        enriched = {
            "client": client_name,
            "round": current_round,
            "model_type": model_type,
            **metrics
        }
        auc_val = metrics.get('auc')
        if isinstance(auc_val, (int, float)):
            if 'val' in model_type and 'val_auc' not in enriched:
                enriched['val_auc'] = auc_val
            if 'test' in model_type and 'test_auc' not in enriched:
                enriched['test_auc'] = auc_val

        with open(metrics_path, "w") as f:
            json.dump(enriched, f, indent=2)

        auc_val = metrics.get('auc') or metrics.get(f'{model_type}_auc') or 'N/A'
        self.log_info(fl_ctx, f"✅ {model_type} model saved (AUC={auc_val})")


    def _save_final_results(self, fl_ctx: FLContext):
        """Save final training results and metrics summary"""
        persistent_dir = getattr(self, 'results_dir', "/workspace/gmic_results")
        os.makedirs(persistent_dir, exist_ok=True)

        client_name = fl_ctx.get_identity_name()

        # If initialize() failed (e.g. data loader threw), self.data_loader is None. Don't raise
        # a confusing secondary AttributeError in END_RUN that masks the real initialize() error;
        # log and return so the original traceback stays the visible one.
        if self.data_loader is None:
            self.log_warning(fl_ctx, "[END_RUN] data_loader is None (initialize failed) — skipping final results; see the earlier initialize() error")
            return

        # Final safeguard: if best_metrics are still default zeros, attempt late recovery
        if (self.best_metrics.get('val_auc', 0.0) == 0.0 or self.best_metrics.get('test_auc', 0.0) == 0.0):
            try:
                self._recover_persisted_best_metrics()
            except Exception as e:
                self.log_warning(fl_ctx, f"Late recovery of best metrics failed: {e}")

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
        self.log_info(fl_ctx, "Training Complete!")
        self.log_info(fl_ctx, f"Best Validation AUC: {self.best_metrics['val_auc']:.4f}")
        self.log_info(fl_ctx, f"Best Test AUC: {self.best_metrics['test_auc']:.4f}")


    def _recover_persisted_best_metrics(self):
        """Load previously saved best_val_overall / best_test_overall metrics JSON files to restore best_metrics.

        This is needed because the simulator may create a fresh ClientRunner (and executor instance)
        for the END_RUN phase, resetting in-memory best_metrics to 0.0. We persist metrics per round
        already (e.g., site-1_best_val_overall_gmic_metrics.json) – load them if present.
        """
        persistent_dir = getattr(self, 'results_dir', self.output_dir or "/workspace/gmic_results")
        if not os.path.isdir(persistent_dir):
            return

        try:
            best_val_file = None
            best_test_file = None
            for fname in os.listdir(persistent_dir):
                if fname.endswith('_best_val_overall_gmic_metrics.json'):
                    best_val_file = os.path.join(persistent_dir, fname)
                elif fname.endswith('_best_test_overall_gmic_metrics.json'):
                    best_test_file = os.path.join(persistent_dir, fname)

            updated = False
            if best_val_file and os.path.isfile(best_val_file):
                with open(best_val_file, 'r') as f:
                    data = json.load(f)
                auc_val = data.get('auc') or data.get('val_auc')
                if isinstance(auc_val, (int, float)) and auc_val > self.best_metrics.get('val_auc', 0.0):
                    self.best_metrics['val_auc'] = float(auc_val)
                    updated = True

            if best_test_file and os.path.isfile(best_test_file):
                with open(best_test_file, 'r') as f:
                    data = json.load(f)
                auc_val = data.get('auc') or data.get('test_auc')
                if isinstance(auc_val, (int, float)) and auc_val > self.best_metrics.get('test_auc', 0.0):
                    self.best_metrics['test_auc'] = float(auc_val)
                    updated = True

            if updated:
                self._logger.info("[EXEC] Recovered persisted best metrics: %s", self.best_metrics)

        except Exception as e:
            self._logger.warning("[EXEC][WARN] Could not recover best metrics: %s", e, exc_info=True)


"""
    def _reset_model_weights(self):

        # Prefer underlying (for DataParallel), fall back to self.model
        model = getattr(self, "_underlying", None) or getattr(self, "model", None)
        if model is None:
            # initialize() hasn't built the model yet; nothing to reset
            self._logger.warning("[RANDOM_INIT] _reset_model_weights called before model was built; skipping.")
            return

        def _init(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        model.apply(_init)
        self._logger.info("[RANDOM_INIT] All modules with reset_parameters() have been reinitialized.")
"""
