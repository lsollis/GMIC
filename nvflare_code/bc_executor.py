# ============================================================================
# bc_executor.py - NVFLARE Executor using existing GMIC components
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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

# Import your existing GMIC components
from src.modeling import gmic
from src.data_loading import loading
from src.utilities import pickling, tools
from src.constants import VIEWS, PERCENT_T_DICT

class GMICDataLoader:
    """Data loader using your existing GMIC data loading infrastructure"""
    
    def __init__(self, exam_list_path, image_path, segmentation_path, batch_size=4):
        self.exam_list_path = exam_list_path
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.batch_size = batch_size
        
        # Load exam list using your existing utilities
        self.exam_list = pickling.unpickle_from_file(exam_list_path)
        self.data_list = self._unpack_exam_into_images()
        
    def _unpack_exam_into_images(self):
        """Convert exam list to individual image entries using your existing format"""
        data_list = []
        for exam in self.exam_list:
            # Handle the format from your existing data structure
            for view in VIEWS.LIST:
                if view in exam and exam[view] is not None:
                    for short_file_path in exam[view]:
                        datum = {
                            'short_file_path': short_file_path,
                            'view': view,
                            'full_view': view,
                            'horizontal_flip': exam['horizontal_flip'],
                            'best_center': exam['best_center'][view][0] if 'best_center' in exam and view in exam['best_center'] else None,
                            'cancer_label': exam['cancer_label']
                        }
                        data_list.append(datum)
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def get_batch_iterator(self):
        """Iterator that yields batches of processed images"""
        for i in range(0, len(self.data_list), self.batch_size):
            batch_data = self.data_list[i:i + self.batch_size]
            batch_images = []
            batch_labels = []
            
            for datum in batch_data:
                try:
                    # Load and process image using your existing loading functions
                    image_path = os.path.join(self.image_path, datum['short_file_path'] + '.png')
                    
                    # Use your existing image loading pipeline
                    loaded_image = loading.load_image(
                        image_path=image_path,
                        view=datum['view'],
                        horizontal_flip=datum['horizontal_flip']
                    )
                    
                    # Process image using your existing preprocessing
                    if datum['best_center'] is not None:
                        processed_image = loading.process_image(
                            image=loaded_image,
                            view=datum['view'],
                            best_center=datum['best_center']
                        )
                    else:
                        # Fallback if no best_center available
                        processed_image = loaded_image.copy()
                        loading.standard_normalize_single_image(processed_image)
                    
                    # Convert to tensor format expected by GMIC (N, C, H, W)
                    processed_image = np.expand_dims(np.expand_dims(processed_image, 0), 0)
                    batch_images.append(processed_image)
                    
                    # Extract labels in the format expected by your model
                    view_prefix = datum['view'].split('-')[0].lower()  # 'l' or 'r'
                    benign_label = datum['cancer_label'].get(f'{view_prefix}_benign', 0)
                    malignant_label = datum['cancer_label'].get(f'{view_prefix}_malignant', 0)
                    batch_labels.append([benign_label, malignant_label])
                    
                except Exception as e:
                    print(f"Error processing {datum['short_file_path']}: {e}")
                    continue
            
            # Convert to tensors
            if batch_images:
                batch_tensor = torch.FloatTensor(np.concatenate(batch_images, axis=0))
                label_tensor = torch.FloatTensor(batch_labels)
                yield batch_tensor, label_tensor


class GMICFederatedExecutor(Executor):
    """NVFLARE Executor that uses your existing GMIC model and data loading"""
    
    def __init__(
        self,
        epochs: int = 5,
        lr: float = 1e-4,
        batch_size: int = 4,
        exam_list_path: str = "/workspace/data/bc_data/data.pkl",
        image_path: str = "/workspace/data/bc_data/cropped_images",
        segmentation_path: str = "/workspace/data/bc_data/segmentation",
        model_path: str = "/workspace/data/bc_model.pth",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        pre_train_task_name: str = AppConstants.TASK_GET_WEIGHTS,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL,
        validate_task_name: str = AppConstants.TASK_VALIDATION,
        exclude_vars=None,
    ):
        """
        GMIC Federated Executor using existing model components
        
        Args:
            epochs: Local training epochs per round
            lr: Learning rate for GMIC training
            batch_size: Batch size for mammography images
            exam_list_path: Path to your exam list pickle file
            image_path: Path to cropped mammography images
            segmentation_path: Path to segmentation files
            model_path: Path to save/load model
            device: Training device (cuda/cpu)
        """
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.exam_list_path = exam_list_path
        self.image_path = image_path
        self.segmentation_path = segmentation_path
        self.model_path = model_path
        self.device = device
        
        # Task names
        self.pre_train_task_name = pre_train_task_name
        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name
        self.validate_task_name = validate_task_name
        self.exclude_vars = exclude_vars
        
        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.data_loader = None
        self.persistence_manager = None
        self.best_auc = 0.0
        
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """Handle NVFLARE events"""
        if event_type == EventType.START_RUN:
            self.initialize()
    
    def initialize(self):
        """Initialize GMIC model and data loading using existing components"""
        
        # Use your existing GMIC model parameters
        self.gmic_parameters = {
            "device_type": "gpu" if "cuda" in self.device else "cpu",
            "gpu_number": 0,
            "max_crop_noise": (100, 100),
            "max_crop_size_noise": 100,
            "image_path": self.image_path,
            "segmentation_path": self.segmentation_path,
            "cam_size": (46, 30),
            "K": 6,
            "crop_shape": (256, 256),
            "post_processing_dim": 256,
            "num_classes": 2,
            "use_v1_global": False,
            "percent_t": PERCENT_T_DICT["1"]  # Use model 1's percent_t
        }
        
        # Create GMIC model using your existing implementation
        self.model = gmic.GMIC(self.gmic_parameters)
        self.model.to(self.device)
        
        # Set up training components
        self.criterion = nn.BCELoss()  # Binary cross entropy for cancer classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Initialize data loader using your existing data loading pipeline
        self.data_loader = GMICDataLoader(
            exam_list_path=self.exam_list_path,
            image_path=self.image_path,
            segmentation_path=self.segmentation_path,
            batch_size=self.batch_size
        )
        
        self._n_iterations = len(self.data_loader) // self.batch_size
        
        # Set up persistence manager
        self._default_train_conf = {"train": {"model": "GMIC"}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(),
            default_train_conf=self._default_train_conf
        )
    
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """Execute federated learning tasks"""
        try:
            if task_name == self.pre_train_task_name:
                # Send initial model weights
                return self._get_model_weights()
                
            elif task_name == self.train_task_name:
                # Local training round
                dxo = from_shareable(shareable)
                if dxo.data_kind != DataKind.WEIGHTS:
                    return make_reply(ReturnCode.BAD_TASK_DATA)
                
                # Load global weights into local model
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self.model.load_state_dict(torch_weights)
                
                # Perform local training
                train_metrics = self._local_train(fl_ctx, abort_signal, shareable)
                
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)
                
                # Save local model
                self._save_local_model(fl_ctx, shareable)
                
                # Return weights with metrics
                weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
                dxo = DXO(
                    data_kind=DataKind.WEIGHTS,
                    data=weights,
                    meta={
                        "train_loss": train_metrics["avg_loss"],
                        "train_samples": train_metrics["num_samples"],
                        MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations
                    }
                )
                shareable = dxo.to_shareable()
                shareable.set_header(AppConstants.MODEL_OWNER, fl_ctx.get_identity_name())
                return shareable
            
            elif task_name == self.validate_task_name:
                # Global model validation
                dxo = from_shareable(shareable)
                if dxo.data_kind == DataKind.WEIGHTS:
                    torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                    self.model.load_state_dict(torch_weights)
                
                # Validate using your existing evaluation approach
                val_metrics = self._local_validate(fl_ctx, shareable)
                
                # Return validation metrics
                dxo = DXO(
                    data_kind=DataKind.METRICS,
                    data={
                        "val_auc": val_metrics["auc"],
                        "val_accuracy": val_metrics["accuracy"],
                        "validation_loss": val_metrics["loss"]
                    }
                )
                return dxo.to_shareable()
            
            elif task_name == self.submit_model_task_name:
                # Submit trained model
                ml = self._load_local_model(fl_ctx)
                dxo = model_learnable_to_dxo(ml)
                return make_reply(dxo.to_shareable())
            
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
                
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in GMIC executor: {e}")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
    
    def _get_model_weights(self) -> Shareable:
        """Get current GMIC model weights"""
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        
        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=weights,
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()
    
    def _local_train(self, fl_ctx, abort_signal, shareable: Shareable):
        """Local training using your existing GMIC model"""
        
        self.model.train()
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.epochs):
            self.log_info(fl_ctx, f"Starting epoch {epoch + 1}/{self.epochs}")
            
            for batch_idx, (inputs, targets) in enumerate(self.data_loader.get_batch_iterator()):
                if abort_signal.triggered:
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass through your GMIC model
                outputs = self.model(inputs)
                
                # Calculate loss for benign/malignant classification
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                num_samples += inputs.size(0)
                
                # Log progress
                if batch_idx % 5 == 0:
                    self.log_info(fl_ctx, 
                        f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_samples, 1)
        
        self.log_info(fl_ctx, f"Local training completed. Avg loss: {avg_loss:.4f}")
        
        return {
            "avg_loss": avg_loss,
            "num_samples": num_samples,
            "epochs_completed": self.epochs
        }
    
    def _local_validate(self, fl_ctx, shareable: Shareable):
        """Validation using your existing GMIC components"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in self.data_loader.get_batch_iterator():
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for AUC calculation
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(predictions[:, 1])  # Malignant probability
                all_targets.extend(targets[:, 1].cpu().numpy())  # Malignant label
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # AUC score for cancer detection
        auc = roc_auc_score(all_targets, all_predictions) if len(np.unique(all_targets)) > 1 else 0.0
        
        # Accuracy using 0.5 threshold
        predicted_labels = (all_predictions > 0.5).astype(int)
        accuracy = 100 * np.mean(predicted_labels == all_targets)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Update best AUC
        if auc > self.best_auc:
            self.best_auc = auc
            self._save_best_model(fl_ctx, {"auc": auc, "accuracy": accuracy, "loss": avg_loss}, shareable)
        
        self.log_info(fl_ctx, f"Validation - AUC: {auc:.4f}, Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return {
            "auc": auc,
            "accuracy": accuracy,
            "loss": avg_loss,
            "total": len(all_targets)
        }
    
    def _save_local_model(self, fl_ctx: FLContext, shareable: Shareable):
        """Save current GMIC model state"""
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM)
        )
        models_dir = os.path.join(run_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save to workspace
        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), self.model_path)
        
        # Save to persistent location
        persistent_dir = "/workspace/gmic_results"
        os.makedirs(persistent_dir, exist_ok=True)
        
        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        
        # Save model with round info
        persistent_model_path = f"{persistent_dir}/{client_name}_gmic_model_round_{current_round}.pth"
        torch.save(self.model.state_dict(), persistent_model_path)
        
        self.log_info(fl_ctx, f"GMIC model saved: {persistent_model_path}")
    
    def _save_best_model(self, fl_ctx, eval_metrics, shareable: Shareable):
        """Save best performing GMIC model"""
        persistent_dir = "/workspace/gmic_results"
        os.makedirs(persistent_dir, exist_ok=True)
        
        client_name = fl_ctx.get_identity_name()
        current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        
        # Save best model
        best_model_path = f"{persistent_dir}/{client_name}_BEST_gmic_model.pth"
        torch.save(self.model.state_dict(), best_model_path)
        
        # Save best metrics
        best_metrics_path = f"{persistent_dir}/{client_name}_BEST_gmic_metrics.txt"
        with open(best_metrics_path, "w") as f:
            f.write(f"Client: {client_name}\n")
            f.write(f"Round: {current_round}\n")
            f.write(f"Best AUC: {eval_metrics['auc']:.4f}\n")
            f.write(f"Best Accuracy: {eval_metrics['accuracy']:.4f}%\n")
            f.write(f"Best Loss: {eval_metrics['loss']:.4f}\n")
        
        self.log_info(fl_ctx, f"✅ Best GMIC model saved with AUC={eval_metrics['auc']:.4f}")
    
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