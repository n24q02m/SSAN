import logging
import os
import sys
import pandas as pd
import psutil
import torch
import optuna
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader, Subset
from torch.nn import Module
from torch.optim import Adam, SGD, lr_scheduler
from torch.cuda import memory_allocated, empty_cache, get_device_properties, Event, synchronize
import gc
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.model.losses import ClassificationLoss, ContrastiveLoss, DomainAdversarialLoss
from src.runner.trainer import Trainer

def find_optimal_batch_size(
    model: Module,
    sample_batch: tuple,
    min_batch: int = 4,
    max_batch: int = 128,
    max_memory_use: float = 0.8,
) -> int:
    """Find optimal batch size based on GPU/CPU memory
    
    Args:
        model: Model to test batch sizes with
        sample_batch: Sample batch to test with (x, depth, label, domain)
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try 
        max_memory_use: Maximum memory utilization (0-1)
        
    Returns:
        Optimal batch size
    """
    if not 0 < max_memory_use <= 1:
        raise RuntimeError("max_memory_use must be between 0 and 1")
    
    device = next(model.parameters()).device
    
    # Get initial memory usage
    if device.type == 'cuda':
        empty_cache()
        gc.collect()
        initial_mem = memory_allocated(device)
        total_mem = get_device_properties(device).total_memory
    else:
        initial_mem = psutil.Process().memory_info().rss
        total_mem = psutil.virtual_memory().total

    def test_batch_size(batch_size: int) -> bool:
        try:
            # Create batch of given size
            x, depth, label, domain = sample_batch
            batch = (
                x.repeat(batch_size, 1, 1, 1),
                depth.repeat(batch_size, 1, 1, 1), 
                label.repeat(batch_size),
                domain.repeat(batch_size)
            )
            
            # Test forward and backward pass
            with torch.no_grad():
                _ = model(*batch)
            
            # Check memory usage
            if device.type == 'cuda':
                used_mem = memory_allocated(device) - initial_mem
            else:
                used_mem = psutil.Process().memory_info().rss - initial_mem
                
            mem_ratio = (used_mem + initial_mem) / total_mem
            
            return mem_ratio < max_memory_use
            
        except RuntimeError:  # Out of memory
            return False
            
    # Binary search for largest working batch size
    left, right = min_batch, max_batch
    optimal_batch = min_batch
    
    print("\nSearching for optimal batch size...")
    # Binary search with progress bar
    tried_sizes = []
    with tqdm(total=int(np.log2(max_batch-min_batch+1))+1, desc="Testing batch sizes") as pbar:
        while left <= right:
            mid = (left + right) // 2
            if test_batch_size(mid):
                optimal_batch = mid
                left = mid + 1
                tried_sizes.append(f"{mid} ✓")
            else:
                right = mid - 1
                tried_sizes.append(f"{mid} ✗")
            pbar.update(1)
            pbar.set_postfix({"Tested": ", ".join(tried_sizes[-3:])})
            
    return optimal_batch

def find_optimal_workers(
    dataloader: DataLoader,
    max_workers: int = None
) -> int:
    """Find optimal number of workers for data loading
    
    Args:
        dataloader: DataLoader to optimize
        max_workers: Maximum number of workers to try
        
    Returns:
        Optimal number of workers
    """
    if dataloader.num_workers == 0:
        return 0
        
    if max_workers is None:
        max_workers = os.cpu_count()

    print("\nFinding optimal number of workers...")
    times = []
    workers = range(0, max_workers + 1, 2)
    
    for num_workers in tqdm(workers, desc="Testing worker counts"):
        dataloader.num_workers = num_workers
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        
        start.record()
        for _ in dataloader:
            pass
        end.record()
        synchronize()
        times.append(start.elapsed_time(end))
        
    optimal_workers = workers[np.argmin(times)]
    return optimal_workers

class HyperparameterOptimizer:
    """Optimize hyperparameters using Optuna"""
    
    def __init__(
        self,
        model_class: type,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        study_name: str = "ssan_optimization",
        n_trials: int = 100,
        timeout: int = 3600,  # 1 hour
        output_dir: Path = None,
        optimization_fraction: float = 0.1
    ):
        self.model_class = model_class
        self.train_loader = train_loader 
        self.val_loader = val_loader
        self.config = config
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.output_dir = output_dir or Path("optuna_studies")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimization_fraction = optimization_fraction
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize validation accuracy
            storage=f"sqlite:///{self.output_dir/f'{study_name}.db'}",
            load_if_exists=True
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _get_balanced_indices(self, dataset, fraction):
        """Get balanced indices for both classes"""
        pos_indices = []
        neg_indices = [] 
        
        # Collect indices
        for i in range(len(dataset)):
            _, _, label, _ = dataset[i]
            if label == 1:
                pos_indices.append(i)
            else:
                neg_indices.append(i)
                
        # Ensure minimum 1 sample per class
        n_samples = max(1, int(min(len(pos_indices), len(neg_indices)) * fraction))
        
        # Sample with replacement if needed
        replace = n_samples > min(len(pos_indices), len(neg_indices))
        
        pos_indices = np.random.choice(pos_indices, n_samples, replace=replace)
        neg_indices = np.random.choice(neg_indices, n_samples, replace=replace)
        
        return np.concatenate([pos_indices, neg_indices])

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for hyperparameter optimization"""
        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "lambda_adv": trial.suggest_float("lambda_adv", 0.01, 1.0),
            "lambda_contrast": trial.suggest_float("lambda_contrast", 0.01, 1.0),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"]),
            "scheduler": trial.suggest_categorical("scheduler", ["step", "cosine"]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5)
        }
        
        if params["optimizer"] == "sgd":
            params["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
            
        if params["scheduler"] == "step":
            params["step_size"] = trial.suggest_int("step_size", 5, 50)
            params["gamma"] = trial.suggest_float("gamma", 0.1, 0.5)
        else:
            params["warmup_epochs"] = trial.suggest_int("warmup_epochs", 5, 20)
            params["min_lr"] = trial.suggest_float("min_lr", 1e-6, 1e-4, log=True)
            
        # Update config & reinitialize training components
        for k, v in params.items():
            setattr(self.config, k, v)

        # Initialize model, trainer etc.
        model = self.model_class(
            num_domains=self.config.num_domains,
            ada_blocks=self.config.ada_blocks,
            dropout=trial.suggest_float("dropout", 0.1, 0.5)
        ).to(self.config.device)
        
        if params["optimizer"] == "adam":
            optimizer = Adam(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
        else:
            optimizer = SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"]
            )
            
        if params["scheduler"] == "step":
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=params["step_size"],
                gamma=params["gamma"]
            )
        else:
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=params["warmup_epochs"],
                eta_min=params["min_lr"]
            )
            
        criterion = {
            'cls': ClassificationLoss(),
            'domain': DomainAdversarialLoss(),
            'contrast': ContrastiveLoss()
        }
        criterion = {k: v.to(self.config.device) for k, v in criterion.items()}

        # Create subset of data for optimization
        train_subset_size = int(len(self.train_loader.dataset) * self.optimization_fraction)
        val_subset_size = int(len(self.val_loader.dataset) * self.optimization_fraction)
        
        train_subset = Subset(
            self.train_loader.dataset,
            indices=self._get_balanced_indices(
                self.train_loader.dataset, 
                self.optimization_fraction
            )
        )
        val_subset = Subset(
            self.val_loader.dataset,
            indices=self._get_balanced_indices(
                self.val_loader.dataset,
                self.optimization_fraction
            )
        )

        # Create new dataloaders with subsets
        train_loader_subset = DataLoader(
            train_subset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers
        )
        val_loader_subset = DataLoader(
            val_subset,
            batch_size=self.val_loader.batch_size,
            shuffle=False,
            num_workers=self.val_loader.num_workers
        )

        # Initialize trainer with subset data
        trainer = Trainer(
            model=model,
            train_loader=train_loader_subset,  # Use subset
            val_loader=val_loader_subset,      # Use subset 
            test_loader=val_loader_subset,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=self.config,
            device=self.config.device
        )
        
       # Train with progress bar
        n_epochs = 5
        min_accuracy = 0.6  # Minimum accuracy threshold
        consecutive_poor = 0
        max_poor = 2  # Maximum allowed consecutive poor performance
        best_val_acc = 0
        
        # Tạo thư mục riêng cho optimization metrics
        opt_metrics_dir = self.output_dir / "metrics"
        opt_metrics_dir.mkdir(exist_ok=True)
        
        # Lưu metrics vào file riêng
        metrics_file = opt_metrics_dir / f"trial_{trial.number}_metrics.csv"
        
        for epoch in range(n_epochs):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = trainer.evaluate(self.val_loader)
            
            # Lưu metrics
            pd.DataFrame([{
                'epoch': epoch,
                'trial': trial.number,
                **train_metrics,
                **{'val_'+k: v for k,v in val_metrics.items()}
            }]).to_csv(metrics_file, mode='a', header=not metrics_file.exists(), index=False)
            
            # Report accuracy to Optuna for pruning
            accuracy = val_metrics["accuracy"]
            trial.report(accuracy, epoch)
            
            # Update best accuracy
            best_val_acc = max(best_val_acc, accuracy)
            
            # Check for pruning based on accuracy threshold
            if accuracy < min_accuracy:
                consecutive_poor += 1
            else:
                consecutive_poor = 0
                
            if consecutive_poor >= max_poor:
                self.logger.info(f"Trial pruned due to poor performance: accuracy={accuracy:.6f}")
                raise optuna.TrialPruned()
                
            # Let Optuna handle pruning
            if trial.should_prune():
                self.logger.info("Trial pruned by Optuna")
                raise optuna.TrialPruned()
                
        return best_val_acc
        
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        print("\nStarting hyperparameter optimization...")
        print(f"Will run {self.n_trials} trials with {self.timeout}s timeout")
        print("This may take a while...\n")

        try:
            self.study.optimize(
                self.objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
                n_jobs=1,  # Run sequentially 
                catch=(Exception,)
            )
            
            # Check if we have any completed trials
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if completed_trials:
                # Print and save results
                self.logger.info("\nOptimization completed!")
                self.logger.info("\nBest trial:")
                trial = self.study.best_trial
                self.logger.info(f"  Value: {trial.value:.6f}")
                self.logger.info("  Params: ")
                for key, value in trial.params.items():
                    self.logger.info(f"    {key}: {value}")

                # Save results
                df = self.study.trials_dataframe()
                results_file = self.output_dir / f"{self.study_name}_results.csv"
                df.to_csv(results_file)
                self.logger.info(f"\nResults saved to {results_file}")
                
                return self.study.best_params
            else:
                self.logger.warning("No trials completed successfully!")
                return {}
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {str(e)}")
            return {}