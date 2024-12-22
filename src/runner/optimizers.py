import logging
import os
import sys
import psutil
import torch
import optuna
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader
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
    max_memory_use: float = 0.6,
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
    
    while left <= right:
        mid = (left + right) // 2
        if test_batch_size(mid):
            optimal_batch = mid
            left = mid + 1
        else:
            right = mid - 1
            
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
    if max_workers is None:
        max_workers = os.cpu_count()

    # Test different numbers of workers
    times = []
    workers = range(0, max_workers + 1, 2)
    for num_workers in workers:
        dataloader.num_workers = num_workers
        
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        
        start.record()
        for _ in dataloader:
            pass
        end.record()
        
        # Waits for everything to finish running
        synchronize()
        times.append(start.elapsed_time(end))
        
    # Find fastest number of workers
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
        output_dir: Path = None
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
        
        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",  # Maximize validation accuracy
            storage=f"sqlite:///{self.output_dir/f'{study_name}.db'}",
            load_if_exists=True
        )
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
            dropout=params["dropout"]
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

        trainer = Trainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=self.config,
            device=self.config.device
        )
        
       # Train with progress bar
        n_epochs = 5
        best_val_acc = 0
        
        with tqdm(range(n_epochs), desc="Optimization epochs") as pbar:
            for epoch in pbar:
                train_metrics = trainer.train_epoch(epoch)
                val_metrics = trainer.evaluate(self.val_loader)
                best_val_acc = max(best_val_acc, val_metrics["accuracy"])
                
                # Update progress bar
                pbar.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_acc': f"{val_metrics['accuracy']:.4f}",
                    'best_acc': f"{best_val_acc:.4f}"
                })
                
                trial.report(val_metrics["accuracy"], epoch)
                if trial.should_prune():
                    self.logger.info("Trial pruned!")
                    raise optuna.TrialPruned()

        return best_val_acc
        
    def optimize(self) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        print("\nStarting hyperparameter optimization...")
        print(f"Will run {self.n_trials} trials with {self.timeout}s timeout")
        print("This may take a while...\n")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Print and save results
        self.logger.info("\nOptimization completed!")
        self.logger.info("\nBest trial:")
        trial = self.study.best_trial
        self.logger.info(f"  Value: {trial.value:.4f}")
        self.logger.info("  Params: ")
        for key, value in trial.params.items():
            self.logger.info(f"    {key}: {value}")

        # Save results
        df = self.study.trials_dataframe()
        results_file = self.output_dir / f"{self.study_name}_results.csv"
        df.to_csv(results_file)
        self.logger.info(f"\nResults saved to {results_file}")
        
        return self.study.best_params