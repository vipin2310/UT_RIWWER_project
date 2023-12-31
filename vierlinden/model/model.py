from contextlib import contextmanager
from io import StringIO
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import torch
import logging
import shutil
import warnings
from vierlinden.config import model_output_path
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar,StochasticWeightAveraging, Callback
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger_internal = logging.getLogger(__name__)

for handler in logging.getLogger("lightning.pytorch").handlers:
    handler.addFilter(lambda record: "PU available:" not in record.getMessage())
    handler.addFilter(lambda record: "LOCAL_RANK:" not in record.getMessage())
    handler.addFilter(lambda record: "Learning rate set to" not in record.getMessage())
    handler.addFilter(lambda record: "Missing logger folder:" not in record.getMessage())
    handler.addFilter(lambda record: "Restoring states from the checkpoint path at" not in record.getMessage())
    handler.addFilter(lambda record: "Restored all states from the checkpoint at" not in record.getMessage())

# Configure warnings
warnings.filterwarnings('ignore', message=".*'loss' is an instance of `nn.Module`.*")
warnings.filterwarnings('ignore', message=".*'logging_metrics' is an instance of `nn.Module`.*")

class TimeSeriesDataSetCreator:
    
    @staticmethod
    def create(dataframe : pd.DataFrame, 
              target_col: str, 
              context_length : int, 
              prediction_length : int, 
              train_frac: float = 0.8, 
              num_time_series: int = 1) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
       
        data = dataframe.copy()
        data = TimeSeriesDataSetCreator.__add_series_and_timeidx(data, num_time_series)
       
        training_cutoff = int(round(data["time_idx"].max() * train_frac))
        
        training = TimeSeriesDataSet(
            data = data[lambda x: x.time_idx <= training_cutoff],
            target_normalizer='auto',
            time_idx='time_idx',
            target=target_col,
            group_ids=['series'],
            time_varying_unknown_reals=list(set(data.columns) - {'Datetime', 'series', 'time_idx'}),
            max_encoder_length=context_length,
            min_encoder_length=context_length,
            max_prediction_length=prediction_length,
            min_prediction_length=prediction_length,
            allow_missing_timesteps=True
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
        
        return training, validation
    
    @staticmethod
    def __add_series_and_timeidx(data: pd.DataFrame, num_time_series: int) -> pd.DataFrame:
        """Adds 'series' and 'timeidx' columns to the dataset for time series modeling.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be processed.
        num_time_series : int
            The number of time series to divide the data into.

        Returns
        -------
        pd.DataFrame
            Dataset with 'series' and 'timeidx' columns added.
        """
        data = data.copy()
        
        # Assign timeidx to each row
        data['time_idx'] = data.index
        
        # Assign series grouping for each row, according to the number of time series
        total_rows = len(data)
        rows_per_group = total_rows // num_time_series

        # Assign each row to a group
        data['series'] = np.arange(total_rows) // rows_per_group
        # Ensure the number of groups does not exceed num_groups
        data['series'] = np.minimum(data['series'], num_time_series - 1)
        
        return data

class MetricCollectionCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = {"train_loss": [], "val_loss": []}

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect train loss
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.metrics["train_loss"].append(train_loss.cpu().detach().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss.cpu().detach().item())

class NHiTSTrainingWrapper:
    
    def __init__(self, 
                 dataframe : pd.DataFrame, 
                 target_col : str,
                 context_length : int,
                 prediction_length : int,
                 batch_size : int,
                 num_workers : int,
                 num_time_series : int = 1,
                 train_val_split : float = 0.8):
        
        self.training_data, self.validation_data = TimeSeriesDataSetCreator.create(dataframe, 
                                                                         target_col = target_col, 
                                                                         context_length = context_length, 
                                                                         prediction_length = prediction_length, 
                                                                         train_frac = train_val_split,
                                                                         num_time_series = num_time_series)
        
        self.train_loader = self.training_data.to_dataloader(train=True, 
                                                        batch_size=batch_size, 
                                                        num_workers=num_workers)
        self.validation_loader = self.validation_data.to_dataloader(train=False, 
                                                               batch_size=batch_size, 
                                                               num_workers=num_workers)
        
        logger_internal.info("Training and validation data and data loaders created successfully.")
        
    def find_optimal_learningrate(self,
                                  seed : int = None,
                                  min_lr = 1e-5,
                                  accelerator = "gpu",
                                  devices = 1,
                                  max_epochs = 20,
                                  gradient_clip_val = 0.01) -> float:
        
        if seed is not None:
            pl.seed_everything(seed)
        
        trainer = pl.Trainer(accelerator=accelerator,
                     devices = devices,
                     max_epochs = max_epochs,
                     gradient_clip_val=gradient_clip_val,
                     enable_model_summary=False,
                     enable_checkpointing=False,
                     benchmark=False,
                     logger = False)
        
        net = NHiTS.from_dataset(self.training_data,
                         learning_rate=3e-2,
                         weight_decay=1e-2,
                         backcast_loss_ratio=1.0)
        
        lr_finder = Tuner(trainer).lr_find(net,
                             train_dataloaders=self.train_loader,
                             val_dataloaders=self.validation_loader,
                             min_lr=min_lr)
        
        return lr_finder.suggestion()
    
    def train(self,
              learning_rate : float,
              weight_decay : float = 1e-2,
              dropout : float = 0.1,
              seed : int = None,
              max_epochs : int = 100,
              accelerator = "gpu",
              devices = 1,
              gradient_clip_val=0.01,
              use_early_stopping : bool = True,
              early_stopping_delta : float = 1e-4,
              early_stopping_patience : int = 10,
              limit_train_batches : float = None,
              enable_model_summary : bool = True,
              clean_up_logging : bool = True,
              logging_steps : int = 5) -> NHiTS:
        
        logger_internal.info("Start setting up trainer and network.")
        self.metrics_callback = MetricCollectionCallback()
        
        if seed is not None:
            pl.seed_everything(seed)
            
        if accelerator == "gpu":
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        
        callbacks=[]
        
        if use_early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", 
                                       min_delta=early_stopping_delta, 
                                       patience=early_stopping_patience, 
                                       verbose=True, 
                                       mode="min"))
        
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        callbacks.append(TQDMProgressBar())
        callbacks.append(self.metrics_callback)
        callbacks.append(StochasticWeightAveraging(swa_lrs=learning_rate,swa_epoch_start=5, device=device))
        
        log_interval = logging_steps
        log_every_n_steps = logging_steps
        log_val_interval = 1
        log_dir = model_output_path + "/" + "training_logs"
        logger = TensorBoardLogger(log_dir)
        
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices = devices,
            enable_model_summary=enable_model_summary,
            gradient_clip_val=gradient_clip_val,
            callbacks=callbacks,
            limit_train_batches=limit_train_batches,
            log_every_n_steps=log_every_n_steps,
            logger = logger
        )
        
        net = NHiTS.from_dataset(
            self.training_data,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            log_interval=log_interval,
            log_val_interval=log_val_interval,
            backcast_loss_ratio=1.0
        )
        
        logger_internal.info("Setup succesful. Starting training procedure.")
        
        trainer.fit(
            net,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.validation_loader
        )
        
        logger_internal.info("Training procedure completed.")
        
        self.final_trainer = trainer
        self.best_model = NHiTS.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # Clean up logs
        if clean_up_logging:
            logger_internal.info("Cleaning up logging files.")
            shutil.rmtree(log_dir)
            logger_internal.info("Logging files cleaned up.")
        
        return self.best_model
    
    def save_trained_model(self, path : str):
        if self.best_model is None:
            raise Exception("No best model available. Please train the model first.")
        
        self.final_trainer.save_checkpoint(path)
        
    def load_trained_model(path : str) -> NHiTS:
        best_model = NHiTS.load_from_checkpoint(path)
        
        return best_model
    
    def print_training_evaluation(self):
        if self.final_trainer is None:
            raise Exception("No final trainer available. Please train the model first.")
        
        print(self.final_trainer.callback_metrics)
    
    def plot_training_result(self):
        if self.final_trainer is None:
            raise Exception("No final trainer available. Please train the model first.")
        
        train_loss = self.metrics_callback.metrics["train_loss"]
        val_loss = self.metrics_callback.metrics["val_loss"]
        epochs = np.arange(0, max(len(train_loss), len(val_loss)))
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(epochs)  # Set x-axis ticks to whole numbers
        plt.legend()
        plt.show()
        
class NHiTSPredictionWrapper:
    
    def __init__(self, trained_model : NHiTS, context_length : int, prediction_length : int):
        
        self.model = trained_model
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model.eval()
        
    def predict(self, dataframe : pd.DataFrame) -> pd.DataFrame:
        
        _, prediction_set = TimeSeriesDataSetCreator.create(
                                dataframe, 
                                target_col="your_target_column", 
                                context_length=self.context_length,
                                prediction_length=self.prediction_length,
                                train_frac=0, # TODO Check if this works
                                num_time_series=1  # Assuming a single time series for prediction TODO make this configurable
                            )

        # Create DataLoader from prediction_set
        prediction_loader = prediction_set.to_dataloader(train=False, batch_size=1, num_workers=0)

        # Make predictions
        predictions = self.model.predict(prediction_loader)
        
        return predictions