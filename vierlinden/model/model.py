from typing import Tuple
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import torch
from vierlinden.config import model_output_path
from pytorch_forecasting import NHiTS, TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar,StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger

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
        data['timeidx'] = data.index
        
        # Assign series grouping for each row, according to the number of time series
        total_rows = len(data)
        rows_per_group = total_rows // num_time_series

        # Assign each row to a group
        data['series'] = np.arange(total_rows) // rows_per_group
        # Ensure the number of groups does not exceed num_groups
        data['series'] = np.minimum(data['series'], num_time_series - 1)
        
        return data

class NHitsTrainingWrapper:
    
    def __init__(self, 
                 dataframe : pd.DataFrame, 
                 target_col : str,
                 batch_size : int,
                 num_workers : int,
                 train_val_split : float = 0.8):
        
        self.training_data, self.validation_data = TimeSeriesDataSetCreator.create(dataframe, 
                                                                         target_col, 
                                                                         batch_size, 
                                                                         num_workers, 
                                                                         train_val_split)
        
        self.train_loader = self.training.to_dataloader(train=True, 
                                                        batch_size=batch_size, 
                                                        num_workers=num_workers)
        self.validation_loader = self.validation.to_dataloader(train=False, 
                                                               batch_size=batch_size, 
                                                               num_workers=num_workers)
        
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
                     logger = False)
        
        net = NHiTS.from_dataset(self.training,
                         learning_rate=3e-2,
                         weight_decay=1e-2,
                         backcast_loss_ratio=1.0)
        
        lr_finder = Tuner(trainer).lr_find(net,
                             train_dataloaders=self.train_dataloader,
                             val_dataloaders=self.val_dataloader,
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
              use_logging : bool = True,
              logging_steps : int = 5) -> pl.Trainer:
              
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
        
        log_interval = None
        log_every_n_steps = None
        log_val_interval = None
        
        if use_logging:
            callbacks.append(LearningRateMonitor(logging_interval='step'))
            callbacks.append(TQDMProgressBar())
            
            log_interval = logging_steps
            log_every_n_steps = logging_steps
            log_val_interval = 1
            
            logger = TensorBoardLogger(model_output_path / "training_logs")
            
        callbacks.append(StochasticWeightAveraging(swa_lrs=learning_rate,swa_epoch_start=5, device=device))
        
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
        
        trainer.fit(
            net,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.validation_loader
        )
        
        self.final_trainer = trainer
        self.final_net = net
        
        return trainer