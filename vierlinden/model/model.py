from contextlib import contextmanager
from io import StringIO
import sys
from typing import Tuple
from matplotlib.ticker import MaxNLocator
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
from pytorch_forecasting.models.base_model import Prediction
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar,StochasticWeightAveraging, Callback
from lightning.pytorch.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from pytorch_forecasting.metrics import QuantileLoss, MAE, MASE, RMSE

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
        data = TimeSeriesDataSetCreator.add_series_and_timeidx(data, num_time_series)
        
        training_cutoff = int(round(data["time_idx"].max() * train_frac))
        if train_frac > 0:
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
        elif train_frac == 0:
            training = None
            validation = TimeSeriesDataSet(
                data = data,
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
        elif train_frac == 1:
            raise NotImplementedError("Train fraction of 1 is not yet implemented.")
        
        return training, validation
    
    @staticmethod
    def add_series_and_timeidx(data: pd.DataFrame, num_time_series: int) -> pd.DataFrame:
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
        data['time_idx'] = data.index - data.index.min() + 1
        
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
              loss = MAE(),
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
            loss=loss,
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
    
    @staticmethod
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
    
    def __init__(self, trained_model : NHiTS, context_length : int, prediction_length : int, target_col : str):
        
        self.model = trained_model
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_col = target_col
        self.model.eval()
        
    def predict(self, dataframe : pd.DataFrame):
        
        # Assuming a single time series for prediction TODO make this configurable
        num_time_series = 1
        
        prepared_df = self.__prepare_for_pred(dataframe)
        
        _, test_set = TimeSeriesDataSetCreator.create(
                                prepared_df, 
                                target_col=self.target_col,
                                context_length=self.context_length,
                                prediction_length=self.prediction_length, # Prediction is now evaluated on entire data but is it still correct?
                                train_frac=0,
                                num_time_series=num_time_series
                            )

        # Create DataLoader from prediction_set
        test_loader = test_set.to_dataloader(train=False, batch_size=1, num_workers=0)
        
        self.raw_predictions = self.model.predict(test_loader,
                                    mode="raw",
                                    return_x=True,
                                    return_y=True,
                                    return_index = True,
                                    return_decoder_lengths=True)
        
        self.predict_resultdf = self.__get_result_df(dataframe, self.raw_predictions, num_time_series)
        return self.predict_resultdf.copy()
    
    def plot_forecast_for_all(self, forecast_step_ahead : int = 1, plot_forecast_distribution : bool = True):

        if self.predict_resultdf is None:
            raise Exception("No prediction result available. Please use predict() first.")
        
        # Get the dates, actual values and forecasted values
        dates = self.predict_resultdf['Datetime']
        actual_values = self.predict_resultdf[self.target_col]
        forecast_series = self.predict_resultdf['Predicted Forecast']
        forecast_series.index = dates

        # Filter out NaN values and unpack the forecast lists
        valid_forecast_indices = forecast_series.dropna().index
        forecasted_values = forecast_series.dropna().tolist()

        # Determine the frequency of the DataFrame
        date_diff = dates.diff().min()

        # Step ahead forecast values and dates
        step_ahead_values = [forecast[forecast_step_ahead - 1] for forecast in forecasted_values]
        step_ahead_dates = valid_forecast_indices + forecast_step_ahead * date_diff

        # Min, max, and quantile calculations
        min_values = np.min(np.array(forecasted_values), axis=1)
        max_values = np.max(np.array(forecasted_values), axis=1)
        lower_quantile = 0.25
        upper_quantile = 0.75
        quantile_values_lower = np.quantile(np.array(forecasted_values), lower_quantile, axis=1)
        quantile_values_upper = np.quantile(np.array(forecasted_values), upper_quantile, axis=1)

        # Forecast dates for the calculations
        forecast_dates = valid_forecast_indices + forecast_step_ahead * date_diff
        
        # Plotting
        plt.figure(figsize=(10, 6))

        if plot_forecast_distribution:
            # Min and max range
            plt.plot(forecast_dates, min_values, color='orange', linestyle='--', linewidth=0.2)
            plt.plot(forecast_dates, max_values, color='orange', linestyle='--', linewidth=0.2)
            min_max_range = plt.fill_between(forecast_dates, min_values, max_values, color='orange', alpha=0.2)

            # Interquantile range
            plt.plot(forecast_dates, quantile_values_lower, color='orange', linestyle='--', linewidth=0.2)
            plt.plot(forecast_dates, quantile_values_upper, color='orange', linestyle='--', linewidth=0.2)
            interquantile_range = plt.fill_between(forecast_dates, quantile_values_lower, quantile_values_upper, color='orange', alpha=0.4)

        # Actual values and step ahead forecast
        plt.plot(dates, actual_values, label='Actual Values', color='blue', linewidth=1)
        plt.plot(step_ahead_dates, step_ahead_values, color='red', label=f'Forecast Step {forecast_step_ahead}', linewidth=1)
    
        # Set the maximum number of x-axis ticks to 5
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Adjusting the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        
        if plot_forecast_distribution:
            legend_items = [handles[0], handles[1], min_max_range, interquantile_range]
            legend_labels = ["Actual Values", f'Forecast {forecast_step_ahead} step ahead', "Min-Max Range", "Interquartile Range"]
        else:
            legend_items = [handles[0], handles[1]]
            legend_labels = ["Actual Values", f'Forecast {forecast_step_ahead} step ahead']
            
        plt.legend(legend_items, legend_labels)

        plt.title('Actual Values with Forecast Ranges')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(False)
        plt.show()
    
    def plot_forecast_per_time_idx(self, time_idx, max_forecast_horizon : int = -1):
        
        if self.predict_resultdf is None:
            raise Exception("No prediction result available. Please use predict() first.")
        
        # Get the dates, actual values and forecasted values
        dates = self.predict_resultdf['Datetime']
        actual_values = self.predict_resultdf[self.target_col]
        forecast_series = self.predict_resultdf['Predicted Forecast']
        actual_values.index = dates
        forecast_series.index = dates
        
        # Filter out NaN values and unpack the forecast lists
        valid_forecast_indices = forecast_series.dropna().index
        forecasted_values = forecast_series.dropna().tolist()
        
        # Get forecast for time_idx
        date = dates[time_idx]
        
        if date not in valid_forecast_indices:
            raise Exception(f"Date {date} is not in the valid forecast indices, maybe there weren't {self.context_length} values before it?")
        
        if max_forecast_horizon == -1:
            max_forecast_horizon = self.prediction_length
        elif max_forecast_horizon > self.prediction_length:
            raise Exception(f"Max forecast horizon {max_forecast_horizon} is larger than the prediction length {self.prediction_length}.")
        
        start_date = dates[time_idx - self.context_length + 1]
        end_date = dates[time_idx + max_forecast_horizon]
        
        actual_values = actual_values[start_date:end_date]
        actual_dates = pd.date_range(start=start_date, periods=len(actual_values), freq=dates.diff().min())
        forecast = forecasted_values[time_idx]
        forecast_dates = pd.date_range(start=date, periods=len(forecast) + 1, freq=dates.diff().min())[1:]
        
        forecast = pd.Series(forecast, index=forecast_dates)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        plt.plot(actual_dates, actual_values, label='Actual Values', color='blue', linewidth=1)
        plt.plot(forecast_dates, forecast, color='red', label=f'Forecast', linewidth=1)
        
        # Set the maximum number of x-axis ticks to 5
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
        
        plt.title(f'Actual Values with Forecast Values for time_idx {time_idx}')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()
    
    def get_average_mae_loss(self):
        
        
        
        raise NotImplementedError("Not yet implemented.")
    
    def __get_result_df(self, dataframe : pd.DataFrame, raw_predictions : Prediction, num_time_series : int):
        
        result_df = TimeSeriesDataSetCreator.add_series_and_timeidx(dataframe, num_time_series)
        predictions = self.__get_predictions_from_raw(raw_predictions)
        result_df = result_df.merge(predictions, on=["time_idx", "series"], how="left")
        
        return result_df.drop(columns=["series", "time_idx"])

    def __get_predictions_from_raw(self, raw_predictions : Prediction) -> pd.DataFrame:
        
        predictions = raw_predictions.output.prediction.detach().cpu().numpy()[:,:,0]
        predictions = [row.tolist() for row in predictions]
        
        time_idx = raw_predictions.index["time_idx"]
        series = raw_predictions.index["series"]
        
        return pd.DataFrame({"Predicted Forecast": predictions, 
                             "time_idx" : time_idx, 
                             "series" : series})
        
        
    def __prepare_for_pred(self, dataframe : pd.DataFrame) -> pd.DataFrame:
        
        # Calculate the frequency of the dates in the input DataFrame
        frequency = dataframe['Datetime'].iloc[1] - dataframe['Datetime'].iloc[0]

        # Generate new dates that will be predicted
        new_dates = pd.date_range(start=dataframe['Datetime'].iloc[-1], periods=self.prediction_length, freq=frequency)[1:]
        
        # Create new DataFrame with the same columns as the input DataFrame and the new dates as the datetime column
        new_df = pd.DataFrame(index=np.arange(len(new_dates)), columns=dataframe.columns)
        new_df['Datetime'] = new_dates
        
        # Fillna because otherwise pytorch forecasting complains with an error
        new_df.fillna(0, inplace=True)

        # Concatenate the new DataFrame with the original DataFrame
        return pd.concat([dataframe, new_df]).reset_index(drop=True)