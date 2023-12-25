import pandas as pd
import numpy as np
import logging
from vierlinden.config import data_path, target_filename, sensor_filename
from pytorch_forecasting import TimeSeriesDataSet
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """A class to process and prepare data for time series analysis.

    Parameters
    ----------
    load_and_process : bool, optional
        Flag to load and process data during initialization, by default False
    """
    
    def __init__(self, load_and_process: bool = False):
        """Constructor for DataProcessor."""
        
        self.processed = True if load_and_process else False
        self.data = self.load_processed_data() if load_and_process else None

    def load_processed_data(self) -> pd.DataFrame:
        """Loads processed data if available, otherwise processes raw sensor and target data.

        Returns
        -------
        pd.DataFrame
            Processed data.
        """
        
        if self.data is not None and self.processed:
            return self.data
        
        sensor_data, target_data = self.__read_data()
        all_data = DataProcessor.__merge_data(sensor_data, target_data)
        self.data = DataProcessor.__process_nan(all_data)
        
        return self.data
    
    def prepare_for_target(self, target_col: str) -> pd.DataFrame:
        """Prepares the data for modeling a specific target variable.

        Parameters
        ----------
        target_col : str
            Name of the target column.

        Returns
        -------
        pd.DataFrame
            Data prepared for the target variable.
        """
        
        t = self.data.drop(['Entleerung_RüB', 'Füllstand_RüB_1', 'Füllstand_RüB_2', 'Füllstand_RüB_3'], axis=1)
        
        # Filter out columns that contain other target variables than the one we want to predict
        if target_col == 'Kaiserstr_outflow [l/s]':
            prediction_ready_data = t.drop(['Kreuzweg_outflow [l/s]'], axis=1)
        elif target_col == 'Kreuzweg_outflow [l/s]':
            prediction_ready_data = t.drop(['Kaiserstr_outflow [l/s]'], axis=1)
        elif target_col not in self.data.columns:
            logger.error(f"Column {target_col} not found in data.")
            return
        else:
            raise ValueError('Column {target_col} is not a valid target variable.}')
        
        return prediction_ready_data
    
    def get_training_and_validation_timeseries_dataset(self, target_col: str, train_frac: float = 0.8, num_time_series: int = 1) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Generates training and validation TimeSeriesDataSet for the specified target column.

        Parameters
        ----------
        target_col : str
            Target variable column name.
        train_frac : float, optional
            Fraction of data to be used for training, by default 0.8.
        num_time_series : int, optional
            Number of time series to create, by default 1.

        Returns
        -------
        Tuple[TimeSeriesDataSet, TimeSeriesDataSet]
            Tuple containing training and validation TimeSeriesDataSet.
        """
        
        prediction_ready_data = self.prepare_for_target(target_col)
        data = self.__add_series_and_timeidx(prediction_ready_data, num_time_series)
        
        training_cutoff = int(round(data["time_idx"].max() * train_frac))
        
        # Parameters for TimeSeriesDataSet (setup by Teo)
        max_encoder_length = 24*2
        max_prediction_length = 5*2
        context_length = max_encoder_length
        prediction_length = max_prediction_length
        
        training = TimeSeriesDataSet(
            data = data[lambda x: x.time_idx <= training_cutoff],
            target_normalizer='auto',
            time_idx='time_idx',
            target=target_col,
            group_ids=['series'],
            time_varying_unknown_reals=list(set(data.columns) - {'Datetime', 'series', 'time_idx'}),
            max_encoder_length=context_length,
            min_encoder_length=max_encoder_length,
            max_prediction_length=prediction_length,
            min_prediction_length=max_prediction_length,
            allow_missing_timesteps=True
        )
        
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
        
        return training, validation
        
    
    def export_data(self, output_path: str):
        """Exports the processed data to a CSV file.

        Parameters
        ----------
        output_path : str
            Path to save the CSV file.
        """
        
        try:
            self.data.to_csv(output_path)
            logger.info(f"Data exported successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting data to {output_path}: {e}")
    
    def __read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reads sensor and target data from predefined file paths.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of sensor and target data.
        """
        
        try:
            sensor_data = pd.read_csv(data_path / sensor_filename, sep=",")
            target_data = pd.read_csv(data_path / target_filename, sep=",")
            logger.info(f"Data loaded successfully from {data_path}")
            return sensor_data, target_data
        except FileNotFoundError:
            logger.error(f"File not found, please check data path: {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

    def __merge_data(self, sensor_data: pd.DataFrame, target_data: pd.DataFrame) -> pd.DataFrame:
        """Merges sensor and target data on a common timestamp.

        Parameters
        ----------
        sensor_data : pd.DataFrame
            DataFrame containing sensor data.
        target_data : pd.DataFrame
            DataFrame containing target data.

        Returns
        -------
        pd.DataFrame
            Merged data.
        """
        
        try:
            # Convert to datetime type
            sensor_data['Datetime'] = pd.to_datetime(sensor_data['Datetime'])
            target_data['Timestamp'] = pd.to_datetime(target_data['Timestamp'])
            
            # Merge
            merged_data = pd.merge(sensor_data, target_data, left_on='Datetime', right_on='Timestamp', how='left')
            merged_data.drop(columns=['Timestamp'], inplace=True)
            return merged_data
        except Exception as e:
            logger.error(f"Error merging data: {e}")

    def __process_nan(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Processes NaN values in the dataset.

        Parameters
        ----------
        all_data : pd.DataFrame
            The dataset with potential NaN values.

        Returns
        -------
        pd.DataFrame
            Dataset with NaN values processed.
        """
        
        # Fill NaN values in Niederschlage (rainfall) column
        all_data['Niederschlag'] = all_data['Niederschlag'].fillna(0)
        
        # Fill NaN values in target data with 0
        all_data['Kaiserstr_outflow [l/s]'] = all_data['Kaiserstr_outflow [l/s]'].fillna(0)
        all_data['Kreuzweg_outflow [l/s]'] = all_data['Kreuzweg_outflow [l/s]'].fillna(0)
        
        data_removed_highnan_cols = self.__remove_high_nan_cols(all_data)
        
        # Find the first index where all columns have non-missing data
        first_valid_index = data_removed_highnan_cols.dropna().index[0]
        trimmed_data = data_removed_highnan_cols.loc[first_valid_index:, :]
        
        # Impute the missing values using linear interpolation
        # For this it is important that the timeseries which it is evenly spaced
        complete_data = trimmed_data.interpolate(method='linear')
        complete_data['Datetime'] = pd.to_datetime(complete_data['Datetime'])
        
        return complete_data
    
    def __remove_high_nan_cols(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Removes columns with a high proportion of NaN values from the dataset.

        Parameters
        ----------
        all_data : pd.DataFrame
            The dataset to be processed.

        Returns
        -------
        pd.DataFrame
            Dataset with high NaN columns removed.
        """
        
        # Remove columns with high proportion of missing values
        data_with_removed_highnans = all_data.drop(
            ["Durchfluss SWP1 und SWP2_pval",
            "FLP_Hohenstand_Pumpensumpf_pval",
            "FLP_Strom_P3_pval",
            "FLP_Strom_P4_pval",
            "FLP_Strom_P5_pval",
            "FLP_Hohenstand_Becken1_pval",
            "FLP_Hohenstand_Becken3_pval",
            "FLP_Hohenstand_Beckne2_pval"], 
            axis=1)
        
        return data_with_removed_highnans
    
    def __add_series_and_timeidx(self, data: pd.DataFrame, num_time_series: int) -> pd.DataFrame:
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