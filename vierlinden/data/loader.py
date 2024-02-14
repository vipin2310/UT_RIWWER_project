import pandas as pd
import numpy as np
import logging
from vierlinden.config import data_path, target_filename, sensor_filename
from pytorch_forecasting import TimeSeriesDataSet
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VierlindenDataProcessor:
    """A class to load, process and prepare the Vierlinden data for time series analysis.
    """
    
    def __init__(self, load_and_process: bool = False):
        """Constructor for DataProcessor.
        
        Parameters
        ----------
        load_and_process : bool
            Flag that determines whether to load and process data during initialization, by default False
        """
        
        self.processed = load_and_process
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
        all_data = self.__merge_data(sensor_data, target_data)
        self.data = self.__process_nan(all_data)
        
        logger.info("Data loaded and processed successfully.")
        
        return self.data
    
    def prepare_for_target(self, df : pd.DataFrame, target_col: str) -> pd.DataFrame:
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
        
        all_targets = ['Füllstand_RRB', 'Entleerung_RüB', 'Füllstand_RüB_1', 'Füllstand_RüB_2', 'Füllstand_RüB_3', 
                       'Kaiserstr_outflow [l/s]', 'Kreuzweg_outflow [l/s]']
        
        # Filter out columns that contain other target variables than the one we want to predict
        if target_col not in all_targets:
            raise ValueError('Column {target_col} is not a valid target variable.}')
        else:
            prediction_ready_data = df.copy().drop([col for col in all_targets if col != target_col], axis=1)
        
        return prediction_ready_data
    
    @staticmethod
    def split_data(df : pd.DataFrame, train_frac : float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test set.

        Parameters
        ----------
        train_size : float, optional
            Fraction of data to use for training, by default 0.9

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of training and validation data.
        """
        
        training_cutoff = int(round(len(df) * train_frac))
        
        train_data = df[:training_cutoff].copy()
        test_data = df[training_cutoff:].copy()
        
        return train_data, test_data
    
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
            
    def plot_target_col(self, target_col: str):
        """Plots the target variable against Datetime.

        Parameters
        ----------
        target_col : str
            Name of the target column.
        """
        
        self.data.plot(x='Datetime', y=target_col, figsize=(10, 6))
    
    def apply_overflow_equation_to_target(self, target_col: str) -> pd.DataFrame:
        """Applies the weir equation to the target variable.

        Parameters
        ----------
        target_col : str
            Name of the target column.

        Returns
        -------
        pd.DataFrame
            Data with the weir equation applied to the target variable.
        """
        
        all_targets = ['Entleerung_RüB', 'Füllstand_RüB_1', 'Füllstand_RüB_2', 'Füllstand_RüB_3']
        
        # Apply weir equation to the target variable if it is a valid target column
        if target_col not in all_targets:
            raise ValueError('Column {target_col} is not a valid target variable.}')
        else:
            name = target_col + "_outflow [l/s]"
            self.data[name] = self.data[target_col].apply(self.__apply_weir_equation)
        
        return self.data
    
    def __apply_weir_equation(self, h : float) -> float:
        """
        Copied from Peter's (Okeanos) repo.
        
        This function calculates outflow in l/s from RRK, RRB and RÜB, based on rectangular Weir hydraulic equation.
        Q = Cd * (2/3) * (2g)^(1/2) * b * h^(3/2)
        source ISO1438, page 11
        where:
        Q: overflow in m3/s
        cd = 0.68: losses coefficient (no unit)
        g = 9.81: gravity acceleration in m/s2
        b = 1: weir depth in m
        h: overflow depth in m

        Parameters
        ----------
        h : float
            Value of water level (RRK, RRB or RÜB).

        Returns
        -------
        float
            The overflow in l/s.
        """
        
        # Weir parameters, see docstring for details
        cd = 0.68
        g = 9.81
        b = 1
        
        q = cd * (2/3) * ((2*g)**(0.5)) * b * h**(1.5)
        q_liter_per_sec = q * 1000 # from m3/s to l/s
        
        return q_liter_per_sec
    
    def __read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Reads sensor and target data from predefined file paths.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of sensor and target data.
        """
        
        try:
            sensor_data = pd.read_csv(data_path + "/" + sensor_filename, sep=",")
            target_data = pd.read_csv(data_path + "/" +  target_filename, sep=",")
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
            logger.info("Sensor and target data merged successfully.")
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
        
        logger.info("NaN values processed successfully.")
        
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