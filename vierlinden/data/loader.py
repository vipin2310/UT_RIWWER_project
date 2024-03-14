import pandas as pd
import numpy as np
import logging
from vierlinden.config import data_path, data_filename
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
        
        all_data = self.__read_data()
        all_data = self.__process_datetime(all_data)
        self.data = self.__process_nan(all_data)
        
        self.processed = True
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
        
        # TODO: Add preparation if needed
        
        return df
    
    @staticmethod
    def split_data(df : pd.DataFrame, split_date : str = '', train_frac : float = 0.9) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into training and test set. (Not random)

        Parameters
        ----------
        split_date : str, optional
            The cutoff date for the split, by default ''. Will be used if not empty. Splitting asuming the data is sorted by date.
        train_size : float, optional
            Fraction of data to use for training, by default 0.9. Will only be used if split_date is empty.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of training and validation data.
        """
        
        if split_date != '':
            split_date = pd.to_datetime(split_date)
            training_cutoff = df[df['Datetime'] < split_date].shape[0]
        else:
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
            
    def plot_target_col(self, column: str):
        """Plots a column against Datetime.

        Parameters
        ----------
        column : str
            Name of column.
        """
        
        self.data.plot(x='Datetime', y=column, figsize=(10, 6))
    
    def remove_nan_flag_cols(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Removes columns that were added to indicate filled missing values, identified by the '_was_nan' suffix.
    
        Parameters
        ----------
        all_data : pd.DataFrame
            The DataFrame from which to remove the nan indicator columns.
    
        Returns
        -------
        pd.DataFrame
            The DataFrame without the nan indicator columns.
        """
        # List comprehension to find columns with '_was_nan' suffix
        columns_to_remove = [col for col in all_data.columns if col.endswith('_was_nan')]
    
        # Drop these columns from the DataFrame
        all_data_cleaned = all_data.copy().drop(columns=columns_to_remove)
    
        return all_data_cleaned
    
    def apply_overflow_statistic_to_column(self, column: str) -> pd.DataFrame:
        """Applies the overflow statistics to a column of the data.
        Its meant to take the 90th percentile as threshold and count above that as overflow.
        After that the weir equation is applied to the overflow to calculate outflow in l/s.

        Parameters
        ----------
        column : str
            Name of the column.

        Returns
        -------
        pd.DataFrame
            Data with the weir equation applied to the target variable.
        """
        
        name = column + "_outflow [l/s]"
        self.data[name] = self.data[column].apply(self.__weir_equation)
        
        return self.data
    
    def __weir_equation(self, h : float) -> float:
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
    
    def __read_data(self) -> pd.DataFrame:
        """Reads data from predefined file path.

        Returns
        -------
        pd.DataFrame
            Data
        """
        
        filepath = data_path + "/" + data_filename
        
        try:
            data = pd.read_csv(filepath, sep=",")
            logger.info(f"Data loaded successfully from {data_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File {filepath} not found, please check data path: {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            
    def __process_datetime(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Processes the datetime column in the dataset.

        Parameters
        ----------
        all_data : pd.DataFrame
            The dataset with a datetime column.

        Returns
        -------
        pd.DataFrame
            Dataset with processed datetime column.
        """
        
        df = all_data.copy()
        
        # Convert datetime column to datetime type
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        logger.info("Datetime processed successfully.")
        
        return df

    def __process_nan(self, all_data: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values by interpolating, and for initial and trailing NaNs, uses backward and forward fill respectively.
        Adds a column if the missing value was filled, indicating the fill operation.
                
        Parameters
        ----------
        all_data : pd.DataFrame
            DataFrame containing the data to process.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled and additional columns for interpolation flags.
        """
        # Make a copy of the data to avoid modifying the original DataFrame
        df = all_data.copy()
        
        # Iterate over each column in the DataFrame
        for col in df.columns:
            # Check if the column contains any missing values
            if df[col].isnull().any():
                # Create a boolean flag column to indicate where NaNs were before filling
                flag_column_name = f"{col}_was_nan"
                df[flag_column_name] = df[col].isnull()
                
                # Interpolate missing values
                df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
                
                # Then apply backward fill for remaining NaNs at the start
                df[col] = df[col].bfill()
                
                # Then apply forward fill for remaining NaNs at the end
                df[col] = df[col].ffill()
                
                # Ensure the flag column is inserted right after the current column
                # Find current column index
                col_index = df.columns.get_loc(col)
                # Move the flag column to the right position
                cols = df.columns.tolist()
                # Remove the flag column from its original position and insert it after the current column
                cols.insert(col_index + 1, cols.pop(cols.index(flag_column_name)))
                df = df[cols]
                
        return df