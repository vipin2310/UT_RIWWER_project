import pandas as pd
import numpy as np
import logging
from vierlinden.config import data_path, target_filename, sensor_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, load_and_process=False):
        
        self.processed = True if load_and_process else False
        self.data = self.load_processed_data() if load_and_process else None

    def load_processed_data(self):
        
        if self.data is not None and self.processed:
            return self.data
        
        sensor_data, target_data = self.__read_data()
        all_data = DataProcessor.__merge_data(sensor_data, target_data)
        self.data = DataProcessor.__process_nan(all_data)
        
        return self.data
    
    def prepare_for_target(self, target_col):
        
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
    
    def get_training_and_validation_timeseries_dataset(self, target_col, train_frac=0.8, num_time_series = 1):
        
        prediction_ready_data = self.prepare_for_target(target_col)
        data = self.__add_series_and_timeidx(prediction_ready_data, num_time_series)
        
        # TODO Create dataloader
        
    
    def export_data(self, output_path):
        """Export data to a file."""
        try:
            self.data.to_csv(output_path)
            logger.info(f"Data exported successfully to {output_path}")
        except Exception as e:
            logger.error(f"Error exporting data to {output_path}: {e}")
    
    def __read_data(self):
        try:
            sensor_data = pd.read_csv(data_path / sensor_filename, sep=",")
            target_data = pd.read_csv(data_path / target_filename, sep=",")
            logger.info(f"Data loaded successfully from {data_path}")
            return sensor_data, target_data
        except FileNotFoundError:
            logger.error(f"File not found, please check data path: {data_path}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")

    def __merge_data(self, sensor_data, target_data):
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

    def __process_nan(self, all_data):
        
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
    
    def __remove_high_nan_cols(self, all_data):
        
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
    
    def __add_series_and_timeidx(self, data, num_time_series):
        
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