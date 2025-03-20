import os
import sqlite3
import pandas as pd

from NISTADS.commons.constants import DATA_PATH, VALIDATION_PATH
from NISTADS.commons.logger import logger

# [DATABASE]
###############################################################################
class AdsorptionDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'NISTADS_database.db')        
        self.configuration = configuration
       
    #--------------------------------------------------------------------------
    def load_source_datasets(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        adsorption_data = pd.read_sql_query(f"SELECT * FROM SINGLE_COMPONENT_ADSORPTION", conn)
        guest_data = pd.read_sql_query(f"SELECT * FROM ADSORBATES", conn)
        host_data = pd.read_sql_query(f"SELECT * FROM ADSORBENTS", conn)
        conn.close()  

        return adsorption_data, guest_data, host_data

    #--------------------------------------------------------------------------
    def load_processed_data(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM PROCESSED_DATA", conn)
        conn.close()  

        return data       

    #--------------------------------------------------------------------------
    def save_experiments_table(self, single_components : pd.DataFrame,
                               binary_mixture : pd.DataFrame): 
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)         
        single_components.to_sql('SINGLE_COMPONENT_ADSORPTION', conn, if_exists='replace')
        binary_mixture.to_sql('BINARY_MIXTURE_ADSORPTION', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_materials_table(self, adsorbates : pd.DataFrame, adsorbents : pd.DataFrame):                               
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)
        if adsorbates is not None:         
            adsorbates.to_sql('ADSORBATES', conn, if_exists='replace')
        if adsorbents is not None:
            adsorbents.to_sql('ADSORBENTS', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_processed_data_table(self, processed_data : pd.DataFrame): 
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)         
        processed_data.to_sql('PROCESSED_DATA', conn, if_exists='replace')       
        conn.commit()
        conn.close()    

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('CHECKPOINTS_SUMMARY', conn, if_exists='replace')
        conn.commit()
        conn.close()     
    
    

    
