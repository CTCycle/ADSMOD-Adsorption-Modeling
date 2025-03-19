import os
import sqlite3
import pandas as pd

from NISTADS.commons.constants import PROCESSED_PATH
from NISTADS.commons.logger import logger

# [DATABASE]
###############################################################################
class AdsorptionDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(PROCESSED_PATH, 'NISTADS_dataset.csv')        
        self.configuration = configuration

    #--------------------------------------------------------------------------
    def load_source_datasets(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        data = pd.read_sql_query(f"SELECT * FROM Processed_data", conn)
        conn.close()  

        return data
    
    #--------------------------------------------------------------------------
    def load_processed_data(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        adsorption_data = pd.read_sql_query(f"SELECT * FROM Processed_data", conn)
        guest_data = pd.read_sql_query(f"SELECT * FROM Adsorbates", conn)
        host_data = pd.read_sql_query(f"SELECT * FROM Adsorbents", conn)
        conn.close()  

        return adsorption_data, guest_data, host_data       

    #--------------------------------------------------------------------------
    def save_experiments_table(self, single_components : pd.DataFrame,
                               binary_mixture : pd.DataFrame): 
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)         
        single_components.to_sql('Single_components', conn, if_exists='replace')
        binary_mixture.to_sql('Binary_mixture', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_materials_table(self, adsorbates : pd.DataFrame, adsorbents : pd.DataFrame):                               
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)
        if adsorbates is not None:         
            adsorbates.to_sql('Adsorbates', conn, if_exists='replace')
        if adsorbents is not None:
            adsorbents.to_sql('Adsorbents', conn, if_exists='replace')
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_processed_data_table(self, processed_data : pd.DataFrame): 
        # connect to sqlite database and save adsorption data in different tables
        # one for single components, and the other for binary mixture experiments
        conn = sqlite3.connect(self.db_path)         
        processed_data.to_sql('Processed_data', conn, if_exists='replace')       
        conn.commit()
        conn.close() 
        
    
    

    
    