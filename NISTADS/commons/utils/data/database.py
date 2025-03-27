import os
import sqlite3
import pandas as pd

from NISTADS.commons.constants import DATA_PATH, INFERENCE_PATH, VALIDATION_PATH
from NISTADS.commons.logger import logger

# [DATABASE]
###############################################################################
class AdsorptionDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'NISTADS_database.db')   
        self.inference_path = os.path.join(
            INFERENCE_PATH, 'inference_adsorption_data.csv')     
        self.configuration = configuration 
        self.initialize_database()
        self.update_database()

    #--------------------------------------------------------------------------       
    def initialize_database(self):        
        # Connect to the SQLite database and create the database if does not exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        create_single_component_table = '''
        CREATE TABLE IF NOT EXISTS SINGLE_COMPONENT_ADSORPTION (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            temperature REAL,
            adsorptionUnits TEXT,
            pressureUnits TEXT,
            adsorbent_name TEXT,
            adsorbate_name TEXT,
            pressure REAL,
            adsorbed_amount REAL,
            composition REAL            
        );
        '''

        create_binary_mixture_table = '''
        CREATE TABLE IF NOT EXISTS BINARY_MIXTURE_ADSORPTION (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            temperature REAL,
            adsorptionUnits TEXT,
            pressureUnits TEXT,
            adsorbent_name TEXT,
            compount_1 TEXT,
            compound_2 TEXT,
            compound_1_composition REAL,
            compound_2_composition REAL,
            compound_1_pressure REAL,
            compound_2_pressure REAL,
            compound_1_adsorption REAL,
            compound_2_adsorption REAL
        );
        '''

        create_adsorbates_table = '''
        CREATE TABLE IF NOT EXISTS ADSORBATES (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            InChIKey TEXT,
            name TEXT,
            InChICode TEXT,
            formula TEXT,
            synonyms TEXT,
            adsorbate_molecular_weight REAL,
            adsorbate_molecular_formula TEXT,
            adsorbate_SMILE TEXT
        );
        '''
      
        create_adsorbents_table = '''
        CREATE TABLE IF NOT EXISTS ADSORBENTS (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            hashkey TEXT,
            formula TEXT,
            synonyms TEXT,
            External_Resources TEXT
        );
        '''

        create_processed_data_table = '''
        CREATE TABLE IF NOT EXISTS PROCESSED_DATA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,            
            pressure REAL,
            adsorbed_amount REAL,
            encoded_adsorbent INTEGER,
            adsorbate_molecular_weight REAL,
            adsorbate_encoded_SMILE TEXT     
        );
        '''    

        create_inference_data_table = '''
        CREATE TABLE IF NOT EXISTS PREDICTED_ADSORPTION (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment TEXT,
            temperature REAL,            
            adsorbent_name TEXT,
            adsorbate_name TEXT,
            pressure REAL,
            adsorbed_amount REAL,
            predicted_adsorbed_amount REAL                
        );
        '''   
        
        create_checkpoints_summary_table = '''
        CREATE TABLE IF NOT EXISTS CHECKPOINTS_SUMMARY (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            checkpoint_name TEXT,
            sample_size REAL,
            validation_size REAL,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation TEXT,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile TEXT,
            jit_backend TEXT,
            device TEXT,
            device_id TEXT,
            number_of_processors INTEGER,
            use_tensorboard TEXT,
            lr_scheduler_initial_lr REAL,
            lr_scheduler_constant_steps REAL,
            lr_scheduler_decay_steps REAL
        );
        '''
        
        cursor.execute(create_single_component_table) 
        cursor.execute(create_binary_mixture_table)  
        cursor.execute(create_adsorbates_table)
        cursor.execute(create_adsorbents_table)
        cursor.execute(create_processed_data_table)    
        cursor.execute(create_inference_data_table)    
        cursor.execute(create_checkpoints_summary_table)

        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def update_database(self):               
        dataset = pd.read_csv(self.inference_path, sep=';', encoding='utf-8')        
        self.save_inference_data(dataset)
       
    #--------------------------------------------------------------------------
    def load_source_datasets(self): 
        # Connect to the database and inject a select all query
        # convert the extracted data directly into a pandas dataframe          
        conn = sqlite3.connect(self.db_path)        
        adsorption_data = pd.read_sql_query(
            f"SELECT * FROM SINGLE_COMPONENT_ADSORPTION", conn)
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
        single_components.to_sql(
            'SINGLE_COMPONENT_ADSORPTION', conn, if_exists='replace')
        binary_mixture.to_sql(
            'BINARY_MIXTURE_ADSORPTION', conn, if_exists='replace')
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
        # connect to sqlite database and save processed data as table
        conn = sqlite3.connect(self.db_path)         
        processed_data.to_sql('PROCESSED_DATA', conn, if_exists='replace')       
        conn.commit()
        conn.close()  

    #--------------------------------------------------------------------------
    def save_inference_data(self, data : pd.DataFrame): 
        # connect to sqlite database and save the inference input data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('PREDICTED_ADSORPTION', conn, if_exists='replace')
        conn.commit()
        conn.close()   

    #--------------------------------------------------------------------------
    def save_checkpoints_summary(self, data : pd.DataFrame): 
        # connect to sqlite database and save the preprocessed data as table
        conn = sqlite3.connect(self.db_path)         
        data.to_sql('CHECKPOINTS_SUMMARY', conn, if_exists='replace')
        conn.commit()
        conn.close()     
    
    

    
