import os
import sqlite3
import pandas as pd

from NISTADS.commons.constants import DATA_PATH, INFERENCE_PATH
from NISTADS.commons.logger import logger


###############################################################################
class SingleComponentAdsorptionTable:

    def __init__(self):
        self.name = 'SINGLE_COMPONENT_ADSORPTION'
        self.dtypes = {            
            'filename': 'VARCHAR',
            'temperature': 'FLOAT',
            'adsorptionUnits': 'VARCHAR',
            'pressureUnits': 'VARCHAR',
            'adsorbent_name': 'VARCHAR',
            'adsorbate_name': 'VARCHAR',
            'pressure': 'FLOAT',
            'adsorbed_amount': 'FLOAT',
            'composition': 'FLOAT'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            filename VARCHAR,
            temperature FLOAT,
            adsorptionUnits VARCHAR,
            pressureUnits VARCHAR,
            adsorbent_name VARCHAR,
            adsorbate_name VARCHAR,
            pressure FLOAT,
            adsorbed_amount FLOAT,
            composition FLOAT            
        );
        '''          
      
        cursor.execute(query)


###############################################################################
class BinaryMixtureAdsorptionTable:

    def __init__(self):
        self.name = 'BINARY_MIXTURE_ADSORPTION'
        self.dtypes = {            
            'filename': 'VARCHAR',
            'temperature': 'FLOAT',
            'adsorptionUnits': 'VARCHAR',
            'pressureUnits': 'VARCHAR',
            'adsorbent_name': 'VARCHAR',
            'compount_1': 'VARCHAR',
            'compound_2': 'VARCHAR',
            'compound_1_composition': 'FLOAT',
            'compound_2_composition': 'FLOAT',
            'compound_1_pressure': 'FLOAT',
            'compound_2_pressure': 'FLOAT',
            'compound_1_adsorption': 'FLOAT',
            'compound_2_adsorption': 'FLOAT'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            filename VARCHAR,
            temperature FLOAT,
            adsorptionUnits VARCHAR,
            pressureUnits VARCHAR,
            adsorbent_name VARCHAR,
            compount_1 VARCHAR,
            compound_2 VARCHAR,
            compound_1_composition FLOAT,
            compound_2_composition FLOAT,
            compound_1_pressure FLOAT,
            compound_2_pressure FLOAT,
            compound_1_adsorption FLOAT,
            compound_2_adsorption FLOAT            
        );
        '''
        cursor.execute(query)
    
    
###############################################################################
class AdsorbatesDataTable:

    def __init__(self):
        self.name = 'ADSORBATES'
        self.dtypes = {            
            'InChIKey': 'VARCHAR',
            'name': 'VARCHAR',
            'InChICode': 'VARCHAR',
            'formula': 'VARCHAR',
            'synonyms': 'VARCHAR',
            'adsorbate_molecular_weight': 'FLOAT',
            'adsorbate_molecular_formula': 'VARCHAR',
            'adsorbate_SMILE': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            InChIKey VARCHAR,
            name VARCHAR,
            InChICode VARCHAR,
            formula VARCHAR,
            synonyms VARCHAR,
            adsorbate_molecular_weight FLOAT,
            adsorbate_molecular_formula VARCHAR,
            adsorbate_SMILE VARCHAR            
        );
        '''
        cursor.execute(query) 

    
###############################################################################
class AdsorbentsDataTable:

    def __init__(self):
        self.name = 'ADSORBENTS'
        self.dtypes = {            
            'name': 'VARCHAR',
            'hashkey': 'VARCHAR',
            'formula': 'VARCHAR',
            'synonyms': 'VARCHAR',
            'External_Resources': 'VARCHAR',
            'adsorbent_molecular_weight': 'FLOAT',
            'adsorbent_molecular_formula': 'VARCHAR',
            'adsorbent_SMILE': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            name VARCHAR,
            hashkey VARCHAR,
            formula VARCHAR,
            synonyms VARCHAR,
            External_Resources VARCHAR
            adsorbent_molecular_weight FLOAT,
            adsorbent_molecular_formula VARCHAR,
            adsorbent_SMILE VARCHAR     
        );
        '''
        cursor.execute(query)
    
    
###############################################################################
class TrainDataTable:

    def __init__(self):
        self.name = 'TRAIN_DATA'
        self.dtypes = {            
            'temperature': 'FLOAT',
            'pressure': 'FLOAT',
            'adsorbed_amount': 'FLOAT',
            'encoded_adsorbent': 'INTEGER',
            'adsorbate_molecular_weight': 'FLOAT',
            'adsorbate_name': 'VARCHAR',
            'adsorbate_encoded_SMILE': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            temperature FLOAT,            
            pressure FLOAT,
            adsorbed_amount FLOAT,
            encoded_adsorbent INTEGER,
            adsorbate_molecular_weight FLOAT,
            adsorbate_name VARCHAR,  
            adsorbate_encoded_SMILE VARCHAR             
        );
        '''
        cursor.execute(query)


###############################################################################
class ValidationDataTable:

    def __init__(self):
        self.name = 'VALIDATION_DATA'
        self.dtypes = {            
            'temperature': 'FLOAT',
            'pressure': 'FLOAT',
            'adsorbed_amount': 'FLOAT',
            'encoded_adsorbent': 'INTEGER',
            'adsorbate_molecular_weight': 'FLOAT',
            'adsorbate_name': 'VARCHAR',
            'adsorbate_encoded_SMILE': 'VARCHAR'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            temperature FLOAT,            
            pressure FLOAT,
            adsorbed_amount FLOAT,
            encoded_adsorbent INTEGER,
            adsorbate_molecular_weight FLOAT,
            adsorbate_name VARCHAR,  
            adsorbate_encoded_SMILE VARCHAR             
        );
        '''
        cursor.execute(query)

    
###############################################################################
class PredictedAdsorptionTable:

    def __init__(self):
        self.name = 'PREDICTED_ADSORPTION'
        self.dtypes = {            
            'experiment': 'VARCHAR',
            'temperature': 'FLOAT',
            'adsorbent_name': 'VARCHAR',
            'adsorbate_name': 'VARCHAR',
            'pressure': 'FLOAT',
            'adsorbed_amount': 'FLOAT',
            'predicted_adsorbed_amount': 'FLOAT'}

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            experiment VARCHAR,
            temperature FLOAT,            
            adsorbent_name VARCHAR,
            adsorbate_name VARCHAR,
            pressure FLOAT,
            adsorbed_amount FLOAT,
            predicted_adsorbed_amount FLOAT               
        );
        '''
        cursor.execute(query)   
    

###############################################################################
class CheckpointSummaryTable:

    def __init__(self):
        self.name = 'CHECKPOINTS_SUMMARY'
        self.dtypes = {
            'checkpoint_name': 'VARCHAR',
            'sample_size': 'FLOAT',
            'validation_size': 'FLOAT',
            'seed': 'INTEGER',
            'precision_bits': 'INTEGER',
            'epochs': 'INTEGER',
            'additional_epochs': 'INTEGER',
            'batch_size': 'INTEGER',
            'split_seed': 'INTEGER',
            'image_augmentation': 'VARCHAR',
            'image_height': 'INTEGER',
            'image_width': 'INTEGER',
            'image_channels': 'INTEGER',
            'jit_compile': 'VARCHAR',
            'jit_backend': 'VARCHAR',
            'device': 'VARCHAR',
            'device_id': 'VARCHAR',
            'number_of_processors': 'INTEGER',
            'use_tensorboard': 'VARCHAR',
            'lr_scheduler_initial_lr': 'FLOAT',
            'lr_scheduler_constant_steps': 'FLOAT',
            'lr_scheduler_decay_steps': 'FLOAT'}    

    #--------------------------------------------------------------------------
    def get_dtypes(self):
        return self.dtypes
    
    #--------------------------------------------------------------------------
    def create_table(self, cursor):
        query = f'''
        CREATE TABLE IF NOT EXISTS {self.name} (            
            checkpoint_name VARCHAR,
            sample_size FLOAT,
            validation_size FLOAT,
            seed INTEGER,
            precision_bits INTEGER,
            epochs INTEGER,
            additional_epochs INTEGER,
            batch_size INTEGER,
            split_seed INTEGER,
            image_augmentation VARCHAR,
            image_height INTEGER,
            image_width INTEGER,
            image_channels INTEGER,
            jit_compile VARCHAR,
            jit_backend VARCHAR,
            device VARCHAR,
            device_id VARCHAR,
            number_of_processors INTEGER,
            use_tensorboard VARCHAR,
            lr_scheduler_initial_lr FLOAT,
            lr_scheduler_constant_steps FLOAT,
            lr_scheduler_decay_steps FLOAT
            );
            ''' 
         
        cursor.execute(query)     
    

# [DATABASE]
###############################################################################
class AdsorptionDatabase:

    def __init__(self, configuration):             
        self.db_path = os.path.join(DATA_PATH, 'NISTADS_database.db')   
        self.inference_path = os.path.join(
            INFERENCE_PATH, 'inference_adsorption_data.csv')     
        self.configuration = configuration 
        self.single_component = SingleComponentAdsorptionTable()
        self.binary_mixture = BinaryMixtureAdsorptionTable()
        self.adsorbates = AdsorbatesDataTable()
        self.adsorbents = AdsorbentsDataTable()
        self.train_data = TrainDataTable()
        self.validation_data = ValidationDataTable()
        self.inference_data = PredictedAdsorptionTable()       
        self.checkpoints_summary = CheckpointSummaryTable()    
        
    #--------------------------------------------------------------------------       
    def initialize_database(self): 
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        self.single_component.create_table(cursor)  
        self.binary_mixture.create_table(cursor)
        self.adsorbates.create_table(cursor)  
        self.adsorbents.create_table(cursor)  
        self.train_data.create_table(cursor)
        self.validation_data.create_table(cursor)
        self.inference_data.create_table(cursor)        
        self.checkpoints_summary.create_table(cursor)   

        conn.commit()
        conn.close()

    #--------------------------------------------------------------------------
    def update_database(self):               
        dataset = pd.read_csv(self.inference_path, sep=';', encoding='utf-8')        
        self.save_predictions_table(dataset)
       
    #--------------------------------------------------------------------------
    def load_source_data_table(self):          
        conn = sqlite3.connect(self.db_path)        
        adsorption_data = pd.read_sql_query(
            f"SELECT * FROM {self.single_component.name}", conn)
        guest_data = pd.read_sql_query(
            f"SELECT * FROM {self.adsorbates.name}", conn)
        host_data = pd.read_sql_query(
            f"SELECT * FROM {self.adsorbents.name}", conn)
        conn.close()  

        return adsorption_data, guest_data, host_data

    #--------------------------------------------------------------------------
    def load_train_and_validation_tables(self):       
        conn = sqlite3.connect(self.db_path)        
        train_data = pd.read_sql_query(
            f"SELECT * FROM {self.train_data.name}", conn)
        validation_data = pd.read_sql_query(
            f"SELECT * FROM {self.validation_data.name}", conn)
        conn.close()  

        return train_data, validation_data  

    #--------------------------------------------------------------------------
    def load_inference_data_table(self):         
        conn = sqlite3.connect(self.db_path)         
        data = pd.read_sql_query(
            f"SELECT * FROM {self.inference_data.name}", conn)
        conn.commit()
        conn.close() 

        return data      

    #--------------------------------------------------------------------------
    def save_experiments_table(self, single_components,
                               binary_mixture):        
        conn = sqlite3.connect(self.db_path)         
        single_components.to_sql(
            self.single_component.name, conn, if_exists='replace', index=False,
            dtype=self.single_component.get_dtypes())
        binary_mixture.to_sql(
            self.binary_mixture.name, conn, if_exists='replace', index=False,
            dtype=self.binary_mixture.get_dtypes())           
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_materials_table(self, adsorbates, adsorbents):    
        conn = sqlite3.connect(self.db_path)
        if adsorbates is not None:         
            adsorbates.to_sql(
                self.adsorbates.name, conn, if_exists='replace', index=False,
                dtype=self.adsorbates.get_dtypes())
        if adsorbents is not None:
            adsorbents.to_sql(
                self.adsorbents.name, conn, if_exists='replace', index=False,
                dtype=self.adsorbents.get_dtypes())
        conn.commit()
        conn.close() 

    #--------------------------------------------------------------------------
    def save_train_and_validation_tables(self, train_data, validation_data):         
        conn = sqlite3.connect(self.db_path)         
        train_data.to_sql(
            self.train_data.name, conn, if_exists='replace', index=False,
            dtype=self.train_data.get_dtypes())  
        validation_data.to_sql(
            self.validation_data.name, conn, if_exists='replace', index=False,
            dtype=self.validation_data.get_dtypes())    
        conn.commit()
        conn.close()  

    #--------------------------------------------------------------------------
    def save_predictions_table(self, data):      
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.inference_data.name, conn, if_exists='replace', index=False,
            dtype=self.inference_data.get_dtypes())
        conn.commit()
        conn.close()  

    #--------------------------------------------------------------------------
    def save_checkpoints_summary_table(self, data):         
        conn = sqlite3.connect(self.db_path)         
        data.to_sql(
            self.checkpoints_summary.name, conn, if_exists='replace', index=False,
            dtype=self.checkpoints_summary.get_dtypes())
        conn.commit()
        conn.close()   
    
    

    
