import re
import numpy as np
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm


from NISTADS.app.src.commons.interface.workers import check_thread_status, update_progress_callback
from NISTADS.app.src.commons.constants import DATA_PATH
from NISTADS.app.src.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:

    def __init__(self, configuration): 
        self.molecular_identifier = 'InChIKey'                 
        self.configuration = configuration   

    # Define a function to handle duplicates, keeping rows with InChIKey
    #--------------------------------------------------------------------------
    def remove_duplicates_without_identifiers(self, data : pd.DataFrame):
        if self.molecular_identifier in data.columns:
            data = (data.assign(_has_id=data['InChIKey'].notna())
                    .sort_values(['name', '_has_id'], ascending=[True, False])
                    .drop_duplicates('name')
                    .drop('_has_id', axis=1))
        else:
            data = data.drop_duplicates('name')
        return data  

    #--------------------------------------------------------------------------
    def map_fetched_properties(self, data : pd.DataFrame, properties : dict):
        if not properties:
            return 
        
        # set all names to lowcase to avoid mismatch
        properties['name'] = [x.lower() for x in properties['name']]
        data['name'] = data['name'].str.lower() 
        # create indexed dataframes using the column name as index 
        indexed_properties = pd.DataFrame(properties).set_index('name')
        indexed_data = data.set_index('name')        
        # update the dataset and reset the index to avoid using names
        indexed_data.update(indexed_properties) 
        dataset = indexed_data.reset_index()

        return dataset    
    
    #--------------------------------------------------------------------------
    def fetch_guest_properties(self, experiments : pd.DataFrame, data : pd.DataFrame, **kwargs):
        guest_properties = GuestProperties(self.configuration) 
        # Combine guest names from experiments and data, cleaning them to ensure consistency
        guest_names = pd.concat([
            experiments['adsorbate_name'].dropna(),
            data['name'].dropna()
            ]).astype(str).str.strip().str.lower().unique()        
        
        # fetch adsorbates molecular properties using pubchem API
        all_guests = pd.DataFrame(guest_names, columns=['name'])                   
        properties = guest_properties.get_properties_for_multiple_compounds(
            all_guests, worker=kwargs.get('worker', None),
            progress_callback=kwargs.get('progress_callback', None))

        # build the enriched dataset using the extracted molecular properties        
        dataset = self.map_fetched_properties(data, properties)

        return dataset
    
    # this function is not used in the current version of the code, since it is 
    # difficult to find a reliable source for the adsorbent materials properties
    #--------------------------------------------------------------------------
    def fetch_host_properties(self, experiments, data, **kwargs): 
        host_properties = HostProperties(self.configuration) 
        # merge adsorbates names with those found in the experiments dataset
        all_hosts = pd.concat([
            experiments['adsorbent_name'].dropna(),
            data['name'].dropna()
            ]).astype(str).str.strip().str.lower().unique()        
        
        # fetch adsorbents molecular properties using pubchem API
        all_hosts = pd.DataFrame(all_hosts, columns=['name'])
        properties = host_properties.get_properties_for_multiple_compounds(
            all_hosts, worker=kwargs.get('worker', None),
            progress_callback=kwargs.get('progress_callback', None))
        
        # build the enriched dataset using the extracted molecular properties        
        dataset = self.map_fetched_properties(data, properties)  

        return dataset
 
# [DATASET OPERATIONS]
###############################################################################
class GuestProperties:    
    
    def __init__(self, configuration):
        self.configuration = configuration
        self.properties = {'name' : [],
                           'adsorbate_molecular_weight' : [],
                           'adsorbate_molecular_formula' : [],
                           'adsorbate_SMILE' : []}      
    
    #--------------------------------------------------------------------------
    def get_molecular_properties(self, identifier, namespace): 
        try:           
            compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
            properties = compounds[0].to_dict() if compounds else {}
        except:
            logger.error(f'Cannot fetch molecules properties for identifier {identifier}: [{namespace}]')
            properties = {}

        return properties     
    
    #--------------------------------------------------------------------------    
    def get_properties_for_multiple_compounds(self, dataset, **kwargs):         
        for i, row in enumerate(tqdm(dataset.itertuples(index=True), total=dataset.shape[0])):  
            properties = self.get_molecular_properties(row.name, namespace='name')               
            if properties:                
                self.distribute_extracted_data(row.name, properties)

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i, dataset.shape[0], kwargs.get("progress_callback", None))                          

        return self.properties
    
    #--------------------------------------------------------------------------
    def distribute_extracted_data(self, name, features):               
        self.properties['name'].append(name)        
        self.properties['adsorbate_molecular_weight'].append(
            features.get('molecular_weight', np.nan))
        self.properties['adsorbate_molecular_formula'].append(
            features.get('molecular_formula', np.nan))
        self.properties['adsorbate_SMILE'].append(
            features.get('canonical_smiles', np.nan))
    

# [DATASET OPERATIONS]
###############################################################################
class HostProperties:    
    
    def __init__(self, configuration):        
        self.configuration = configuration        
        self.properties = {'name' : [],
                           'adsorbent_molecular_weight' : [],
                           'adsorbent_molecular_formula' : [],
                           'adsorbent_SMILE' : []}

    #--------------------------------------------------------------------------
    def is_chemical_formula(self, string):    
        formula_pattern = r"^[A-Za-z0-9\[\](){}·.,+\-_/]+$"
        return bool(re.match(formula_pattern, string))   

    #--------------------------------------------------------------------------
    def get_molecular_properties(self, identifier, namespace): 
        try:           
            compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
            properties = compounds[0].to_dict() if compounds else {}
        except:
            logger.error(f'Cannot fetch molecules properties for identifier {identifier}: [{namespace}]')
            properties = {}

        return properties     
    
    #--------------------------------------------------------------------------    
    def get_properties_for_multiple_compounds(self, dataset, **kwargs):         
        for i, row in enumerate(tqdm(dataset.itertuples(index=True), total=dataset.shape[0])): 
            formula_as_name = self.is_chemical_formula(row.name) 
            properties = self.get_molecular_properties(row.name, namespace='name')             
            if properties:                
                self.distribute_extracted_data(row.name, properties) 

            check_thread_status(kwargs.get("worker", None))
            update_progress_callback(
                i, dataset.shape[0], kwargs.get("progress_callback", None))                                 

        return self.properties
    
    #--------------------------------------------------------------------------
    def distribute_extracted_data(self, name, features):               
        self.properties['name'].append(name)        
        self.properties['adsorbent_molecular_weight'].append(
            features.get('molecular_weight', np.nan))
        self.properties['adsorbent_molecular_formula'].append(
            features.get('molecular_formula', np.nan))
        self.properties['adsorbent_SMILE'].append(
            features.get('canonical_smile', np.nan))
    

    
 
   
        
    


