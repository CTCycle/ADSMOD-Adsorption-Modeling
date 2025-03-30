import re
import numpy as np
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:

    def __init__(self, configuration): 
        self.molecular_identifier = 'InChIKey'       
        self.guest_properties = GuestProperties(configuration)
        self.host_properties = HostProperties(configuration)                
        self.configuration = configuration   

    # Define a function to handle duplicates, keeping rows with InChIKey
    #--------------------------------------------------------------------------
    def remove_duplicates_without_identifiers(self, data : pd.DataFrame):
        if self.molecular_identifier in data.columns:
            data['has_inchikey'] = data['InChIKey'].notna()  
            data = data.sort_values(by=['name', 'has_inchikey'], ascending=[True, False])
            data = data.drop_duplicates(subset=['name'], keep='first')  
            data = data.drop(columns=['has_inchikey'])
        else:
            data = data.drop_duplicates(subset=['name'], keep='first')  

        return data       
    
    #--------------------------------------------------------------------------
    def fetch_guest_properties(self, experiments : pd.DataFrame, data : pd.DataFrame): 
        # Combine guest names from experiments and data, cleaning them to ensure consistency
        guest_names = pd.concat([
            experiments['adsorbate_name'].dropna(),
            data['name'].dropna()
            ]).astype(str).str.strip().str.lower().unique()        
        
        # fetch adsorbates molecular properties using pubchem API
        all_guests = pd.DataFrame(guest_names, columns=['name'])                   
        properties = self.guest_properties.get_properties_for_multiple_compounds(all_guests)

        # build the enriched dataset using the extracted molecular properties
        property_table = pd.DataFrame.from_dict(properties)       
        data['name'] = data['name'].str.lower()
        property_table['name'] = property_table['name'].str.lower()        
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data
    
    # this function is not used in the current version of the code, since it is 
    # difficult to find a reliable source for the adsorbent materials properties
    #--------------------------------------------------------------------------
    def fetch_host_properties(self, experiments : pd.DataFrame, data : pd.DataFrame): 
        # merge adsorbates names with those found in the experiments dataset
        all_hosts = pd.concat([
            experiments['adsorbate_name'].dropna(),
            data['name'].dropna()
            ]).astype(str).str.strip().str.lower().unique()        
        
        # fetch adsorbents molecular properties using pubchem API
        all_hosts = pd.DataFrame(all_hosts, columns=['name'])
        properties = self.host_properties.get_properties_for_multiple_compounds(all_hosts)
        
        # build the enriched dataset using the extracted molecular properties
        property_table = pd.DataFrame.from_dict(properties)       
        data['name'] = data['name'].str.lower()
        property_table['name'] = property_table['name'].str.lower()        
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data

 
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
    def get_properties_for_multiple_compounds(self, dataset : pd.DataFrame):         
        for row in tqdm(dataset.itertuples(index=True), total=dataset.shape[0]):  
            properties = self.get_molecular_properties(row.name, namespace='name')               
            if properties:                
                self.distribute_extracted_data(row.name, properties)                          

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
    def get_properties_for_multiple_compounds(self, dataset : pd.DataFrame):         
        for row in tqdm(dataset.itertuples(index=True), total=dataset.shape[0]): 
            formula_as_name = self.is_chemical_formula(row.name) 
            properties = self.get_molecular_properties(row.name, namespace='name')             
            if properties:                
                self.distribute_extracted_data(row.name, properties)                                    

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
    

    
 
   
        
    


