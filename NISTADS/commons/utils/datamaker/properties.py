import re
import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import requests


from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [DATASET OPERATIONS]
###############################################################################
class MolecularProperties:

    def __init__(self, configuration):        
        self.guest_properties = GuestProperties()
        self.host_properties = HostProperties()                
        self.configuration = configuration       
   

    # Define a function to handle duplicates, keeping rows with InChIKey
    #--------------------------------------------------------------------------
    def remove_duplicates_without_identifiers(self, data : pd.DataFrame):
        if 'InChIKey' in data.columns:
            data['has_inchikey'] = data['InChIKey'].notna()  
            data = data.sort_values(by=['name', 'has_inchikey'], ascending=[True, False])
            data = data.drop_duplicates(subset=['name'], keep='first')  
            data = data.drop(columns=['has_inchikey'])
        else:
            data = data.drop_duplicates(subset=['name'], keep='first')  

        return data       
    
    #--------------------------------------------------------------------------
    def fetch_guest_properties(self, experiments : pd.DataFrame, data : pd.DataFrame): 
        # merge adsorbates names with those found in the experiments dataset
        adsorbates = pd.DataFrame(experiments['adsorbate_name'].unique().tolist(), columns=['name'])
        all_guests = pd.concat([data, adsorbates], ignore_index=True)
        all_guests['name'] = all_guests['name'].astype(str).str.strip().str.lower()

        # fetch adsorbates molecular properties using pubchem API
        properties = self.guest_properties.get_properties_for_multiple_compounds(all_guests)

        # build the enriched dataset using the extracted molecular properties
        property_table = pd.DataFrame.from_dict(properties)       
        data['name'] = data['name'].apply(lambda x : x.lower())
        property_table['name'] = property_table['name'].apply(lambda x : x.lower())
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data
    
    #--------------------------------------------------------------------------
    def fetch_host_properties(self, experiments : pd.DataFrame, data : pd.DataFrame): 
        adsorbents = pd.DataFrame(experiments['adsorbent_name'].unique().tolist(), columns=['name'])
        all_hosts = pd.concat([data, adsorbents], ignore_index=True)
        all_hosts['name'] = all_hosts['name'].astype(str).str.strip().str.lower()
        
        # fetch adsorbents molecular properties using pubchem API
        properties = self.host_properties.get_properties_for_multiple_compounds(all_hosts)
        
        property_table = pd.DataFrame.from_dict(properties)        
        data['name'] = data['name'].apply(lambda x : x.lower())
        property_table['name'] = property_table['name'].apply(lambda x : x.lower())
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data
 


 
# [DATASET OPERATIONS]
###############################################################################
class GuestProperties:    
    
    def __init__(self):
        self.properties = {'name' : [],
                           'adsorbate_molecular_weight' : [],
                           'adsorbate_molecular_formula' : [],
                           'adsorbate_SMILE' : []}      
    
    #--------------------------------------------------------------------------
    def get_properties(self, identifier, namespace): 
        try:           
            compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
            properties = compounds[0].to_dict() if compounds else {}
        except:
            logger.error(f'Cannot fetch molecules properties for identifier: [{identifier}]')
            properties = {}

        return properties     
    
    #--------------------------------------------------------------------------    
    def get_properties_for_multiple_compounds(self, dataset : pd.DataFrame): 
        
        for row in tqdm(dataset.itertuples(index=True), total=dataset.shape[0]):  
            if pd.notna(row.name):
                properties = self.get_properties(row.name, namespace='name')                
            if not properties and isinstance(row.synonyms, list):
                for synonym in row.synonyms:
                    properties = self.get_properties(synonym, namespace='name')
                    if properties:
                        continue
            
            if not properties and pd.notna(row.InChIKey):
                properties = self.get_properties(row.name, namespace='inchikey')            
            if not properties and pd.notna(row.InChICode):
                properties = self.get_properties(row.name, namespace='inchi')            
            if properties:                
                self.process_extracted_properties(row.name, properties)                          

        return self.properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):               
        self.properties['name'].append(name)        
        self.properties['adsorbate_molecular_weight'].append(features.get('molecular_weight', 'NA'))
        self.properties['adsorbate_molecular_formula'].append(features.get('molecular_formula', 'NA'))
        self.properties['adsorbate_SMILE'].append(features.get('canonical_smiles', 'NA'))
    

# [DATASET OPERATIONS]
###############################################################################
class HostProperties:    
    
    def __init__(self):
        self.properties = {'name' : [],
                           'adsorbent_molecular_weight' : [],
                           'adsorbent_molecular_formula' : [],
                           'adsorbent_SMILE' : []}  
    
    #--------------------------------------------------------------------------
    def get_properties(self, identifier, namespace): 
        try:           
            compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
            properties = compounds[0].to_dict() if compounds else {}
        except:
            logger.error(f'Cannot fetch molecules properties for identifier: [{identifier}]')
            properties = {}

        return properties   

    #--------------------------------------------------------------------------
    def is_chemical_formula(self, string):    
        formula_pattern = r"^[A-Za-z0-9\[\](){}·.,+\-_/]+$"
        return bool(re.match(formula_pattern, string))

    #--------------------------------------------------------------------------
    def get_properties_for_multiple_compounds(self, dataset : pd.DataFrame):
        
        for row in tqdm(dataset.itertuples(index=True), total=dataset.shape[0]):  
            formula_as_name = self.is_chemical_formula(row.name)
            if pd.notna(row.name):
                properties = self.get_properties(row.name, namespace='name')

            # adsorbents are often named as their chemical formula, so the name
            # is checked for pattern matching with chemical formulas and if true
            # the name is used as formula identifier to fetch properties                       
            if not properties and pd.notna(row.name) and formula_as_name:
                properties = self.get_properties(row.name, namespace='formula')

            if not properties and isinstance(row.synonyms, list):
                for synonym in row.synonyms:
                    properties = self.get_properties(synonym, namespace='name')
                    if properties:
                        continue          
                   
            if properties:                
                self.process_extracted_properties(row.name, properties)                

        return self.properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):           
        self.properties['name'].append(name)        
        self.properties['adsorbent_molecular_weight'].append(features.get('molecular_weight', 'NA'))
        self.properties['adsorbent_molecular_formula'].append(features.get('molecular_formula', 'NA'))
        self.properties['adsorbent_SMILE'].append(features.get('canonical_smiles', 'NA'))
   
        
    


