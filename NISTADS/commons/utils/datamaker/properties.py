import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [DATASET OPERATIONS]
###############################################################################
class MaterialsProperties:

    def __init__(self, configuration):        
        self.guest_properties = GuestProperties()
        self.host_properties = HostProperties()                
        self.configuration = configuration       

    #--------------------------------------------------------------------------           
    def get_complete_materials_pool(self, experiments : pd.DataFrame, 
                                    guests : pd.DataFrame, hosts : pd.DataFrame):
      
        adsorbates = pd.DataFrame(experiments['adsorbate_name'].unique().tolist(), columns=['name'])
        adsorbents = pd.DataFrame(experiments['adsorbent_name'].unique().tolist(), columns=['name'])

        guests = pd.concat([guests, adsorbates], ignore_index=True)
        hosts = pd.concat([hosts, adsorbents], ignore_index=True)        
        guests, hosts = guests.dropna(subset=['name']), hosts.dropna(subset=['name'])
        
        # remove all duplicated names, keeping only rows where InChiKey is available   
        guests['name'], hosts['name'] = guests['name'].str.lower(), hosts['name'].str.lower()        
            
        return guests, hosts    

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

        adsorbates = pd.DataFrame(experiments['adsorbate_name'].unique().tolist(), columns=['name'])
        all_guests = pd.concat([data, adsorbates], ignore_index=True)
        all_guests['name'] = all_guests['name'].str.lower()

        properties = self.guest_properties.get_properties_for_multiple_compounds(data)

        # property_table = pd.DataFrame.from_dict(properties)        
        # data['name'] = data['name'].apply(lambda x : x.lower())
        # property_table['name'] = property_table['name'].apply(lambda x : x.lower())
        # merged_data = data.merge(property_table, on='name', how='outer')

        #return merged_data
    
    #--------------------------------------------------------------------------
    def add_host_properties(self, data : pd.DataFrame):                
        properties = self.host_props.get_properties_for_multiple_hosts(data)
        property_table = pd.DataFrame.from_dict(properties)        
        data['name'] = data['name'].apply(lambda x : x.lower())
        property_table['name'] = property_table['name'].apply(lambda x : x.lower())
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data
 


 
# [DATASET OPERATIONS]
###############################################################################
class GuestProperties:    
    
    def __init__(self):
        pass   
        
    
    #--------------------------------------------------------------------------
    def get_properties(self, identifier, namespace):            
        compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
        properties = compounds[0].to_dict() if compounds else {}

        return {identifier : properties}        
    
    #--------------------------------------------------------------------------    
    def get_properties_for_multiple_compounds(self, dataset : pd.DataFrame): 

        
        
        
        for name, syno, inchikey, inchicode in tqdm(zip(names, synonyms, inchikeys, inchicodes), total=len(names)):
            features = None

            # attempt to find molecular properties using the compound name
            if name:
                try:
                    features = self.get_properties_for_single_guest(name, namespace='name')
                except Exception as e:
                    logger.error(f"Error fetching properties for name '{name}': {e}")

            # attempt to find molecular properties using the synonims for the compounds, 
            # where the main name does not lead to valid properties
            if not features and syno:
                for synonym in syno:
                    try:
                        features = self.get_properties_for_single_guest(synonym.lower(), namespace='name')
                        if features:
                            break
                    except:
                        logger.error(f"Error fetching properties for synonym '{synonym}'")

            # attempt to find molecular properties using the InChIKey for the compounds, 
            # where the main name and synonyms do not lead to valid properties
            if not features and inchikey:
                try:
                    features = self.get_properties_for_single_guest(inchikey, namespace='inchikey')
                except:
                    logger.error(f"Error fetching properties for InChIKey '{inchikey}'")

            # attempt to find molecular properties using the InChICode for the compounds, 
            # where the main name synonyms and InChIKey do not lead to valid properties
            if not features and inchicode:
                try:
                    features = self.get_properties_for_single_guest(inchicode, namespace='inchi')
                except:
                    logger.error(f"Error fetching properties for InChICode '{inchicode}'")
            
            if features:
                try:
                    self.process_extracted_properties(name, features)
                except:
                    logger.error(f"Error processing extracted properties for '{name}'")
            else:
                logger.warning(f"No properties found for {name}, synonyms, InChIKey, or InChICode")

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
    def get_properties_for_single_host(self, identifier, namespace):          
        compounds = pcp.get_compounds(identifier, namespace=namespace, list_return='flat')
        properties = compounds[0].to_dict() if compounds else None

        return properties

    #--------------------------------------------------------------------------
    def get_properties_for_multiple_hosts(self, dataset : pd.DataFrame):
        names = dataset['name'].str.strip().str.lower().tolist()
        synonyms = dataset['synonyms'].apply(lambda x: x if isinstance(x, list) else []).tolist()
        formula = dataset['formula'].apply(lambda x: x if isinstance(x, list) else []).tolist()

        for name, syno, form in tqdm(zip(names, synonyms, formula), total=len(names)):
            features = None

            # attempt to find molecular properties using the compound name
            if name:
                try:
                    features = self.get_properties_for_single_host(name, namespace='name')
                except:
                    logger.error(f"Error fetching properties for name '{s}'")

            # attempt to find molecular properties using the synonims for the compounds, 
            # where the main name does not lead to valid properties
            if not features and syno:
                for s in syno:
                    try:
                        features = self.get_properties_for_single_host(s, namespace='name')                        
                    except:
                        logger.error(f"Error fetching properties for synonym '{s}'")

            # attempt to find molecular properties using the synonims for the compounds, 
            # where the main name does not lead to valid properties
            if not features and form:
                for f in form:
                    try:
                        features = self.get_properties_for_single_host(f, namespace='formula')                        
                    except:
                        logger.error(f"Error fetching properties for formula '{f}'")

            # Process extracted properties or log warning
            if features:
                try:
                    self.process_extracted_properties(name, features)
                except Exception as e:
                    logger.error(f"Error processing extracted properties for '{name}': {e}")
            else:
                logger.warning(f"No properties found for {name} nor synonyms or formula")

        return self.properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):           
        self.properties['name'].append(name)        
        self.properties['adsorbent_molecular_weight'].append(features.get('molecular_weight', 'NA'))
        self.properties['adsorbent_molecular_formula'].append(features.get('molecular_formula', 'NA'))
        self.properties['adsorbent_SMILE'].append(features.get('canonical_smiles', 'NA'))
   
        
    


