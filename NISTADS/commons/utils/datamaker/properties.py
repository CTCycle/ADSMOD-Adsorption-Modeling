import pubchempy as pcp
from tqdm import tqdm

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger

 
# [DATASET OPERATIONS]
###############################################################################
class GuestProperties:    
    
    def __init__(self):

        self.properties = {'name' : [],
                            'atoms' : [],
                            'heavy_atoms' : [],
                            'bonds' : [],
                            'elements' : [],
                            'molecular_weight' : [],
                            'molecular_formula' : [],
                            'SMILE' : [],
                            'H_acceptors' : [],
                            'H_donors' : [],
                            'heavy_atoms' : []}
    
    #--------------------------------------------------------------------------
    def get_properties_for_single_guest(self, name):
        
        try:
            compounds = pcp.get_compounds(name, namespace='name', list_return='flat')
            properties = compounds[0].to_dict()
            logger.debug(f'Successfully retrieved properties for {name}')
            return properties
        except Exception as e:
            logger.error(f'Error fetching properties for {name}: {e}')
            return {}

    #--------------------------------------------------------------------------
    def get_properties_for_multiple_guests(self, names : list, synonims=None):
            
        for name, syno in tqdm(zip(names, synonims)):
            features = self.get_properties_for_single_guest(name)
            
            if not features and syno:
                logger.debug(f'Could not find properties of {name}. Now trying with synonims: {synonims}')
                for s in syno:
                    features = self.get_properties_for_single_guest(s)
                    if features:
                        break  
          
            if features:
                self.process_extracted_properties(name, features)                

        return self.properties
    
    #--------------------------------------------------------------------------
    def process_extracted_properties(self, name, features):        
           
        self.properties['name'].append(name)
        self.properties['atoms'].append(features.get('atoms', 'NA'))
        self.properties['heavy_atoms'].append(features.get('heavy_atom_count', 'NA'))
        self.properties['bonds'].append(features.get('bonds', 'NA'))
        self.properties['elements'].append(' '.join(features.get('elements', 'NA')))
        self.properties['molecular_weight'].append(features.get('molecular_weight', 'NA'))
        self.properties['molecular_formula'].append(features.get('molecular_formula', 'NA'))
        self.properties['SMILE'].append(features.get('canonical_smiles', 'NA'))
        self.properties['H_acceptors'].append(features.get('h_bond_acceptor_count', 'NA'))
        self.properties['H_donors'].append(features.get('h_bond_donor_count', 'NA'))          
        
        
    


# [DATASET OPERATIONS]
###############################################################################
class HostProperties:    
    
    def __init__(self):
        pass   
    
        
    


