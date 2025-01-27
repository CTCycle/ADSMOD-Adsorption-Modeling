import os
import pandas as pd
from tqdm import tqdm

from NISTADS.commons.utils.datamaker.properties import GuestProperties, HostProperties
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class BuildMaterialsDataset:

    def __init__(self, configuration):        
        self.guest_props = GuestProperties()
        self.host_props = HostProperties()                
        self.configuration = configuration       

    #--------------------------------------------------------------------------           
    def retrieve_materials_from_experiments(self, experiments : pd.DataFrame, 
                                            guests : pd.DataFrame, hosts : pd.DataFrame):
      
        adsorbates = pd.DataFrame(experiments['adsorbate_name'].unique().tolist(), columns=['name'])
        adsorbents = pd.DataFrame(experiments['adsorbent_name'].unique().tolist(), columns=['name'])                
        guests = pd.concat([guests, adsorbates], ignore_index=True)
        hosts = pd.concat([hosts, adsorbents], ignore_index=True)        
        guests, hosts = guests.dropna(subset=['name']), hosts.dropna(subset=['name'])
        
        # remove all duplicated names, keeping only rows where InChiKey is available
        # fill nan with empty lists
        guests['name'] = guests['name'].str.lower()
        hosts['name'] = hosts['name'].str.lower()
        guests = self.remove_duplicates_without_identifiers(guests)
        hosts = self.remove_duplicates_without_identifiers(hosts)
        guests, hosts = guests.fillna('[]'), hosts.fillna('[]')
    
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
    def add_guest_properties(self, data : pd.DataFrame):                     
        properties = self.guest_props.get_properties_for_multiple_guests(data)
        property_table = pd.DataFrame.from_dict(properties)        
        data['name'] = data['name'].apply(lambda x : x.lower())
        property_table['name'] = property_table['name'].apply(lambda x : x.lower())
        merged_data = data.merge(property_table, on='name', how='outer')

        return merged_data
    
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
class BuildAdsorptionDataset:

    def __init__(self):
        self.drop_cols = ['DOI', 'category', 'tabular_data', 'digitizer', 'isotherm_type', 
                          'articleSource', 'concentrationUnits', 'compositionType']
        self.explode_cols = ['pressure', 'adsorbed_amount']

    #--------------------------------------------------------------------------           
    def drop_excluded_columns(self, dataframe : pd.DataFrame):
        df_drop = dataframe.drop(columns=self.drop_cols, axis=1)

        return df_drop

    #--------------------------------------------------------------------------           
    def split_by_mixture_complexity(self, dataframe : pd.DataFrame):        
        dataframe['numGuests'] = dataframe['adsorbates'].apply(lambda x : len(x))          
        df_grouped = dataframe.groupby('numGuests')
        single_compound = df_grouped.get_group(1)
        binary_mixture = df_grouped.get_group(2)                
        
        return single_compound, binary_mixture   

    #--------------------------------------------------------------------------
    def extract_nested_data(self, dataframe : pd.DataFrame):         
        dataframe['adsorbent_ID'] = dataframe['adsorbent'].apply(lambda x : x['hashkey'])      
        dataframe['adsorbent_name'] = dataframe['adsorbent'].apply(lambda x : x['name'].lower())           
        dataframe['adsorbates_ID'] = dataframe['adsorbates'].apply(lambda x : [f['InChIKey'] for f in x])            
        dataframe['adsorbate_name'] = dataframe['adsorbates'].apply(lambda x : [f['name'].lower() for f in x])

        # check if the number of guest species is one (single component dataset)
        if (dataframe['numGuests'] == 1).all():
            dataframe['pressure'] = dataframe['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
            dataframe['adsorbed_amount'] = dataframe['isotherm_data'].apply(lambda x : [f['total_adsorption'] for f in x])
            dataframe['adsorbate_name'] = dataframe['adsorbates'].apply(lambda x : [f['name'].lower() for f in x][0])
            dataframe['composition'] = 1.0 

        # check if the number of guest species is two (binary mixture dataset)
        elif (dataframe['numGuests'] == 2).all():
            data_placeholder = {'composition' : 1.0, 'adsorption': 1.0}
            dataframe['total_pressure'] = dataframe['isotherm_data'].apply(lambda x : [f['pressure'] for f in x])                
            dataframe['all_species_data'] = dataframe['isotherm_data'].apply(lambda x : [f['species_data'] for f in x])
            dataframe['compound_1'] = dataframe['adsorbate_name'].apply(lambda x : x[0].lower())        
            dataframe['compound_2'] = dataframe['adsorbate_name'].apply(lambda x : x[1].lower() if len(x) > 1 else None)              
            dataframe['compound_1_data'] = dataframe['all_species_data'].apply(lambda x : [f[0] for f in x])               
            dataframe['compound_2_data'] = dataframe['all_species_data'].apply(lambda x : [f[1] if len(f) > 1 else data_placeholder for f in x])
            dataframe['compound_1_composition'] = dataframe['compound_1_data'].apply(lambda x : [f['composition'] for f in x])              
            dataframe['compound_2_composition'] = dataframe['compound_2_data'].apply(lambda x : [f['composition'] for f in x])            
            dataframe['compound_1_pressure'] = dataframe.apply(lambda x: [a * b for a, b in zip(x['compound_1_composition'], x['total_pressure'])], axis=1)             
            dataframe['compound_2_pressure'] = dataframe.apply(lambda x: [a * b for a, b in zip(x['compound_2_composition'], x['total_pressure'])], axis=1)                
            dataframe['compound_1_adsorption'] = dataframe['compound_1_data'].apply(lambda x : [f['adsorption'] for f in x])               
            dataframe['compound_2_adsorption'] = dataframe['compound_2_data'].apply(lambda x : [f['adsorption'] for f in x])

        return dataframe           
    
    #--------------------------------------------------------------------------
    def expand_dataset(self, single_component : pd.DataFrame, binary_mixture : pd.DataFrame):
           
        # processing and exploding data for single component dataset
        explode_cols = ['pressure', 'adsorbed_amount']
        drop_columns = ['date', 'adsorbent', 'adsorbates', 'numGuests', 
                        'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']
                
        SC_dataset = single_component.explode(explode_cols)
        SC_dataset[explode_cols] = SC_dataset[explode_cols].astype('float32')
        SC_dataset.reset_index(inplace=True, drop=True)       
        SC_dataset = SC_dataset.drop(columns=drop_columns, axis=1)
        SC_dataset.dropna(inplace=True)
                 
        # processing and exploding data for binary mixture dataset
        explode_cols = ['compound_1_pressure', 'compound_2_pressure',
                        'compound_1_adsorption', 'compound_2_adsorption',
                        'compound_1_composition', 'compound_2_composition']
        drop_columns.extend(['adsorbate_name', 'all_species_data', 'compound_1_data', 
                             'compound_2_data', 'adsorbent_ID', 'adsorbates_ID'])        

        BM_dataset = binary_mixture.explode(explode_cols)
        BM_dataset[explode_cols] = BM_dataset[explode_cols].astype('float32')       
        BM_dataset.reset_index(inplace=True, drop=True)        
        BM_dataset = BM_dataset.drop(columns=drop_columns)
        BM_dataset.dropna(inplace=True)    
        
        return SC_dataset, BM_dataset
 
    



       

