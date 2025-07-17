from NISTADS.app.constants import DATA_PATH
from NISTADS.app.logger import logger


# [DATASET OPERATIONS]
###############################################################################
class BuildAdsorptionDataset:

    def __init__(self):
        self.raw_explode_cols = ['pressure', 'adsorbed_amount']
        self.raw_drop_cols = [
            'DOI', 'category', 'tabular_data', 'digitizer', 'isotherm_type', 
            'articleSource', 'concentrationUnits', 'compositionType'] 
        self.SC_explode_cols = ['pressure', 'adsorbed_amount']
        self.SC_drop_columns = [
            'date', 'adsorbent', 'adsorbates', 'numGuests', 
            'isotherm_data', 'adsorbent_ID', 'adsorbates_ID']
        self.BM_explode_cols = [
            'compound_1_pressure', 'compound_2_pressure',
            'compound_1_adsorption', 'compound_2_adsorption',
            'compound_1_composition', 'compound_2_composition']
        self.BM_drop_columns = [
            'date', 'adsorbent', 'adsorbates', 'numGuests', 
            'isotherm_data', 'adsorbent_ID', 'adsorbates_ID',
            'adsorbate_name', 'total_pressure', 'all_species_data', 
            'compound_1_data', 'compound_2_data', 'adsorbent_ID', 'adsorbates_ID']       

    #--------------------------------------------------------------------------           
    def drop_excluded_columns(self, dataframe):
        df_drop = dataframe.drop(columns=self.raw_drop_cols, axis=1)

        return df_drop

    #--------------------------------------------------------------------------           
    def split_by_mixture_complexity(self, dataframe):        
        dataframe['numGuests'] = dataframe['adsorbates'].apply(lambda x : len(x))          
        df_grouped = dataframe.groupby('numGuests')
        single_compound = df_grouped.get_group(1)
        binary_mixture = df_grouped.get_group(2)                
        
        return single_compound, binary_mixture   

    #--------------------------------------------------------------------------
    def extract_nested_data(self, dataframe):         
        dataframe['adsorbent_ID'] = dataframe['adsorbent'].apply(
            lambda x : x['hashkey']).astype(str)      
        dataframe['adsorbent_name'] = dataframe['adsorbent'].apply(
            lambda x : x['name'].lower()).astype(str)           
        dataframe['adsorbates_ID'] = dataframe['adsorbates'].apply(
            lambda x : [f['InChIKey'] for f in x]).astype(str)            
        dataframe['adsorbate_name'] = dataframe['adsorbates'].apply(
            lambda x : [f['name'].lower() for f in x]).astype(str)

        # check if the number of guest species is one (single component dataset)
        if (dataframe['numGuests'] == 1).all():
            dataframe['pressure'] = dataframe['isotherm_data'].apply(
                lambda x : [f['pressure'] for f in x])                
            dataframe['adsorbed_amount'] = dataframe['isotherm_data'].apply(
                lambda x : [f['total_adsorption'] for f in x])
            dataframe['adsorbate_name'] = dataframe['adsorbates'].apply(
                lambda x : [f['name'].lower() for f in x][0])
            dataframe['composition'] = 1.0 

        # check if the number of guest species is two (binary mixture dataset)
        elif (dataframe['numGuests'] == 2).all():
            data_placeholder = {'composition' : 1.0, 'adsorption': 1.0}
            dataframe['total_pressure'] = dataframe['isotherm_data'].apply(
                lambda x : [f['pressure'] for f in x])                
            dataframe['all_species_data'] = dataframe['isotherm_data'].apply(
                lambda x : [f['species_data'] for f in x])
            dataframe['compound_1'] = dataframe['adsorbate_name'].apply(
                lambda x : x[0].lower())        
            dataframe['compound_2'] = dataframe['adsorbate_name'].apply(
                lambda x : x[1].lower() if len(x) > 1 else None)              
            dataframe['compound_1_data'] = dataframe['all_species_data'].apply(
                lambda x : [f[0] for f in x])               
            dataframe['compound_2_data'] = dataframe['all_species_data'].apply(
                lambda x : [f[1] if len(f) > 1 else data_placeholder for f in x])
            dataframe['compound_1_composition'] = dataframe['compound_1_data'].apply(
                lambda x : [f['composition'] for f in x])              
            dataframe['compound_2_composition'] = dataframe['compound_2_data'].apply(
                lambda x : [f['composition'] for f in x])            
            dataframe['compound_1_pressure'] = dataframe.apply(
                lambda x: [a * b for a, b in zip(x['compound_1_composition'], x['total_pressure'])], axis=1)             
            dataframe['compound_2_pressure'] = dataframe.apply(
                lambda x: [a * b for a, b in zip(x['compound_2_composition'], x['total_pressure'])], axis=1)                
            dataframe['compound_1_adsorption'] = dataframe['compound_1_data'].apply(
                lambda x : [f['adsorption'] for f in x])               
            dataframe['compound_2_adsorption'] = dataframe['compound_2_data'].apply(
                lambda x : [f['adsorption'] for f in x])

        return dataframe           
    
    #--------------------------------------------------------------------------
    def expand_dataset(self, single_component, binary_mixture):           
        # processing and exploding data for single component dataset                
        SC_dataset = single_component.explode(self.SC_explode_cols)        
        SC_dataset.reset_index(inplace=True, drop=True)       
        SC_dataset = SC_dataset.drop(columns=self.SC_drop_columns, axis=1)
        SC_dataset.dropna(inplace=True)
                 
        # processing and exploding data for binary mixture dataset
        BM_dataset = binary_mixture.explode(self.BM_explode_cols)               
        BM_dataset.reset_index(inplace=True, drop=True)        
        BM_dataset = BM_dataset.drop(columns=self.BM_drop_columns)
        BM_dataset.dropna(inplace=True)    
        
        return SC_dataset, BM_dataset
 
    



       

