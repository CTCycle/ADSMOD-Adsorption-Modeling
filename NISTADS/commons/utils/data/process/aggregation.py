import pandas as pd
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
class AggregateDatasets:

    def __init__(self, configurations):
        self.configurations = configurations
        self.guest_properties = [
            'name', 'adsorbate_molecular_weight', 'adsorbate_SMILE']
        self.host_properties = ['name']

    #--------------------------------------------------------------------------
    def join_materials_properties(self, adsorption : pd.DataFrame, guests : pd.DataFrame,
                                  hosts : pd.DataFrame): 
        all_dataset_merge = (adsorption
            .merge(guests[self.guest_properties], left_on='adsorbate_name', right_on='name', how='left')
            .drop(columns=['name'])
            .merge(hosts[self.host_properties], left_on='adsorbent_name', right_on='name', how='left')
            .drop(columns=['name'])
            .dropna())
        
        return all_dataset_merge        
        
    #--------------------------------------------------------------------------
    def aggregate_adsorption_measurements(self, dataset : pd.DataFrame):
        aggregate_dict = {'temperature' : 'first',                  
                          'adsorbent_name' : 'first',
                          'adsorbate_name' : 'first',
                          'pressureUnits' : 'first',
                          'adsorptionUnits' : 'first',                            
                          'pressure' : lambda x: [float(v) for v in x],
                          'adsorbed_amount' : lambda x: [float(v) for v in x]}   

        grouped_data = dataset.groupby(by='filename').agg(aggregate_dict).reset_index()
        grouped_data.drop(columns=['filename'], inplace=True)        

        return grouped_data

            



            