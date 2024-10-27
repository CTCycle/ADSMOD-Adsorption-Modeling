import os
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


# [MERGE DATASETS]
###############################################################################
def merge_all_datasets(adsorption : pd.DataFrame, guests : pd.DataFrame,
                       hosts : pd.DataFrame):

    guest_properties = ['name', 'heavy_atoms', 'elements', 'molecular_weight',
                        'molecular_formula', 'SMILE', 'H_acceptors', 'H_donors']
    host_properties = ['name']

    all_dataset_merge = (adsorption
        .merge(guests[guest_properties], left_on='adsorbate_name', right_on='name', how='left')
        .drop(columns=['name'])
        .merge(hosts[host_properties], left_on='adsorbent_name', right_on='name', how='left')
        .drop(columns=['name']))
    
    return all_dataset_merge
    
    
# [MERGE DATASETS]
###############################################################################
def aggregate_adsorption_measurements(dataset : pd.DataFrame):

    aggregate_dict = {'temperature' : 'first',                  
                    'adsorbent_name' : 'first',
                    'adsorbate_name' : 'first',
                    'pressureUnits' : 'first',
                    'adsorptionUnits' : 'first',                            
                    'pressure' : list,
                    'adsorbed_amount' : list}   

    grouped_data = dataset.groupby(by='filename').agg(aggregate_dict).reset_index()
    grouped_data.drop(columns=['filename'], inplace=True)

    return grouped_data

        



        