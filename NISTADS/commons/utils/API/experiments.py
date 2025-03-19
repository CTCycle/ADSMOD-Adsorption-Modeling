import numpy as np
import pandas as pd
import requests as r
import asyncio

from NISTADS.commons.utils.API.status import GetServerStatus
from NISTADS.commons.utils.API.asynchronous import AsyncDataFetcher
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [NIST DATABASE API]
###############################################################################
class AdsorptionDataFetch:  
    
    def __init__(self, configuration): 
        self.server = GetServerStatus()
        self.server.check_status()  
        self.async_fetcher = AsyncDataFetcher(configuration)     
        self.exp_fraction = configuration["collection"]["EXP_FRACTION"]       
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
        self.exp_identifier = 'filename'
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_experiments_index(self):      
        response = r.get(self.url_isotherms)
        if response.status_code == 200:             
            isotherm_index = response.json()    
            experiments_data = pd.DataFrame(isotherm_index) 
            logger.info(f'Successfully retrieved adsorption isotherm index from {self.url_isotherms}')    
        else:
            logger.error(f'Error: Failed to retrieve data. Status code: {response.status_code}')
            experiments_data = None
            
        return experiments_data    
    
    # function to retrieve HTML data
    #--------------------------------------------------------------------------
    def get_experiments_data(self, experiments_data : pd.DataFrame):           
        n_samples = int(np.ceil(self.exp_fraction * experiments_data.shape[0]))        
        if isinstance(experiments_data, pd.DataFrame) and experiments_data.shape[0] > 0:
            loop = asyncio.get_event_loop()            
            exp_URLs = [f'https://adsorption.nist.gov/isodb/api/isotherm/{n}.json' 
                        for n in experiments_data[self.exp_identifier].to_list()[:n_samples]]
            adsorption_isotherms_data = loop.run_until_complete(
                self.async_fetcher.get_call_to_multiple_endpoints(exp_URLs))
            adsorption_isotherms_data = [d for d in adsorption_isotherms_data if d is not None]
            adsorption_isotherms_data = pd.DataFrame(adsorption_isotherms_data)        

        return adsorption_isotherms_data

             