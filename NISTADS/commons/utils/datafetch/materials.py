import numpy as np
import pandas as pd
import requests as r
import asyncio

from NISTADS.commons.utils.datafetch.status import GetServerStatus
from NISTADS.commons.utils.datafetch.asynchronous import AsyncDataFetcher
from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# [NIST DATABASE API: GUEST/HOST]
###############################################################################
class GuestHostDataFetch: 

    def __init__(self, configuration):
        self.server = GetServerStatus()
        self.server.check_status()  
        self.async_fetcher = AsyncDataFetcher(configuration)      
        self.url_GUEST = 'https://adsorption.nist.gov/isodb/api/gases.json'
        self.url_HOST = 'https://adsorption.nist.gov/matdb/api/materials.json'
        self.guest_fraction = configuration["collection"]["GUEST_FRACTION"]
        self.host_fraction = configuration["collection"]["HOST_FRACTION"]       
        self.guest_identifier = 'InChIKey'
        self.host_identifier = 'hashkey'

    #--------------------------------------------------------------------------
    def get_materials_index(self):       
        guest_json, guest_data = r.get(self.url_GUEST), None
        host_json, host_data = r.get(self.url_HOST), None
        if guest_json.status_code == 200:             
            guest_data = pd.DataFrame(guest_json.json() )
            logger.info(f'Total number of adsorbents: {guest_data.shape[0]}')
        else:
            logger.error(f'Failed to retrieve adsorbents data. Status code: {guest_json.status_code}')           
        if host_json.status_code == 200:            
            host_data = pd.DataFrame(host_json.json()) 
            logger.info(f'Total number of adsorbates: {host_data.shape[0]}')
        else:
            logger.error(f'Failed to retrieve adsorbates data. Status code: {host_json.status_code}')            
  
        return guest_data, host_data   
    
    #--------------------------------------------------------------------------
    def get_materials_data(self, guest_index=None, host_index=None):
        # initialize the asyncronous event loop         
        loop = asyncio.get_event_loop()
        guest_data, host_data = None, None
        if isinstance(guest_index, pd.DataFrame) and guest_index.shape[0] > 0:
            # Calculate the number of samples to retrieve for the guest data
            # Create a list of URLs for guest data based on the guest identifiers
            n_samples = int(np.ceil(self.guest_fraction * guest_index.shape[0]))            
            guest_urls = [f'https://adsorption.nist.gov/isodb/api/gas/{n}.json' 
                          for n in guest_index[self.guest_identifier].to_list()[:n_samples]]
            
            # Fetch guest data asynchronously from the provided URLs
            # Filter out any None results from the fetched data before converting to a DataFrame
            guest_data = loop.run_until_complete(
                self.async_fetcher.get_call_to_multiple_endpoints(guest_urls))
            guest_data = [data for data in guest_data if data is not None]
            guest_data = pd.DataFrame(guest_data) 
        else:
            logger.error('No available guest data has been found. Skipping directly to host index')
            
        if isinstance(host_index, pd.DataFrame) and host_index.shape[0] > 0:
            # Calculate the number of samples to retrieve for the host data
            # Create a list of URLs for host data based on the host identifiers
            n_samples = int(np.ceil(self.host_fraction * host_index.shape[0]))                   
            host_urls = [f'https://adsorption.nist.gov/isodb/api/material/{n}.json' 
                         for n in host_index[self.host_identifier].to_list()[:n_samples]] 

            # Fetch guest data asynchronously from the provided URLs
            # Filter out any None results from the fetched data before converting to a DataFrame      
            host_data = loop.run_until_complete(
                self.async_fetcher.get_call_to_multiple_endpoints(host_urls))        
            host_data = [data for data in host_data if data is not None] 
            host_data = pd.DataFrame(host_data)            
        else:
            logger.error('No available host data has been found.')        
            
        return guest_data, host_data 

      


