import sys
import time
import requests as r

import pandas as pd
import numpy as np
import aiohttp
import asyncio
from tqdm import tqdm

from NISTADS.app.interface.workers import check_thread_status, update_progress_callback
from NISTADS.app.logger import logger



# function to retrieve HTML data
###############################################################################
class GetServerStatus:

    def __init__(self):
        self.server_url = 'https://adsorption.nist.gov'

    #--------------------------------------------------------------------------
    def check_status(self):       
        response = r.get(self.server_url)
        # Checking if the request was successful
        if response.status_code == 200:
            logger.info(f'NIST server is up and running. Status code: {response.status_code}')
        else:            
            logger.error(f'Failed to reach the server. Status code: {response.status_code}') 
            time.sleep(5)
            sys.exit()
        

# function to retrieve HTML data
###############################################################################
class AsyncDataFetcher:

    def __init__(self, configuration, num_calls=None):
        num_calls_by_config = configuration.get('parallel_tasks', 20)
        self.num_calls = num_calls_by_config if num_calls is None else num_calls
        self.semaphore = asyncio.Semaphore(self.num_calls)

    #--------------------------------------------------------------------------
    async def get_call_to_single_endpoint(self, session, url):
        async with self.semaphore:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f'Could not fetch data from {url}. Status: {response.status}')
                    return None
                try:
                    return await response.json()
                except aiohttp.client_exceptions.ContentTypeError as e:
                    logger.error(f'Error decoding JSON from {url}: {e}')
                    return None                

    #--------------------------------------------------------------------------
    async def get_call_to_multiple_endpoints(self, urls, **kwargs):
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_call_to_single_endpoint(session, url)
                     for url in urls]
            results = []
            total = len(tasks)
            with tqdm(total=total, desc="Fetching", unit="req") as pbar:
                for idx, future in enumerate(asyncio.as_completed(tasks)):
                    result = await future
                    results.append(result)
                    check_thread_status(kwargs.get("worker", None))
                    update_progress_callback(idx + 1, total, kwargs.get("progress_callback", None))
                    pbar.update(1) 
        return results


# [NIST DATABASE API]
###############################################################################
class AdsorptionDataFetch:  
    
    def __init__(self, configuration):
        # get server status before running any API method, 
        # success when returning 200
        self.server = GetServerStatus()
        self.server.check_status()        
        # define experiments fraction and main endpoint for fetching adsorption isotherms
        self.exp_fraction = configuration.get('experiments_fraction', 1.0)       
        self.url_isotherms = 'https://adsorption.nist.gov/isodb/api/isotherms.json'
        self.exp_identifier = 'filename'
        self.configuration = configuration  
    
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
    def get_experiments_data(self, experiments_data, **kwargs):
        async_fetcher = AsyncDataFetcher(self.configuration)                
        num_samples = int(np.ceil(self.exp_fraction * experiments_data.shape[0]))        
        if isinstance(experiments_data, pd.DataFrame) and experiments_data.shape[0] > 0:
            exp_URLs = [f'https://adsorption.nist.gov/isodb/api/isotherm/{n}.json' 
                        for n in experiments_data[self.exp_identifier].to_list()[:num_samples]]
            
            # Always create a new event loop in a QThread context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # parallel endpoint calls using run_unit_complete on the event loop
                adsorption_isotherms_data = loop.run_until_complete(
                    async_fetcher.get_call_to_multiple_endpoints(
                        exp_URLs,
                        worker=kwargs.get('worker', None),
                        progress_callback=kwargs.get('progress_callback', None)))                
            finally:
                loop.close()
                
            adsorption_isotherms_data = [d for d in adsorption_isotherms_data if d is not None]
            adsorption_isotherms_data = pd.DataFrame(adsorption_isotherms_data)

            return adsorption_isotherms_data    
        


# [NIST DATABASE API: GUEST/HOST]
###############################################################################
class GuestHostDataFetch: 

    def __init__(self, configuration):
        # get server status before running any API method, 
        # success when returning 200
        self.server = GetServerStatus()
        self.server.check_status()        
        # define guest/host identifiers and endpoints for fetching materials
        self.guest_identifier = 'InChIKey'
        self.host_identifier = 'hashkey'
        self.url_GUEST = 'https://adsorption.nist.gov/isodb/api/gases.json'
        self.url_HOST = 'https://adsorption.nist.gov/matdb/api/materials.json'            
        
        self.extra_guest_columns = [
            'adsorbate_molecular_weight', 'adsorbate_molecular_formula', 'adsorbate_SMILE']
        self.extra_host_columns = [            
            'adsorbent_molecular_weight', 'adsorbent_molecular_formula', 'adsorbent_SMILE']
        
        self.guest_fraction = configuration.get('guest_fraction', 1.0)
        self.host_fraction = configuration.get('host_fraction', 1.0) 
        self.configuration = configuration          

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
    def get_materials_data(self, guest_index=None, host_index=None, **kwargs):
        # Always create a new event loop in a QThread context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        guest_data, host_data = None, None
        async_fetcher = AsyncDataFetcher(self.configuration)   

        try:
            if isinstance(guest_index, pd.DataFrame) and guest_index.shape[0] > 0:
                num_samples = int(np.ceil(self.guest_fraction * guest_index.shape[0]))            
                guest_urls = [f'https://adsorption.nist.gov/isodb/api/gas/{n}.json' 
                            for n in guest_index[self.guest_identifier].to_list()[:num_samples]]
                guest_data = loop.run_until_complete(
                    async_fetcher.get_call_to_multiple_endpoints(
                        guest_urls, 
                        worker=kwargs.get('worker', None),
                        progress_callback=kwargs.get('progress_callback', None)))
                
                guest_data = [data for data in guest_data if data is not None]
                guest_data = pd.DataFrame(guest_data)
                guest_data = guest_data.assign(
                    **{col: np.nan for col in self.extra_guest_columns})
            else:
                logger.error('No available guest data has been found. Skipping directly to host index')
            
            if isinstance(host_index, pd.DataFrame) and host_index.shape[0] > 0:
                num_samples = int(np.ceil(self.host_fraction * host_index.shape[0]))
                host_urls = [f'https://adsorption.nist.gov/isodb/api/material/{n}.json' 
                            for n in host_index[self.host_identifier].to_list()[:num_samples]]
                host_data = loop.run_until_complete(
                    async_fetcher.get_call_to_multiple_endpoints(
                        host_urls, 
                        worker=kwargs.get('worker', None),
                        progress_callback=kwargs.get('progress_callback', None)))
                
                host_data = [data for data in host_data if data is not None]
                host_data = pd.DataFrame(host_data)
                host_data = host_data.assign(
                    **{col: np.nan for col in self.extra_host_columns})
            else:
                logger.error('No available host data has been found.')
        finally:
            loop.close()
        
        return guest_data, host_data
