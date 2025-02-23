import aiohttp
import asyncio
from tqdm.asyncio import tqdm_asyncio

from NISTADS.commons.constants import CONFIG
from NISTADS.commons.logger import logger


# function to retrieve HTML data
###############################################################################
class AsyncDataFetcher:

    def __init__(self, configuration, num_calls=None):
        num_calls_by_config = configuration["collection"]["PARALLEL_TASKS"]
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
    async def get_call_to_multiple_endpoints(self, urls):        
        async with aiohttp.ClientSession() as session:
            tasks = [self.get_call_to_single_endpoint(session, url) for url in urls]
            results = await tqdm_asyncio.gather(*tasks)

        return results


