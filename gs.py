from googlesearch import search
from typing import List
import aiohttp
import asyncio

class SearchResults:
     
     @staticmethod
     async def google_search(query, num_results=5) -> List[str]:
         search_results = []
         async with aiohttp.ClientSession() as session:
             for url in search(query, num_results=num_results):
                 search_results.append(url)
             return search_results
