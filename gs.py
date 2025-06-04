from googlesearch import search
from typing import List

class SearchResults:
    @staticmethod
    def google_search(query: str, num_results: int = 5) -> List[str]:
        """
        Perform a Google search and return a list of URLs.

        Args:
            query (str): The search query.
            num_results (int): Number of results to return (default: 5).

        Returns:
            List[str]: List of URLs from the search results.

        Raises:
            ValueError: If query is empty or num_results is invalid.
            Exception: For other search-related errors.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if not isinstance(num_results, int) or num_results < 1:
            raise ValueError("num_results must be a positive integer")

        try:
            search_results = []
            for url in search(query, num_results=num_results):
                search_results.append(url)
            return search_results
        except Exception as e:
            raise Exception(f"Error during Google search: {str(e)}")
