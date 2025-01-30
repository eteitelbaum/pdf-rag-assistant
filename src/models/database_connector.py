"""
Academic Database Connector
-------------------------

Handles connections and queries to academic databases like JSTOR, Web of Science,
and Google Scholar. Note: Requires institutional access and API keys.
"""

from typing import List, Dict
from scholarly import scholarly

# Optional imports
try:
    from pyjstor import JSTOR  # type: ignore
    JSTOR_AVAILABLE = True
except ImportError:
    JSTOR_AVAILABLE = False

try:
    from wos import WOSClient  # type: ignore
    WOS_AVAILABLE = True
except ImportError:
    WOS_AVAILABLE = False


class AcademicDatabaseConnector:
    """Handles connections to various academic databases."""

    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize database connections.

        Args:
            api_keys: Dictionary of API keys for different services
        """
        self.api_keys = api_keys or {}
        self.jstor_client = None
        self.wos_client = None
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize database clients if API keys are available."""
        if 'jstor' in self.api_keys and JSTOR_AVAILABLE:
            try:
                self.jstor_client = JSTOR(self.api_keys['jstor'])
            except (ValueError, KeyError) as e:
                print(f"Failed to initialize JSTOR client: {str(e)}")

        if 'wos' in self.api_keys and WOS_AVAILABLE:
            try:
                self.wos_client = WOSClient(self.api_keys['wos'])
            except (ValueError, KeyError) as e:
                print(f"Failed to initialize WOS client: {str(e)}")

    async def search_databases(self, query: str) -> Dict[str, List[Dict]]:
        """
        Search across multiple academic databases.

        Args:
            query: Search query string

        Returns:
            Dictionary of results from each database
        """
        results = {'scholar': []}

        try:
            results['scholar'] = list(scholarly.search_pubs(query))
        except (ConnectionError, ValueError) as e:
            print(f"Error searching Google Scholar: {str(e)}")

        if self.jstor_client:
            try:
                results['jstor'] = await self.jstor_client.search(query)
            except (ConnectionError, ValueError) as e:
                print(f"Error searching JSTOR: {str(e)}")
                results['jstor'] = []

        if self.wos_client:
            try:
                results['wos'] = await self.wos_client.search(query)
            except (ConnectionError, ValueError) as e:
                print(f"Error searching Web of Science: {str(e)}")
                results['wos'] = []

        return results

    def format_citations(self, results: Dict[str, List[Dict]]) -> List[str]:
        """
        Format search results as citations.

        Args:
            results: Search results from databases

        Returns:
            List of formatted citations
        """
        citations = []

        for pub in results.get('scholar', []):
            try:
                citation = (
                    f"{pub.bib['author']} ({pub.bib['year']}). "
                    f"{pub.bib['title']}. {pub.bib['journal']}"
                )
                citations.append(citation)
            except (KeyError, AttributeError) as e:
                print(f"Error formatting citation: {str(e)}")
                continue

        return citations
