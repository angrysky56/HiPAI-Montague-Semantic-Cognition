"""
Data source module for retrieving text from external URLs.
"""
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

import aiohttp
from bs4 import BeautifulSoup


class BaseSource(ABC):
    """
    Abstract base class for all data ingestion sources.
    """

    @abstractmethod
    async def get_texts(self) -> AsyncGenerator[str]:
        """
        Asynchronously yields chunks of text or sentences from the source.
        """
        yield ""  # For type checking only


class TextSource(BaseSource):
    """
    A simple source that yields items from an in-memory list of strings.
    """
    def __init__(self, texts: list[str]):
        self.texts = texts

    async def get_texts(self) -> AsyncGenerator[str]:
        for text in self.texts:
            yield text


class WebScraperSource(BaseSource):
    """
    A simple source that fetches a URL and yields its text content.
    """
    def __init__(self, url: str):
        self.url = url

    async def get_texts(self) -> AsyncGenerator[str]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(self.url) as response:
                    response.raise_for_status()
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text(separator=' ', strip=True)
                    yield text
        except Exception as e:
            print(f"Failed to fetch {self.url}: {e}")
