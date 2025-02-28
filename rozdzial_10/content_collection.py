from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.documents import Document
import os
import pandas as pd
from serpapi.google_search import GoogleSearch
from typing import List


class ChromiumLoader(AsyncChromiumLoader):
    async def load(self):
        raw_text = [await self.ascrape_playwright(url) for url in self.urls]
        # Zwróć nieprzetworzone dokumenty:
        return [Document(page_content=text) for text in raw_text]


async def get_html_content_from_urls(
    df: pd.DataFrame, number_of_urls: int = 3, url_column: str = "link"
) -> List[Document]:
    # Pobierz zawartość HTML z pierwszych trzech adresów URL:
    urls = df[url_column].values[:number_of_urls].tolist()

    # Jeśli jest tylko jeden adres URL, przekonwertuj go na listę:
    if isinstance(urls, str):
        urls = [urls]

    # Sprawdź puste adresy URL:
    urls = [url for url in urls if url != ""]

    # Sprawdź duplikaty adresów URL:
    urls = list(set(urls))

    # Rzuć błąd, jeśli znaleziono adresów URL:
    if len(urls) == 0:
        raise ValueError("Nie znaleziono adresów URL!")
    # loader = AsyncHtmlLoader(urls) # Szybsze, ale może nie zawsze zadziałać
    loader = ChromiumLoader(urls)
    docs = await loader.load()
    return docs


def extract_text_from_webpages(documents: List[Document]):
    html2text = Html2TextTransformer()
    return html2text.transform_documents(documents)


async def collect_serp_data_and_extract_text_from_webpages(
    topic: str,
) -> List[Document]:
    search = GoogleSearch(
        {
            "q": topic,
            "location": "Austin, Texas", # należy pozostawić – nie działa przy ustawieniu np. Warszawy
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    # Pobierz wyniki:
    result = search.get_dict()
    # Umieść wyniki w ramce danych Pythona:
    serp_results = pd.DataFrame(result["organic_results"])

    # Pobierz treść HTML pobraną z adresów URL:
    html_documents = await get_html_content_from_urls(serp_results)

    # Wyekstrahuj tekst z dokumentów HTML:
    text_documents = extract_text_from_webpages(html_documents)

    return text_documents
