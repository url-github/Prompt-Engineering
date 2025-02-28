import asyncio
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic.v1 import BaseModel
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from typing import Dict, List, Optional, Union
import time
from langchain_core.prompts import PromptTemplate


class DocumentSummary(BaseModel):
    concise_summary: str
    writing_style: str
    key_points: List[str]
    expert_opinions: Optional[List[str]] = None
    metadata: Optional[
        Dict[str, str]
    ] = {}  # Pochodzi natywnie z ładowarki dokumentów LangChain


async def create_summary_from_text(
    document: Document,
    parser: PydanticOutputParser,
    llm: ChatOpenAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> Union[DocumentSummary, None]:
    # Podziel dokument-rodzica na fragmenty:
    split_docs = text_splitter.split_documents([document])

    # Jeśli nie ma dokumentów po podziale, zwróć None:
    if len(split_docs) == 0:
        return None

    # Pobierz pierwszy dokument, który będzie jedynym użytym do podsumowań:
    first_document = split_docs[0]

    # Uruchom rafinowanny łańcuch podsumowania, który wydobędzie unikatowe założenia i opinie wyrażone w artykule:
    prompt_template = """Jesteś analitykiem SEO. Twoim zadaniem jest podsumowanie i wyodrębnienie kluczowych punktów z poniższego tekstu. Uzyskane informacje zostaną wykorzystane do badań nad treścią, a my porównamy kluczowe punkty, spostrzeżenia i podsumowania z wielu artykułów.
    --- 
    - Musisz przeanalizować tekst i wyodrębnić kluczowe punkty oraz opinie z poniższego tekstu. 
    - Musisz wyodrębnić kluczowe punkty i opinie z poniższego tekstu: 
    {text} 
    {format_instructions}
    Dokument musi zawierać odpowiedź zgodnie z określonym formatem JSON.
    """
    print("first")
    print(first_document)
    # Zdefiniuj łańcuch LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

    print("Podsumowanie danych!")

    # Start time:
    start_time = time.time()
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="text",
    )
    summary_result = await stuff_chain.ainvoke(
        {
            "input_documents": [first_document],
            "format_instructions": parser.get_format_instructions(),
        },
    )
    print(f"Czas trwania operacji: {time.time() - start_time}")
    print("Zakończono podsumowanie danych!\n---" "")
    print(summary_result["output_text"])
    document_summary = parser.parse(summary_result["output_text"])
    document_summary.metadata = document.metadata
    return document_summary


async def create_all_summaries(
    text_documents: List[Document],
    parser: PydanticOutputParser,
    llm: ChatOpenAI,
    text_splitter: RecursiveCharacterTextSplitter,
) -> List[DocumentSummary]:
    # Utwórz tablię korutyn:
    tasks = [
        create_summary_from_text(document, parser, llm, text_splitter)
        for document in text_documents
    ]

    # Wykonaj zadania współbieżnie i zbierz wszystkie wyniki:
    results = await asyncio.gather(*tasks)

    # Odfiltruj wartości None
    summaries = [summary for summary in results if summary is not None]

    if len(summaries) == 0:
        raise ValueError("Nie utworzono podsumowań!")

    return summaries
