from langchain.chains import LLMChain
from typing import List, Dict, Any
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage


class OnlyStoreAIMemory(ConversationSummaryBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_ai_message(output_str)


class ContentGenerator:
    def __init__(
        self,
        topic: str,
        outline: Any,
        questions_and_answers: dict,
        chunk_size: int = 400,
        chunk_overlap: int = 100,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.topic = topic
        self.outline = outline
        self.questions_and_answers = questions_and_answers

        prompt = f"""
        Jesteś autorem treści SEO.
        Aktualnie piszesz post na bloga na temat: {self.topic}.
        Oto zarys artykułu: {self.outline.json()}. Masz napisać sekcje posta na bloga w języku polskim.
        ---
        SKorzystaj z poprzednich komunikatów AI, aby uniknąć powtarzania się gdy będziesz przepisywać sekcje posta na bloga.
        """
        chat = ChatOpenAI(model="gpt-3.5-turbo-16k")
        memory = OnlyStoreAIMemory(
            llm=chat,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=1200,
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        self.blog_post_chain = LLMChain(
            llm=chat, prompt=chat_prompt, memory=memory, output_key="blog_post"
        )
        self.chroma_db = None

    def split_and_vectorize_documents(self, text_documents):
        chunked_docs = self.text_splitter.split_documents(text_documents)
        self.chroma_db = Chroma.from_documents(chunked_docs, embedding=self.embeddings)  # type: ignore
        return self.chroma_db

    def generate_blog_post(self) -> List[str]:
        blog_post = []
        print("Generowanie posta na bloga...\n---")
        for subheading in self.outline.sub_headings:
            k = 5  # Inicjalizuj k 
            while k >= 0:
                try:
                    relevant_documents = self.chroma_db.as_retriever().invoke(  # type: ignore
                        subheading.title, k=k
                    )
                    section_prompt = f"""
                    Obecnie piszesz sekcję: {subheading.title}
                    Oto odpowiednie dokumenty dla tej sekcji: {relevant_documents}. Jeśli te dokumenty nie są przydatne, możesz je zignorować. Nigdy nie możesz kopiować treści z tych dokumentów, ponieważ jest to plagiat.
                    Oto istotne spostrzeżenia, które zebraliśmy z naszych pytań i odpowiedzi z wywiadu: {self.questions_and_answers}. Musisz uwzględnić te spostrzeżenia tam, gdzie to możliwe, ponieważ są one ważne i pomogą naszej treści osiągnąć lepsze pozycjonowanie.
                    Musisz przestrzegać następujących zasad:
                    - Musisz napisać sekcję: {subheading.title}
                    - Wyrenderuj wynik w formacie .md
                    - Uwzględnij odpowiednie formaty odpowiedzi, takie jak wypunktowania, listy numerowane itp.
                    - Generuj treści w języku polskim.
                    ---
                    Treść sekcji:

                    """
                    result = self.blog_post_chain.predict(human_input=section_prompt)
                    blog_post.append(result)
                    break
                except Exception as e:
                    print(f"Wystąpił błąd: {e}")
                    k -= 1
                if k < 0:
                    print(
                        "Wszystkie próby pobrania istotnych dokumentów nie powiodły się. Stosuję pusty tekst jako wartość parametru relevant_documents."
                    )
                    relevant_documents = ""
        print("Zakończono generowanie posta na bloga!\n---")
        return blog_post
