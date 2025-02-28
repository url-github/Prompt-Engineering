# Standard libraries
from pydantic.v1 import BaseModel, Field
from typing import List, Any

# Langchain libraries
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.runnables import RunnableParallel


class Question(BaseModel):
    """Single Output - A question with no answer"""

    question: str = Field(None, description="An interview question to ask.")
    answer: None = None


class InterviewQuestions(BaseModel):
    """Output for Interview questions"""

    questions: List[Question] = Field(
        ..., min_items=5, max_items=5, description="List of interview questions."
    )


class InterviewChain:
    def __init__(self, topic: str, document_summaries: Any):
        self.topic = topic
        self.llm = ChatOpenAI(temperature=0)
        self.document_summaries = document_summaries

    def __call__(self) -> Any:
        # Create an LLM:
        model = ChatOpenAI(temperature=0.6)

        # Ustaw parser i wstrzyknij instrukcje do szablonu promptu:
        parser: PydanticOutputParser = PydanticOutputParser(
            pydantic_object=InterviewQuestions
        )

        system_message = """Jesteś analitykiem SEO. Wcześniej podsumowałeś i wyciągnąłeś kluczowe punkty z wyników SERP. Uzyskane informacje zostaną wykorzystane do badania treści, a my porównamy kluczowe punkty, spostrzeżenia i podsumowania z różnych artykułów. Teraz przeprowadzisz wywiad z ekspertem ds. treści. Będziesz zadawać mu pytania na następujący temat: {topic}. Prowadź wywiad w języku polskim.
Musisz przestrzegać następujących zasad:
- Przedstaw listę pytań, które zadałbyś ekspertowi ds. treści na dany temat.
- Musisz zadać co najmniej 5 i maksymalnie 5 pytań.
- Szukasz informacji, które przyniosą dodatkową wartość i unikalne spostrzeżenia, które nie są już zawarte w {document_summaries}.
- Musisz zadawać pytania otwarte, a nie takie, na które można odpowiedzieć "tak" lub "nie". {format_instructions}

        """

        system_prompt = SystemMessagePromptTemplate.from_template(system_message)
        human_prompt = HumanMessagePromptTemplate.from_template(
            """Przedstaw mi pięć pierwszych pytań"""
        )

        # Utwórz prompt:
        prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

        chain = (
            RunnableParallel(
                topic=lambda x: self.topic,
                document_summaries=lambda x: self.document_summaries,
                format_instructions=lambda x: parser.get_format_instructions(),
            )
            | prompt
            | model
        )

        # Uruchom czat:
        result = chain.invoke(
            {
                "topic": self.topic,
                "document_summaries": self.document_summaries,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # Sparsuj odpowiedź LLM:
        return parser.parse(result.content)
