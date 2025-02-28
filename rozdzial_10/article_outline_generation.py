from typing import List, Any
from pydantic.v1 import BaseModel


class SubHeading(BaseModel):
    title: str # Każdy podnagłówek powinien mieć tytuł


class BlogOutline(BaseModel):
    title: str
    sub_headings: List[SubHeading] # Zarys ma wiele podnagłówków


# Biblioteki Langchain:
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai.chat_models import ChatOpenAI

# Typy własne:
from custom_summarize_chain import DocumentSummary


class BlogOutlineGenerator:
    def __init__(self, topic: str, questions_and_answers: Any):
        self.topic = topic
        self.questions_and_answers = questions_and_answers

        # Utwórz prompt
        prompt_content = """
        Utwórz zarys na bloga na podstawie moich odpowiedzi i podsumowania w języku polskim.
        Artykuł na temat {topic}.
        temat: {topic}
        podsumowania: {document_summaries}
        ---
        Oto wywiad, na który odpowiedziałem:
        {interview_questions_and_answers}
        ---
        Format odpowiedzi: {format_instructions}
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            prompt_content
        )
        self.chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        # Utwórz parser odpowiedzi
        self.parser = PydanticOutputParser(pydantic_object=BlogOutline)

        # Ustaw łańcuch
        self.outline_chain = self.chat_prompt | ChatOpenAI() | self.parser

    def generate_outline(self, summaries: List[DocumentSummary]) -> Any:
        print("Generowanie zarysu...\n---")
        result = self.outline_chain.invoke(
            {
                "topic": self.topic,
                "document_summaries": [s.dict() for s in summaries],
                "interview_questions_and_answers": self.questions_and_answers,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        print("Zakończono generowanie zarysu!\n---")
        return result
