import asyncio
from langchain_openai.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import gradio as gr
import getpass
import os

# Własne importy:
from content_collection import collect_serp_data_and_extract_text_from_webpages
from custom_summarize_chain import create_all_summaries, DocumentSummary
from expert_interview_chain import InterviewChain
from article_outline_generation import BlogOutlineGenerator
from article_generation import ContentGenerator
from image_generation_chain import create_image

# Sprawdź, czy ustawiona jest zmienna środowiskowa SERPAPI_API_KEY:
os.environ["SERPAPI_API_KEY"] = getpass.getpass("Wprowadź SERPAPI API key: ")


def get_summary(topic):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    try:
        result = new_loop.run_until_complete(async_get_summary(topic))
    finally:
        new_loop.close()

    return result


async def async_get_summary(topic):
    # Wyekstrahuj treść ze stron internetowych do dokumentów LangChain:
    text_documents = await collect_serp_data_and_extract_text_from_webpages(topic=topic)

    # Utwórz podsumowania za pomocą dużego modelu językowego:
    llm = ChatOpenAI(temperature=0)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=7000
    )
    parser = PydanticOutputParser(pydantic_object=DocumentSummary)

    # Utwórz podsumowania:
    print("Tworzenie wszystkich podsumowań...\n---" "")
    summaries = await create_all_summaries(text_documents, parser, llm, text_splitter)

    # Utwórz pytania do wywiadu:
    print("Tworzenie pytań do wywiadu...\n---" "")
    interview_chain = InterviewChain(topic=topic, document_summaries=summaries)
    interview_questions = interview_chain()

    # Wyekstrahuj jedynie pytania:
    interview_questions = [
        question.question for question in interview_questions.questions
    ]
    question_one = interview_questions[0]
    question_two = interview_questions[1]
    question_three = interview_questions[2]
    question_four = interview_questions[3]
    question_five = interview_questions[4]

    # Zaktualizuj interfejs użytkownika gradio:
    return [
        text_documents,
        summaries,
        interview_questions,
        question_one,
        question_two,
        question_three,
        question_four,
        question_five,
    ]


def generate_content(topic, summaries, text_documents):
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)

    # Parsowanie wejść skonwertowanych do postaci tekstów:
    summaries = eval(summaries)
    text_documents = eval(text_documents)

    try:
        result = new_loop.run_until_complete(
            async_generate_content(topic, text_documents, summaries)
        )
    finally:
        new_loop.close()

    return result


with gr.Blocks() as demo:
    with gr.Row():
        topic = gr.Textbox(label="Temat", scale=85, value="Memetyka")
        summarize_btn = gr.Button("Podsumuj oraz wygeneruj pytania", scale=15)

    with gr.Row():
        summaries = gr.Textbox(label="Podsumowanie", lines=10)
        interview_questions = gr.Textbox(label="Pytania", lines=10)
        text_documents = gr.Textbox(label="Dokumenty tekstowe", lines=20, visible=False)

    with gr.Column(scale=4) as output_col:
        question_one = gr.Textbox(label="Pytanie 1", lines=1)
        answer_one = gr.Textbox(label="Odpowiedź 1", lines=1, interactive=True)
        question_two = gr.Textbox(label="Pytanie 2", lines=1)
        answer_two = gr.Textbox(label="Odpowiedź 2", lines=1, interactive=True)
        question_three = gr.Textbox(label="Pytanie 3", lines=1)
        answer_three = gr.Textbox(label="Odpowiedź 3", lines=1, interactive=True)
        question_four = gr.Textbox(label="Pytanie 4", lines=1)
        answer_four = gr.Textbox(label="Odpowiedź 4", lines=1, interactive=True)
        question_five = gr.Textbox(label="Pytanie 5", lines=1)
        answer_five = gr.Textbox(label="Odpowiedź 5", lines=1, interactive=True)

        async def async_generate_content(topic, text_documents, summaries):
            questions_and_answers = {
                question_one: answer_one,
                question_two: answer_two,
                question_three: answer_three,
                question_four: answer_four,
                question_five: answer_five,
            }

            # Zarys artykułu:
            blog_outline_generator = BlogOutlineGenerator(
                topic=topic, questions_and_answers=questions_and_answers
            )
            questions_and_answers = blog_outline_generator.questions_and_answers
            outline_result = blog_outline_generator.generate_outline(summaries)

            # Generowanie treści artykułu:
            content_gen = ContentGenerator(
                topic=topic,
                outline=outline_result,
                questions_and_answers=questions_and_answers,
            )
            content_gen.split_and_vectorize_documents(text_documents)
            generated_text = content_gen.generate_blog_post()

            # Treść zastępcza do generowania obrazu i promptu:
            generated_image = create_image(title=outline_result.title)
            # Przekonwertuj na obraz w formacie PIL:
            import PIL.Image as Image

            generated_image = Image.open(generated_image[0])
            return generated_text, generated_image

    with gr.Row():
        summarize_btn.click(
            fn=get_summary,
            inputs=[topic],
            outputs=[
                text_documents,
                summaries,
                interview_questions,
                question_one,
                question_two,
                question_three,
                question_four,
                question_five,
            ],
        )

    clear_btn = gr.Button("Wyczyść wszystkie odpowiedzi", scale=15)
    generate_btn = gr.Button("Wygeneruj post i obrazek na bloga", scale=30)

    with gr.Row():
        with gr.Column():
            generated_content = gr.Textbox(
                label="Treść", lines=15, show_copy_button=True
            )
            generated_image = gr.Image()

        generate_btn.click(
            fn=generate_content,
            inputs=[topic, summaries, text_documents],
            outputs=[generated_content, generated_image],
        )

        # Zresetuj elementy interfejsu do wartości domyślnych
        clear_btn.click(
            fn=lambda: (
                "",  # topic
                "",  # summaries
                "",  # text_documents
                "",  # interview_questions
                "",  # generated_content
                None,  # generated_image
                "",  # question_one
                "",  # answer_one
                "",  # question_two
                "",  # answer_two
                "",  # question_three
                "",  # answer_three
                "",  # question_four
                "",  # answer_four
                "",  # question_five
                "",  # answer_five
            ),
            inputs=[],
            outputs=[
                topic,
                summaries,
                text_documents,
                interview_questions,
                generated_content,
                generated_image,
                question_one,
                answer_one,
                question_two,
                answer_two,
                question_three,
                answer_three,
                question_four,
                answer_four,
                question_five,
                answer_five,
            ],
        )

demo.launch()
