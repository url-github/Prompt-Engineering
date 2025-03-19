from langchain_core.prompts.chat import ChatPromptTemplate
character_generation_prompt = ChatPromptTemplate.from_template(
  """ Chciałbym, żebyś wymyślił od trzech do pięciu postaci do mojego krótkiego opowiadania. Gatunek to {genre}. Każda postać musi mieć Imię i Biografię. Musisz dostarczyć imię i biografię dla każdej postaci, to jest bardzo ważne!
  ---
  Przykładowa odpowiedź:
  Imię: Posczar, Biografia: Czarodziej, który jest mistrzem magii.
  Imię: Poswoj, Biografia: Wojownik, który jest mistrzem miecza.
  ---
  Postaci: """
)
plot_generation_prompt = ChatPromptTemplate.from_template(
"""Zakładając dane postaci i gatunek, wygeneruje ciekawą fabułę dla krótkiej historii: 
Postaci:
{characters}
---
Gatunek: {genre}
---
Fabuła: """
)
scene_generation_plot_prompt = ChatPromptTemplate.from_template( """Działaj jako efektywny twórca treści. Zakładając dane postaci i fabułę, jesteś odpowiedzialna za wygenerowanie wielu scen dla każdego fragmen. Musisz podzielić fabułę na wiele scen:
 ---
Postaci:
{characters}
---
Gatunek: {genre}
---
Fabuła: {plot}
---
Przykładowa odpowiedź:
Sceny:
Scena 1: To jest przykładowy tekst.
Scena 2: To jest przykładowy tekst.
Scena 3: To jest przykładowy tekst.
----
Sceny:
"""
)
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
chain = RunnablePassthrough() | {
  "genre": itemgetter("genre"),
}
chain.invoke({"genre": "fantasy"}) # {'genre': 'fantasy'}
from langchain_core.runnables import RunnableLambda
chain = RunnablePassthrough() | {
  "genre": itemgetter("genre"),
  "upper_case_genre": lambda x: x["genre"].upper(), 
  "lower_case_genre": RunnableLambda(lambda x: x["genre"].lower()),
}
chain.invoke({"genre": "fantasy"})
# {'genre': 'fantasy', 'upper_case_genre': 'FANTASY', # 'lower_case_genre': 'fantasy'}
from langchain_core.runnables import RunnableParallel
master_chain = RunnablePassthrough() | {
  "genre": itemgetter("genre"),
  "upper_case_genre": lambda x: x["genre"].upper(), 
  "lower_case_genre": RunnableLambda(lambda x: x["genre"].lower()),
}
master_chain_two = RunnablePassthrough() | RunnableParallel(
  genre=itemgetter("genre"),
  upper_case_genre=lambda x: x["genre"].upper(), 
  lower_case_genre=RunnableLambda(lambda x: x["genre"].lower()),
)
story_result = master_chain.invoke({"genre": "Fantasy"})
print(story_result)
story_result = master_chain_two.invoke({"genre": "Fantasy"})
print(story_result)
# master chain: {'genre': 'Fantasy', 'upper_case_genre': 'FANTASY',
# 'lower_case_genre': 'fantasy'}
# master chain two: {'genre': 'Fantasy', 'upper_case_genre': 'FANTASY',
# 'lower_case_genre': 'fantasy'}
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# Utwórz model czatowy:
model = ChatOpenAI()
# Utwórz podłańcuchy:
character_generation_chain = ( character_generation_prompt
| model
| StrOutputParser() )
plot_generation_chain = ( plot_generation_prompt
| model
| StrOutputParser() )
scene_generation_plot_chain = ( scene_generation_plot_prompt
| model
| StrOutputParser()  )
from langchain_core.runnables import RunnableParallel 
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
master_chain = (
  {"characters": character_generation_chain, "genre":
  RunnablePassthrough()}
  | RunnableParallel(
    characters=itemgetter("characters"),
    genre=itemgetter("genre"),
    plot=plot_generation_chain,
  )
  | RunnableParallel(
    characters=itemgetter("characters"),
    genre=itemgetter("genre"),
    plot=itemgetter("plot"),
    scenes=scene_generation_plot_chain,
  )
 )
story_result = master_chain.invoke({"genre": "Fantasy"})

# Dokonujemy ekstrakcji scen wywołując metodę .split('\n') i usuwając puste teksty.
scenes = [scene for scene in story_result["scenes"].split("\n") if scene] 
generated_scenes = []
previous_scene_summary = ""
character_script_prompt = ChatPromptTemplate.from_template(
  template="""Zakładając następujące postaci: {characters} i gatunek: {genre}, utwórz dobry skrypt postaci dla tej sceny.
  Musisz przyjać następujące założenia:
  - Skorzystaj z podsumowania poprzedniej sceny: {previous_scene_summary}, aby uniknąć powtarzania się.
  - Skorzystaj z fabuły: {plot}, aby utworzyć efektywny skrypt postaci dla danej sceny
  - Aktualnie generujesz skrypt dialogowy postaci dla sceny: {scene}
  ---
  Oto przykładowa odpowiedź:
  SCENA 1: MIESZKANIE ANNY
  (ANNA sortuje stare książki, gdy nagle słychać pukanie do drzwi.
  Otwiera je i widzi JANA.)
  ANNA: Czy mogę panu jakoś pomóc?
  JAN: Być może, myślę, że to ja mogę pomóc pani. Słyszałem, że
  prowadzi pani badania nad podróżami w czasie.
  (Anna wygląda na zaintrygowaną, ale również ostrożną.)
  ANNA: To prawda, ale skąd pan wie?
  JAN: Można by powiedzieć... że jestem świadkiem naocznym.
  ---
  NUMER SCENY: {index}
  """,
)
summarize_prompt = ChatPromptTemplate.from_template(
  template="""Zakładając skrypt postaci, utwórz podsumowanie sceny w języku polskim.        
  Skrypt postaci: {character_script}""",
)

# Wczytywanie modelu czata:
model = ChatOpenAI(model='gpt-3.5-turbo-16k')
# Utwórz łańcuchy LCEL:
character_script_generation_chain = (
  {
    "characters": RunnablePassthrough(),
    "genre": RunnablePassthrough(),
    "previous_scene_summary": RunnablePassthrough(),
    "plot": RunnablePassthrough(),
    "scene": RunnablePassthrough(),
    "index": RunnablePassthrough(),
  }
  | character_script_prompt
  | model
  | StrOutputParser()
)
summarize_chain = summarize_prompt | model | StrOutputParser()
# Możesz skorzystać z funkcji tqdm do śledzenia postępu lub skorzystać ze wszystkich scen:
for index, scene in enumerate(scenes[0:3]):
        # # Wygeneruj sceny:
        scene_result = character_script_generation_chain.invoke(
          {
           "characters": story_result["characters"],
           "genre": "fantasy",
           "previous_scene_summary": previous_scene_summary,
           "index": index,
          } 
        )
        # Przechowaj wygenerowane sceny:
        generated_scenes.append(
            {"character_script": scene_result, "scene": scenes[index]}
        )
        # Jeśli ta scena jest pierwszą, nie dysponujemy podsumowaniem poprzedniej sceny:
        if index == 0:
          previous_scene_summary = scene_result  
        else:
          # Jeśli jest to scena druga lub dalsza, możemy wygenerować podsumowanie:
          summary_result = summarize_chain.invoke(
           {"character_script": scene_result}
          )
          previous_scene_summary = summary_result

print(generated_scenes)
from langchain.text_splitter import CharacterTextSplitter 
from langchain.chains.summarize import load_summarize_chain 
import pandas as pd
df = pd.DataFrame(generated_scenes)
all_character_script_text = "\n".join(df.character_script.tolist())
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=200
)
docs = text_splitter.create_documents([all_character_script_text])
chain = load_summarize_chain(llm=model, chain_type="map_reduce")
summary = chain.invoke(docs)
print(summary['output_text'])

