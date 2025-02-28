from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
import re

chat = ChatOpenAI(model_kwargs={"stop": ["tool_result:"],})

tools = {}
def search_on_google(query: str):
  return f"Jason Derulo nie ma żony ani partnerki."
tools["search_on_google"] = {
  "function": search_on_google,
  "description": "Wyszukuje zapytanie w wyszukiwarce Google",
}

def extract_last_action_and_input(text):
  # Skompiluj wzorce wyrażeń regularnych
  action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
  action_input_pattern = re.compile(r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE)
  # Znajdź wszystkie wystąpienia wyrażeń action i action_input
  actions = action_pattern.findall(text)
  action_inputs = action_input_pattern.findall(text)
  # Znajdź ostatnie wystąpienie wyrażeń action i action_input
  last_action = actions[-1] if actions else None
  last_action_input = action_inputs[-1] if action_inputs else None
  return {"action": last_action, "action_input": last_action_input}

def extract_final_answer(text):
  final_answer_pattern = re.compile(r"(?i)Znalazłem ostateczną odpowiedź:\s*([^\n]+)", re.MULTILINE)
  final_answers = final_answer_pattern.findall(text)
  if final_answers:
    return final_answers[0]
  else:
    return None

base_prompt = """
Spróbujesz rozwiązać problem polegający na znalezieniu odpowiedzi na pytanie. Aby znaleźć rozwiązanie problemu, używaj łańcucha myśli, stosując następujący schemat:
1. Przeanalizuj oryginalne pytanie: 
original_question: original_problem_text 
2. Utwórz obserwację według poniższego wzoru: 
observation: observation_text 
3. Na podstawie obserwacji wytwórz myśl zgodnie z tym wzorem: 
thought: thought_text 
4. Skorzystaj z narzędzi, aby działać według myśli, używając poniższego wzoru: 
action: tool_name 
action_input: tool_input 
Nie zgaduj ani nie zakładaj wyników narzędzia, zamiast tego dostarcz wyniki w zorganizowanej formie, która zawiera działanie i wejście do działania.
Masz dostęp do następujących narzędzi: {tools}.
original_problem: {question}
W oparciu o wynik dostarczonego narzędzia:
Podaj kolejną obserwację, działanie, dane wejściowe do działania lub finalną odpowiedź, jeśli jest dostępna.
Jeśli przedstawiasz finalną odpowiedź, musisz zwrócić ją zgodnie z wzorem: "Znalazłem odpowiedź: final_answer    
"""

output = chat.invoke(SystemMessagePromptTemplate \
.from_template(template=base_prompt) \
.format_messages(tools=tools, question="Czy Jason Derulo ma partnerkę?"))
print(output)

tool_name = extract_last_action_and_input(output.content)["action"]
tool_input = extract_last_action_and_input(output.content)["action_input"]
tool_result = tools[tool_name]["function"](tool_input)

print(f"""Agent wybrał do swojego działania następujące narzędzie:
  tool_name: {tool_name}
  tool_input: {tool_input}
  tool_result: {tool_result}"""
)

current_prompt = """
Odpowiadasz na pytanie: Czy Jason Derulo ma partnera lub partnerkę?
Na podstawie wyniku narzędzia:
tool_result: {tool_result}
Przedstaw kolejną obserwację, akcję, informacje wejściowe lub finalną odpowiedź, jeśli nią dysponujesz. Jeśli przedstawiasz finalną odpowiedź, musisz skorzystać z następującego wzorca: "Znalazłem odpowiedź: final_answer."
"""

output = chat.invoke(SystemMessagePromptTemplate. \
from_template(template=current_prompt) \
.format_messages(tool_result=tool_result))

print("----------\n\nOdpowiedź modelu to:", output.content) 
final_answer = extract_final_answer(output.content)
if final_answer:
  print(f"odpowiedź: {final_answer}") 
else:
  print("Nie znaleziono odpowiedzi.")

