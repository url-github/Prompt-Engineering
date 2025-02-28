import re

text = """
Action: search_on_google
Action_input: Obecna żona Toma Hanksa
action: search_on_wikipedia
action_input: Ile lat miała Rita Wilson w roku 2023?
action : search_on_google
action input: jakieś inne zapytanie
"""


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
final_answer_text = "Znalazłem ostateczną odpowiedź: final_answer" 
print(extract_final_answer(final_answer_text))

