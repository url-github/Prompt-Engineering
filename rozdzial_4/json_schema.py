from openai import OpenAI 
import json
from os import getenv
def schedule_meeting(date, time, attendees):
  # Połącz z usługą kalendarza:
  return { "event_id": "1234", "status": "Planowanie spotkania zakończono sukcesem!", "date": date, "time": time, "attendees": attendees }
OPENAI_FUNCTIONS = {
  "schedule_meeting": schedule_meeting
}

client = OpenAI(api_key=getenv("OPENAI_API_KEY"))
# Schemat JSON naszej predefiniowanej funkcji: 
functions = [
  {
    "type": "function",
    "function": {
      "type": "object",
      "name": "schedule_meeting",
      "description": '''Ustawia spotkanie o określonej dacie i godzinie dla określonych uczestników''',
      "parameters": {
        "type": "object",
        "properties": {
          "date": {"type": "string", "format": "date"},
          "time": {"type": "string", "format": "time"},
          "attendees": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["date", "time", "attendees"],
      },
    }, 
  }
]

# Zacznij konwersację:
messages = [ 
  {
    "role": "user",
    "content": '''Zaplanuj spotkanie dnia 2023-11-01 o godzinie 14:00 z Alicją i Zbyszkiem.''',
  }
]
# Wyślij konwersację i schemat funkcji do modelu:
response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=messages,
    tools=functions,
)
response = response.choices[0].message

# Sprawdź, czy model chce wywołać naszą funkcję: 
if response.tool_calls:
  # Pobierz pierwsze wywołanie funkcji:
  first_tool_call = response.tool_calls[0]
  # Znajdź nazwę i argumenty funkcji do wywołania
  function_name = first_tool_call.function.name
  function_args = json.loads(first_tool_call.function.arguments)
  print("Nazwa funkcji to: ", function_name)
  print("Argumenty funkcji to: ", function_args)
  function = OPENAI_FUNCTIONS.get(function_name) 
  if not function:
    raise Exception(f"Nie znaleziono funkcji {function_name}.") 
    # Wywołaj funkcję:
    function_response = function(**function_args)
    # Przekaż odpowiedź funkcji do modelu:
    messages.append(
        {
            "role": "function",
            "name": "schedule_meeting",
            "content": json.dumps(function_response),
        } 
    )
    # Pozwól modelowi wygenerować odpowiedź przyjazną dla użytkownika:
    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613", messages=messages
    )
    print(second_response.choices[0].message.content)

# Zacznij konwersację:
messages = [ 
  {
    "role": "user",
    "content": '''Zaplanuj spotkanie dnia 2023-11-01 o 14:00 z Alicją i Zbyszkiem. Następnie chcę zaplanować kolejne spotkanie dnia 2023-11-02 o godzinie 15:00 z Justyną i Mateuszem.'''
  } 
]
# Wyślij konwersację i schemat funkcji do modelu:
response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=messages,
  tools=functions,
)

response = response.choices[0].message
# Sprawdź czy model chce wywołać naszą funkcję:
if response.tool_calls:
  for tool_call in response.tool_calls:
    # Pobierz nazwę i argumenty funkcji do wywołania:
      function_name = tool_call.function.name
      function_args = json.loads(tool_call.function.arguments)
      print("To jest nazwa funkcji: ", function_name)
      print("To są argumenty funkcji: ", function_args)
      function = OPENAI_FUNCTIONS.get(function_name) 
      if not function:
        raise Exception(f"Nie znaleziono funkcji {function_name}.") 
        # Wywołaj funkcję:
        function_response = function(**function_args)
        # Przekaż odpowiedź funkcji do modelu:
        messages.append(
          {
             "role": "function",
             "name": function_name,
             "content": json.dumps(function_response),
          } 
        )
# Wygeneruj przyjazną dla użytkownika odpowiedź: 
second_response = client.chat.completions.create(
  model="gpt-3.5-turbo-1106", messages=messages
)
print(second_response.choices[0].message.content)

