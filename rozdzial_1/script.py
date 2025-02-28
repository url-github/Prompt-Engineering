from dotenv import load_dotenv
load_dotenv() # wczytaj zmienne środowiskowe z pliku .env.

# Zdefiniuj dwa warianty prompta, aby porównać podejście bez przykładów z podejściem z kilkoma przykładami
prompt_A = """Opis produktu: Buty, które pasują dla każdego rozmiaru stopy.
Słowa początkowe: dostosowanie, dopasowanie, uniwersalny rozmiar
Nazwy produktów:"""
prompt_B = """Opis produktu: Automat do przygotowania mlecznych koktajli w domu.
Słowa początkowe: szybki, zdrowy, kompaktowy.
Nazwy produktów: HomeShaker, Fit Shaker, QuickShake, Shake Maker
Opis produktu: Zegarek, który podaje dokładny czas w przestrzeni kosmicznej
Słowa początkowe: astronauta, odporne na przestrzeń kosmiczną, orbita eliptyczna
Nazwy produktów: AstroTime, SpaceGuard, Orbit-Accurate, EliptoTime.
Opis produktu: Buty, które pasują dla każdego rozmiaru stopy.

Słowa początkowe: dostosowanie, dopasowanie, uniwersalny rozmiar
Nazwy produktów:"""
test_prompts = [prompt_A, prompt_B]
import pandas as pd
from openai import OpenAI 
import os
# Przekaż swój klucz OpenAI jako zmienną środowiskową 
# https://platform.openai.com/api-keys
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'], # Domyślnie 
)
def get_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Jesteś pomocnym asystentem."
            }, 
            {
                "role": "user",
                "content": prompt
            }
         ])
    return response.choices[0].message.content

# Przejdź przez wszystkie prompty i pozyskaj odpowiedzi
responses = []
num_tests = 5
for idx, prompt in enumerate(test_prompts): 
    # numer prompta jako litera 
    var_name = chr(ord('A') + idx)
    for i in range(num_tests):
        # Pobierz odpowiedź z modelu
        response = get_response(prompt)
        data = {
            "variant": var_name,
            "prompt": prompt,
            "response": response
            }
        responses.append(data)
# Skonwertuj odpowiedzi do obiektu ramki danych
df = pd.DataFrame(responses)
# Zapisz ramkę danych do pliku CSV
df.to_csv("odpowiedzi.csv", index=False)
print(df)

