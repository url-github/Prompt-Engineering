from dotenv import load_dotenv
load_dotenv() # wczytaj zmienne środowiskowe z pliku .env.

import ipywidgets as widgets
from IPython.display import display 
import pandas as pd
# Wczytaj plik odpowiedzi.csv
df = pd.read_csv("odpowiedzi.csv") # Wymieszaj ramkę danych
df = df.sample(frac=1).reset_index(drop=True)
# df to Twoja ramka danych, a response to kolumna zawierająca numer tekstu, który chcesz sprawdzić
response_index = 0
# dodaj nową kolumnę w celu przechowania informacji zwrotnej
df['feedback'] = pd.Series(dtype='str')
def on_button_clicked(b):
    global response_index
    # podmień kciuki w górę i w dół na jedynki i zera
    user_feedback = 1 if b.description == "\U0001F44D" else 0
    # zaktalizuj kolumnę feedback 
    df.at[response_index, 'feedback'] = user_feedback
    response_index += 1
    if response_index < len(df):
        update_response() 
    else:
        # zapisz informację zwrotną do pliku CSV
        df.to_csv("results.csv", index=False)
        print("Zakończono testy A/B. Oto wyniki: ") 
        # Policz wynik i liczbę wierszy dla każdego z wariantów 
        summary_df = df.groupby('variant').agg(
                count=('feedback', 'count'),
                score=('feedback', 'mean')).reset_index()
        print(summary_df)
def update_response():
    new_response = df.iloc[response_index]['response'] 
    if pd.notna(new_response):
        new_response = "<p>" + new_response + "</p>" 
    else:
        new_response = "<p>Brak odpowiedzi</p>"
    response.value = new_response
    count_label.value = f"Odpowiedź: {response_index + 1}"
    count_label.value += f"/{len(df)}"
response = widgets.HTML()
count_label = widgets.Label()
update_response()
thumbs_up_button = widgets.Button(description='\U0001F44D')
thumbs_up_button.on_click(on_button_clicked)
thumbs_down_button = widgets.Button(description='\U0001F44E')
thumbs_down_button.on_click(on_button_clicked)
button_box = widgets.HBox([thumbs_down_button, thumbs_up_button])
display(response, button_box, count_label)
