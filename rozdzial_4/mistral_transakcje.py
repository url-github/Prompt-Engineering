# przyp. tłum.: kod do ładowania ramki, niezbędny do uruchomienia przykładu!
import pandas as pd
from tqdm import tqdm
import requests
import io
# Dataset URL:
url = "https://storage.googleapis.com/oreilly-content/transaction_data_with_expanded_descriptions.csv"

# Download the file from the URL:
downloaded_file = requests.get(url)

# Load the transactions dataset and only look at 20 transactions:
df = pd.read_csv(io.StringIO(downloaded_file.text))[:20]

# koniec przypisu

import os
from langchain_mistralai.chat_models import ChatMistralAI 
from langchain.output_parsers import PydanticOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from pydantic.v1 import BaseModel
from typing import Literal, Union
from langchain_core.output_parsers import StrOutputParser
# 1. Zdefiniuj model:
mistral_api_key = os.environ["MISTRAL_API_KEY"]
model = ChatMistralAI(model="mistral-small", mistral_api_key=mistral_api_key)
# 2. Zdefiniuj prompt:
system_prompt = """Jesteś ekspertem od analizy bankowych transakcji. Twoim zadaniem jest kategoryzowanie pojedynczych transakcji.
Zawsze zwróć rodzaj i kategorię transakcji:
nie zwracaj wartości None.
Instrukcje dotyczące formatu odpowiedzi:
{format_instructions}"""
user_prompt = """Opis transakcji: 
{transaction}"""
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      system_prompt,
    ),
    (
      "user",
      user_prompt,
    ),
  ] 
)
# 3. Zdefiniuj model Pydantic:
class EnrichedTransactionInformation(BaseModel): 
  transaction_type: Union[
    Literal["Purchase", "Withdrawal", "Deposit",
      "Bill Payment", "Refund"], None 
  ]
  transaction_category: Union[
      Literal["Food", "Entertainment", "Transport", "Utilities", "Rent", "Other"],
      None,
  ]
# 4. Zdefiniuj parser wyjściowy:
output_parser = PydanticOutputParser(
    pydantic_object=EnrichedTransactionInformation)
# 5. Zdefiniuj funkcję do usuwania ukośników wstecznych:
def remove_back_slashes(string):
  # podwójny ukośnik wsteczny pozwala nam zabezpieczyć (escapować) znak 
  cleaned_string = string.replace("\\", "") 
  return cleaned_string
# 6. Utwórz łańcuch LCEL, który poprawi format odpowiedzi.
chain = prompt | model | StrOutputParser() \
| remove_back_slashes | output_parser
transaction = df.iloc[0]["Transaction Description"]
result = chain.invoke(
  {
    "transaction": transaction,
    "format_instructions": \
    output_parser.get_format_instructions(),
  } 
)
# 7. Wywołaj łańcuch dla całego zbioru danych:
results = []
for i, row in tqdm(df.iterrows(), total=len(df)): 
  transaction = row["Transaction Description"] 
  try:
    result = chain.invoke(
      {
        "transaction": transaction,
        "format_instructions": \
        output_parser.get_format_instructions(),
      } 
    )
  except:
    result = EnrichedTransactionInformation(
      transaction_type=None,
      transaction_category=None 
    )
  results.append(result)
# 8. Dodaj wyniki do ramki danych jako kolumny rodzaj i kategoria transakcji:
transaction_types = []
transaction_categories = []
for result in results: 
  transaction_types.append(result.transaction_type) 
  transaction_categories.append(
    result.transaction_category)
df["mistral_transaction_type"] = transaction_types
df["mistral_transaction_category"] = transaction_categories
print(df.head())

