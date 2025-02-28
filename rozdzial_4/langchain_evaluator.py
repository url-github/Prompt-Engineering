# przyp. tłum.: kod do ładowania ramki, niezbędny do uruchomienia przykładu!
import os
from langchain_mistralai.chat_models import ChatMistralAI 
from langchain.output_parsers import PydanticOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from pydantic.v1 import BaseModel
from typing import Literal, Union
from langchain_core.output_parsers import StrOutputParser

class EnrichedTransactionInformation(BaseModel): 
  transaction_type: Union[
    Literal["Purchase", "Withdrawal", "Deposit",
      "Bill Payment", "Refund"], None
  ]
  transaction_category: Union[
      Literal["Food", "Entertainment", "Transport", "Utilities", "Rent", "Other"], 
          None,
  ]


output_parser = PydanticOutputParser(
    pydantic_object=EnrichedTransactionInformation)

system_prompt = """You are are an expert at analyzing bank transactions, 
you will be categorising a single transaction. 
Always return a transaction type and category: do not return None.
Format Instructions:
{format_instructions}"""

user_prompt = """Transaction Text:
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

from langchain_openai.chat_models import ChatOpenAI

# 1. Define the model:
model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    model_kwargs={"response_format": {"type": "json_object"}},
)

chain = prompt | model | output_parser

# 2. Invoke the chain for the first transaction:
transaction = df.iloc[0]["Transaction Description"]
result = chain.invoke(
        {
            "transaction": transaction,
            "format_instructions": output_parser.get_format_instructions(),
        }
    )
print(result)
# 3. Invoke the chain for the whole dataset:
results = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    transaction = row["Transaction Description"]
    try:
        result = chain.invoke(
            {
                "transaction": transaction,
                "format_instructions": output_parser.get_format_instructions(),
            }
        )
    except:
        result = EnrichedTransactionInformation(
            transaction_type=None, transaction_category=None
        )
    
    results.append(result)

# 4. Add the results to the dataframe, as columns transaction type and transaction category
transaction_types = []
transaction_categories = []

for result in results:
    transaction_types.append(result.transaction_type)
    transaction_categories.append(result.transaction_category)

df["gpt3.5_transaction_type"] = transaction_types
df["gpt3.5_transaction_category"] = transaction_categories

df.head()
# Loop through the dataframe and evaluate the predictions
transaction_types = []
transaction_categories = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    transaction_type = row["transaction_type"]
    predicted_transaction_type = row["gpt3.5_transaction_type"]
    transaction_category = row["transaction_category"]
    predicted_transaction_category = row["gpt3.5_transaction_category"]

    transaction_type_score = evaluator.evaluate_strings(
        prediction=predicted_transaction_type,
        reference=transaction_type,
    )

    transaction_category_score = evaluator.evaluate_strings(
        prediction=predicted_transaction_category,
        reference=transaction_category,
    )

    transaction_types.append(transaction_type_score)
    transaction_categories.append(transaction_category_score)

accuracy_score = 0

for transaction_type_score, transaction_category_score in zip(
    transaction_types, transaction_categories
):
    accuracy_score += transaction_type_score['score'] + transaction_category_score['score']

accuracy_score = accuracy_score / (len(transaction_types) * 2)
print(f"Accuracy score: {accuracy_score}")
# koniec przypisu
# Oceń odpowiedzi korzystając z ewaluatorów LangChain:
from langchain.evaluation import load_evaluator 
evaluator = load_evaluator("labeled_pairwise_string")
row = df.iloc[0]
transaction = row["Transaction Description"]
gpt3pt5_category = row["gpt3.5_transaction_category"]
gpt3pt5_type = row["gpt3.5_transaction_type"]
mistral_category = row["mistral_transaction_category"]
mistral_type = row["mistral_transaction_type"]
reference_category = row["transaction_category"]
reference_type = row["transaction_type"]
# Umieść dane w formacie JSON dla ewaluatora:
gpt3pt5_data = f"""{{
  "transaction_category": "{gpt3pt5_category}", 
  "transaction_type": "{gpt3pt5_type}"
}}"""
mistral_data = f"""{{
  "transaction_category": "{mistral_category}", 
  "transaction_type": "{mistral_type}"
}}"""
reference_data = f"""{{
  "transaction_category": "{reference_category}", 
  "transaction_type": "{reference_type}"
}}"""
# Ustaw prompt wejściowy dla kontekstu ewaluatora:
input_prompt = """Jesteś ekspertem od analizy bankowych transakcji. Twoim zadaniem jest kategoryzowanie pojedynczych transakcji.
Zawsze zwróć rodzaj i kategorię transakcji:
nie zwracaj wartości None.
Instrukcje dotyczące formatu odpowiedzi:
{format_instructions}"""
user_prompt = """Opis transakcji: 
{transaction}"""
transaction_types.append(transaction_type_score)
transaction_categories.append(
  transaction_category_score)
accuracy_score = 0
for transaction_type_score, transaction_category_score in zip(
  transaction_types, transaction_categories):
  accuracy_score += transaction_type_score['score'] + \
  transaction_category_score['score']
accuracy_score = accuracy_score / (len(transaction_types) * 2)
print(f"Dokładność: {accuracy_score}")
evaluator.evaluate_string_pairs(
  prediction=gpt3pt5_data,
  prediction_b=mistral_data,
  input=input_prompt.format(
    format_instructions=output_parser.get_format_instructions(),
    transaction=transaction),
  reference=reference_data,
)

