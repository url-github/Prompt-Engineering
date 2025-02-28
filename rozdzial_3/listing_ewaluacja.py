from openai import OpenAI 
import os
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
responses = []
for i in range(10):
  # zwięzły, jeśli iteracja parzysta, rozbudowany, jeśli iteracja nieparzysta
  style = "zwięzły" if i % 2 == 0 else "rozbudowany"
  if style == "zwięzły":
    prompt = f"""Zwróć krótką, zwięzłą odpowiedź na pytanie: Jaki jest sens życia? """
  else:
    prompt = f"""Zwróć rozbudowaną odpowiedź na pytanie: Jaki jest sens życia?"""
  response = client.chat.completions.create( # w tym przykładzie skorzystaj z GPT-3.5 Turbo 
    model="gpt-3.5-turbo", 
    messages=[{"role": "user",
      "content": prompt}])
  responses.append(
    response.choices[0].message.content.strip())
system_prompt = """Oceniasz zwięzłość odpowiedzi czatbota. Odpowiedź liczbą 1 tylko, jeśli odpowiedź jest zwięzła, a 0 — jeśli nie jest zwięzła.
"""
ratings = []
for idx, response in enumerate(responses): 
  rating = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "system",
      "content": system_prompt},
      {"role": "system",
      "content": response}])
  ratings.append(
    rating.choices[0].message.content.strip())
for idx, rating in enumerate(ratings):
  style = "zwięzły" if idx % 2 == 0 else "rozbudowany" 
  print(f"Styl: {style}, ", f"Ocena: {rating}")

