import asyncio
from langchain.callbacks import get_openai_callback 
from langchain_core.messages import SystemMessage 
from langchain_openai.chat_models import ChatOpenAI 
model = ChatOpenAI()

async def fun():
  with get_openai_callback() as cb:
    await asyncio.gather(
      model.agenerate(
        [
          [SystemMessage(content="Czy sens życia to liczba 42?")],
          [SystemMessage(content="Czy sens życia to liczba 42?")],
        ],
      )
    )
  print(cb.__dict__)

loop = asyncio.get_event_loop()
tasks = [
  loop.create_task(fun())
]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
