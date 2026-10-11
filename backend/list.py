"""Run from the backend folder:  python list_models.py
Prints the model IDs your Groq key can actually use."""
from groq import Groq

from app.config import settings

for m in sorted(Groq(api_key=settings.llm_api_key).models.list().data, key=lambda m: m.id):
    print(m.id)