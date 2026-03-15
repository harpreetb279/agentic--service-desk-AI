from pathlib import Path
from dotenv import load_dotenv
import os

base_dir = Path(__file__).resolve().parent
if (base_dir / '.env').exists():
    load_dotenv(base_dir / '.env', override=True)

keys = [
    'MODEL_PROVIDER',
    'GEMINI_API_KEY',
    'OPENAI_API_KEY',
    'ENABLE_LANGGRAPH',
    'ENABLE_CREWAI',
    'ENABLE_LANGFUSE',
    'LANGFUSE_HOST',
]

for key in keys:
    value = os.getenv(key, '')
    if value:
        print(f'{key}=set')
    else:
        print(f'{key}=missing')
