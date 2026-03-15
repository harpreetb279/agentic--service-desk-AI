from pathlib import Path
from models.agent_pipeline import SupportAgent

base_dir = Path(__file__).resolve().parent.parent
agent = SupportAgent(base_dir=base_dir)
for question in agent.get_faq_questions()[:10]:
    print(question)
