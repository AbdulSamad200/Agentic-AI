import typer
from typing import Optional, List

from phi.agent import Agent         # ✅ Use Agent instead of Assistant
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage  # ✅ Correct storage import
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# ✅ Set up PDF knowledge base with PgVector2 (collection name used)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url),
)
knowledge_base.load()

# ✅ Use PgAgentStorage for persistent sessions
storage = PgAgentStorage(table_name="pdf_agent_sessions", db_url=db_url)

def pdf_agent(new: bool = False, user: str = "user"):
    session_id: Optional[str] = None
    if not new:
        existing = storage.get_all_session_ids(user_id=user)
        if existing:
            session_id = existing[0]

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="PDF Agent",
        user_id=user,
        session_id=session_id,
        knowledge=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        markdown=True,
    )

    if session_id is None:
        print(f"Started Session: {agent.session_id}\n")
    else:
        print(f"Continuing Session: {agent.session_id}\n")

    agent.cli_app(markdown=True, stream=True)

if __name__ == "__main__":
    typer.run(pdf_agent)
