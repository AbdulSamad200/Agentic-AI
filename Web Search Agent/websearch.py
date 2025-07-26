from phi.agent import Agent
import google.generativeai as genai
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

class GeminiChat:
    def __init__(self, model_name="gemini-pro"):
        self.model = genai.GenerativeModel(model_name)

    def response_stream(self, messages):
        # Convert messages to a single input string
        input_text = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        
        # Generate response using Gemini
        response = self.model.generate_content(input_text)
        
        # Yield the response in a streaming-like format
        yield {"content": response.text}

# Use the custom GeminiChat model
web_agent = Agent(
    name="Web Agent",
    model=GeminiChat(),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Generate response
web_agent.print_response("Tell me about OpenAI Sora?", stream=True)
