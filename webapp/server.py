import os

import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import FunctionTool, QueryEngineTool, RetrieverTool
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pydantic import BaseModel

# load our environment variables
# here we are using the .env file in the root directory
# and loading them using dotenv. You can use pydantic Settings as well
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# this is a constant series id for the series we are
# going to fetch the info for.
SERIES_ID = "e079ef23-b5e9-4802-93e9-dd2f27db0533"
CRICKET_API_KEY = os.getenv("CRICKET_API_KEY")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# load and prepare the data for the T20 world cup
batting_df = pd.read_csv("../data/batting_stats_for_icc_mens_t20_world_cup_2024.csv")
bowling_df = pd.read_csv("../data/bowling_stats_for_icc_mens_t20_world_cup_2024.csv")


# Initialise the gemini embedding model to be used for search
gemini_embedding_model = GeminiEmbedding(
    api_key=GEMINI_API_KEY, model_name="models/embedding-001"
)

# Initialise Gemini Flash LLM to be used for generating answers
gemini_llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY)

# Mark the LLM and embedding model to be used as default
Settings.embed_model = gemini_embedding_model
Settings.llm = gemini_llm

# Read the player data from the directory
reader = SimpleDirectoryReader(input_dir="../data/player_wiki/")
player_documents = reader.load_data()

# Create a in-memory vector store index from the player documents
player_index = VectorStoreIndex.from_documents(
    documents=player_documents, show_progress=True
)
player_retriever = player_index.as_retriever()

# Read the team data from the directory
reader = SimpleDirectoryReader(input_dir="../data/team_wiki/")
team_documents = reader.load_data()

# Create a in-memory vector store index from the team documents
team_index = VectorStoreIndex.from_documents(
    documents=team_documents, show_progress=True
)
team_retriever = team_index.as_retriever()

player_query_engine_tool = RetrieverTool.from_defaults(
    retriever=player_retriever,
    name="player_biography",
    description="Get info about a specific player's biography",
)

team_query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=team_retriever,
    name="team_information",
    description="Get info about a specific cricket team",
)


def get_matches_list():
    """
    Return the list of Matches and their information
    for the 2024 T20 world cup
    """
    url = f"https://api.cricapi.com/v1/series_info?apikey={CRICKET_API_KEY}&id={SERIES_ID}"

    response = requests.get(url)
    return response.json()


def get_match_scorecard(match_id: str):
    """
    Returns score card for a specific match for the 2024 T20 world cup
    Args
     - match_id - UUID of the match which can be obtained using get_matches_list
    """
    url = url = (
        f"https://api.cricapi.com/v1/match_scorecard?apikey={CRICKET_API_KEY}&id={match_id}"
    )

    response = requests.get(url)
    return response.json()


def get_series_squad():
    """
    Returns team squads for the 2024 T20 world cup
    """
    url = url = (
        f"https://api.cricapi.com/v1/series_squad?apikey={CRICKET_API_KEY}&id={SERIES_ID}"
    )

    response = requests.get(url)
    return response.json()


def get_t20_world_cup_stat_insights(stats_type: str, insight_type: str):
    """
    This will return the requested insights for the the given stats_type
    and insight_type

    Supported stats_type:
    1. batting
    2. bowling

    Supported insight_types
    1. For batting
      a. highest_run_scorer
      b. player_with_highest_strike_rate
    2. For bowling
      a. highest_wicket_taker
      b. player_with_least_economy
    """
    if stats_type == "batting":
        if insight_type == "highest_run_scorer":
            row_with_highest_score = batting_df.loc[batting_df["Runs"].idxmax()]

            return row_with_highest_score.to_dict()
        elif insight_type == "player_with_highest_strike_rate":
            row_with_highest_strike_rate = batting_df.loc[batting_df["SR"].idxmax()]

            return row_with_highest_strike_rate.to_dict()
        else:
            return "Invalid insight_type for batting"
    elif stats_type == "bowling":
        if insight_type == "highest_wicket_taker":
            row_with_highest_wickets = bowling_df.loc[bowling_df["Wkts"].idxmax()]

            return row_with_highest_wickets.to_dict()
        elif insight_type == "player_with_least_economy":
            row_with_least_economy = bowling_df.loc[bowling_df["Econ"].idxmin()]

            return row_with_least_economy.to_dict()
        else:
            return "Invalid insight_type for bowling"
    else:
        return "Invalid stats_type"


get_t20_world_cup_stat_insights_tool = FunctionTool.from_defaults(
    fn=get_t20_world_cup_stat_insights
)
get_match_scorecard_tool = FunctionTool.from_defaults(fn=get_match_scorecard)
get_matches_list_tool = FunctionTool.from_defaults(fn=get_matches_list)
get_series_squad_tool = FunctionTool.from_defaults(fn=get_series_squad)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

memory = ChatMemoryBuffer.from_defaults(token_limit=150000)


class GetAnswerAction:
    def __init__(self, query: str = None):
        self.llm = Gemini(api_key=GOOGLE_API_KEY, model="models/gemini-1.5-flash")
        self.agent = ReActAgent.from_tools(
            [
                player_query_engine_tool,
                team_query_engine_tool,
                get_matches_list_tool,
                get_series_squad_tool,
                get_match_scorecard_tool,
                get_t20_world_cup_stat_insights_tool,
            ],
            memory=memory,
            llm=self.llm,
            verbose=True,
            max_iterations=50,
        )
        self.query = query

    def execute(self):
        response = self.agent.chat(self.query)
        return response.response


class Query(BaseModel):
    query: str


@app.post("/api/v1/answer")
def get_answer(query: Query):
    query = query.query
    answer = GetAnswerAction(query=query).execute()
    return {"answer": answer}


@app.get("/api/v1/status")
def get_status():
    return {"status": "ok"}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
