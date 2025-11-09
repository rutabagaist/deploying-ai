# Here we define tools for our model agent to use when applicable.
import requests
import json
import os
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import tool
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

class Coordinates(BaseModel):
    latitude: float
    longitude: float

# This tool calls open-meteo to get coordinates for a city, because the weather API needs lat/lon to work.
@tool
def get_coords(city: str):
    """Get coordinates for a given city"""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    response = requests.get(url)
    resp_dict = json.loads(response.text)
    results = resp_dict.get("results", [])
    if results:
        return Coordinates(
            latitude = results[0].get("latitude"),
            longitude = results[0].get("longitude")
        )
    else:
        return f"No coordinates found for this {city}"



#This tool actually gets the current weather based on the coordinates obtained by get_coords.
@tool
def get_weather(latitude, longitude):
    """
    Returns the temperature, wind speed and description of current conditions for a given city.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m,visibility"
    response = requests.get(url)
    resp_dict = json.loads(response.text)
    # Format the weather data based on the API response structure
    # The weather data is nested inside the "current" object
    current = resp_dict.get("current", {})
    current_units = resp_dict.get("current_units", {})
    
    temperature = current.get("temperature_2m", "N/A")
    temp_unit = current_units.get("temperature_2m", "")
    
    wind = current.get("wind_speed_10m", "N/A")
    wind_unit = current_units.get("wind_speed_10m", "")
    
    visibility = current.get("visibility", "N/A")
    visibility_unit = current_units.get("visibility", "")
    
    weather = f"Temperature: {temperature}{temp_unit}\nWind Speed: {wind} {wind_unit}\nVisibility: {visibility} {visibility_unit}"
    return weather



# This tool allows our agent to search the web for the best photography spots in a city.
@tool
def search_web(query: str):
    """Searches the web using DuckDuckGo and returns the top 5 results."""
    ddg_settings = DuckDuckGoSearchAPIWrapper(max_results=5)
    results = ddg_settings.results(query, max_results=5)
    return str(results)  # Convert list to string for the LLM


# This tool searces a list of 6000 book descriptions in our ChromaDB file store for a book recommendation that has something to do with the city we're visiting. See readme.MD for details on this DB lookup.
# Initialize the chroma db client. 
@tool
def get_book_reco(city: str):
    """This gets a book recommendation from our database of book descriptions. Use this tool to recommend a book that relates to the visited location."""
    dbclient = chromadb.PersistentClient()

    # Connect to the chromadb containing embeddings and metadata
    collection = dbclient.get_or_create_collection(
        name = "bookSummaries_filtered",
        embedding_function = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
    )
    # Run a query for the city in question
    search_results = collection.query(
        query_texts = [city], 
        n_results = 1
    )
# Return the results to the llm
    return(search_results)