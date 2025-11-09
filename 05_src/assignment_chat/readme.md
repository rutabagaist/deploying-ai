# The Travel Chatbot

This travel chatbot is an attempt to satisfy the requirements of assignment 2 as part of the 'Deploying AI' microcredential. 

## Chatbot setup

The chatbot uses the following tools:
1. (API)an open-meteo API that converts city names to latitude and longitude
2. (API) another open-meteo endpoint that takes the lat/lon and returns temperature, wind speed and current visibility
3. (WebSearch) duckduckgo web search tool
4. (ChromaDB Semantic Search) a semantic query that relies on a ChromaDB database I created from the CMU Book Summary Dataset (https://www.cs.cmu.edu/~dbamman/booksummaries.html)
    - to create the embeddings I used code that I've placed into embeddings_create.py

## Creating embeddings

The original dataset of book summaries was about 42Mb
Some of the 16000+ book summaries were too long for the 8192 token limit imposed by the OpenAI embeddings model

### Step 1: Truncate the descriptions

I used `tiktoken` to make a function to truncate the inputs to 8192 tokens. I figured  the gist of the book had likely been captured by then. Not many descriptions exceeded this threshold.

### Step 2: Create a ChromaDB `.PersistentClient()` and define the embeddings function

I used the OpenAIEmbeddingFunction from chromdb's utils to generate embeddings. It took 82 minutes to generate them all.

### Step 3: Attempts to make the db smaller

The resulting DB was huge, around 375Mb. I culled it by discarding descriptions that didn't match to a list of world cities and place names. It's crude, but it made it smaller. Unfortunately, it was still around 100Mb. I didn't want to waste time trying to tame this dataset, so in your version of it will be missing unfortunately! It works well however and the recommender is very often able to find a book in the dataset that is at least peripherally related to the travel plans described.

### Step 4: Read the book summaries into the Chromadb collection

I then added all the books into the ChromaDB collection and at the same time created embeddings from the book summaries.

## Creating Instructions/System Prompt

The instructions work fairly well. I find it difficult to sense the 'tone' I've requested unless it's something really discernable. I've tested the guardrails and they were respected in my attempts to jailbreak the chatbot.

The combining of tool information with narrative is also pretty functional, as is the fail-safe in case one or more tools fails. 