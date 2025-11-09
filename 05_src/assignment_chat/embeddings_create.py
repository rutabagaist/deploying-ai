
import os
import csv
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import tiktoken


# Initialize the tokenizer for the embedding model
encoding = tiktoken.get_encoding("cl100k_base")  # This is used by text-embedding-3-small

# Create a function to truncate text becase the model has a max. token # of 8192
def truncate_text(text, max_tokens=8192):
    """Truncate text to a maximum number of tokens."""
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text


# Initialize the chroma db client.
dbclient = chromadb.PersistentClient()

# Create the collection and run embeddings at the same time
collection = dbclient.get_or_create_collection(
   name = "bookSummaries",
   embedding_function = OpenAIEmbeddingFunction(
       api_key=os.getenv("OPENAI_API_KEY"),
       model_name="text-embedding-3-small"
   )
)

with open ('./documents/booksummaries.txt', newline ='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = '\t')
    for idx, row in enumerate(csv_reader):
        document_text = truncate_text(row[6], max_tokens = 8192) # Truncating the descriptions
        metadata = {
            "title": row[3] if len(row) > 2 else "",
            "author": row[4] if len(row) > 3 else "",
            "year": row[5] if len(row) > 2 else "",
        }
        
        collection.add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[f"book_{idx}"]
            )
        
        if idx % 100 == 0:
            print(f"Processed {idx} books..")

print(f"Total books added: {collection.count()}")