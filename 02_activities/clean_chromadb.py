import chromadb
from dotenv import load_dotenv

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import re
import csv

load_dotenv('../05_src/.env')
load_dotenv('../05_src/.secrets')

# Connect to the database
dbclient = chromadb.PersistentClient()

# List all collections to verify what exists
print("Current collections:")
collections = dbclient.list_collections()
for col in collections:
    print(f"  - {col.name}: {col.count()} documents")
    

import sqlite3
import os

# Path to your SQLite file
db_path = "./chroma/chroma.sqlite3"

if os.path.exists(db_path):
    print(f"Original size: {os.path.getsize(db_path) / (1024 * 1024):.1f} MB")
    
    # Connect and vacuum to reclaim space
    print("Running VACUUM to reclaim space (this may take a minute)...")
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()
    
    print(f"New size: {os.path.getsize(db_path) / (1024 * 1024):.1f} MB")
    print("✓ Done!")
else:
    print(f"Database file not found at: {db_path}")
    print("Please check your ChromaDB path.")