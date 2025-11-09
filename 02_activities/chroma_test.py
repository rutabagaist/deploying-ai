from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import re
import csv


load_dotenv('../05_src/.env')
load_dotenv('../05_src/.secrets')


def load_cities_from_csv(csv_path):
    """Load city names from CSV file."""
    cities = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if row:  # Skip empty rows
                # Assuming city name is in the first column
                city = row[0].strip()
                if city:
                    cities.append(city)
    return cities

def create_city_pattern(cities):
    """Create a regex pattern from list of cities."""
    # Escape special regex characters in city names
    escaped_cities = [re.escape(city) for city in cities]
    # Create pattern with word boundaries
    pattern = r'\b(' + '|'.join(escaped_cities) + r')\b'
    return pattern

def contains_city(text, pattern):
    """Check if text contains any city from the pattern."""
    if not text:
        return False
    return re.search(pattern, text, re.IGNORECASE) is not None

# Load cities from CSV
print("Loading cities from CSV...")
cities = load_cities_from_csv('./documents/cities_list.csv')
print(f"Loaded {len(cities)} cities")
print(f"Sample cities: {cities[:10]}")  # Show first 10 for verification

# Create regex pattern
city_pattern = create_city_pattern(cities)

# Connect to existing database
dbclient = chromadb.PersistentClient()

# Get the original collection
original_collection = dbclient.get_collection(
    name="bookSummaries",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
)

print(f"\nOriginal collection size: {original_collection.count()}")

# Create new filtered collection
try:
    dbclient.delete_collection(name="bookSummaries_filtered")
    print("Deleted existing filtered collection")
except:
    pass

filtered_collection = dbclient.create_collection(
    name="bookSummaries_filtered",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
)

# Get all documents from original collection (in batches)
batch_size = 1000
total_docs = original_collection.count()
filtered_count = 0

for offset in range(0, total_docs, batch_size):
    # Get batch of documents
    results = original_collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Filter documents that contain city names
    for i, doc in enumerate(results['documents']):
        if contains_city(doc, city_pattern):
            filtered_collection.add(
                documents=[doc],
                metadatas=[results['metadatas'][i]],
                embeddings=[results['embeddings'][i]],  # Reuse existing embeddings!
                ids=[results['ids'][i]]
            )
            filtered_count += 1
    
    print(f"Processed {min(offset + batch_size, total_docs)}/{total_docs} documents. Filtered: {filtered_count}")

print(f"\n{'='*50}")
print(f"Filtered collection size: {filtered_collection.count()}")
print(f"Reduction: {total_docs - filtered_count} documents removed")
print(f"Percentage kept: {(filtered_count/total_docs)*100:.1f}%")
print(f"{'='*50}")