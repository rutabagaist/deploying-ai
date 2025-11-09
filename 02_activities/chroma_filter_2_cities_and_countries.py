import chromadb
from dotenv import load_dotenv

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
import re
import csv

load_dotenv('../05_src/.env')
load_dotenv('../05_src/.secrets')

def load_names_from_csv(csv_path, min_length=4):
    """Load place names from CSV file, filtering out very short names."""
    names = []
    common_words = {
        'reading', 'bath', 'orange', 'hope', 'man', 'china', 'turkey', 
        'mobile', 'phoenix', 'aurora', 'columbia', 'dover', 'independence',
        'franklin', 'jackson', 'lincoln', 'madison', 'monroe', 'washington',
        'clinton', 'cleveland', 'harrison', 'jefferson', 'hamilton',
        'spring', 'summer', 'winter', 'grove', 'lake', 'river', 'mount',
        'north', 'south', 'east', 'west', 'new', 'saint', 'san', 'santa',
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'jordan', 'chad', 'nice', 'mali', 'togo', 'peru'  # Country names that are also common words
    }
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)  # Skip header if present
        
        for row in csv_reader:
            if row:
                name = row[0].strip()
                # Filter criteria:
                # 1. Not empty
                # 2. Minimum length
                # 3. Not a common English word
                # 4. Contains at least one letter
                if (name and 
                    len(name) >= min_length and 
                    name.lower() not in common_words and
                    any(c.isalpha() for c in name)):
                    names.append(name)
    
    return names

def create_pattern(place_names):
    """Create a regex pattern from list of place names."""
    # Escape special regex characters
    escaped_names = [re.escape(name) for name in place_names]
    # Create pattern with word boundaries
    pattern = r'\b(' + '|'.join(escaped_names) + r')\b'
    return pattern

def contains_place(text, pattern):
    """Check if text contains any place name from the pattern."""
    if not text:
        return False
    return re.search(pattern, text, re.IGNORECASE) is not None

# Load cities and countries
print("Loading place names...")
cities = load_names_from_csv('./documents/cities_with_pop.csv', min_length=5)
countries = load_names_from_csv('./documents/countries_of_world.csv', min_length=4)  # Countries can be shorter (e.g., Peru, Cuba, Iran)

print(f"Loaded {len(cities)} cities")
print(f"Loaded {len(countries)} countries")
print(f"Total place names: {len(cities) + len(countries)}")

# Combine and deduplicate
all_places = list(set(cities + countries))
print(f"Unique place names after deduplication: {len(all_places)}")
print(f"Sample place names: {all_places[:20]}")

# Create regex pattern
place_pattern = create_pattern(all_places)

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

# Delete and recreate filtered collection
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

print("\nProcessing documents...")
for offset in range(0, total_docs, batch_size):
    # Get batch of documents
    results = original_collection.get(
        limit=batch_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"]
    )
    
    # Filter documents that contain place names
    for i, doc in enumerate(results['documents']):
        if contains_place(doc, place_pattern):
            filtered_collection.add(
                documents=[doc],
                metadatas=[results['metadatas'][i]],
                embeddings=[results['embeddings'][i]],
                ids=[results['ids'][i]]
            )
            filtered_count += 1
    
    print(f"Processed {min(offset + batch_size, total_docs)}/{total_docs} documents. Kept: {filtered_count}")

print(f"\n{'='*50}")
print(f"Original collection size: {total_docs}")
print(f"Filtered collection size: {filtered_collection.count()}")
print(f"Removed: {total_docs - filtered_count} documents")
print(f"Percentage kept: {(filtered_count/total_docs)*100:.1f}%")
print(f"{'='*50}")

# Show some examples of what was kept
print("\n=== SAMPLE BOOKS KEPT (first 5) ===")
kept = filtered_collection.get(limit=5, include=["documents", "metadatas"])
for i in range(len(kept['ids'])):
    print(f"\n{i+1}. Title: {kept['metadatas'][i].get('title', 'N/A')}")
    print(f"   Author: {kept['metadatas'][i].get('author', 'N/A')}")
    print(f"   Summary: {kept['documents'][i][:150]}...")