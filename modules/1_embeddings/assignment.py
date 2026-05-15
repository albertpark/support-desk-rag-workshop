# -*- coding: utf-8 -*-

# ALBERT PARK - Hour 1 Exercises: Embeddings & Similarity Search

import json
import numpy as np              # For numerical operations on embedding vectors
import random
import time
import os
from openai import OpenAI       # OpenAI API client for generating embeddings
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors
import matplotlib.pyplot as plt # For visualizing embeddings
from dotenv import load_dotenv  # Load environment variables from .env file

load_dotenv()
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'text-embedding-3-small'

# EMBEDDING MODEL SELECTION
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', model)
embedding_dim = 1536  # Number of dimensions in the embedding vector
print(f"Using OpenAI model: {embedding_model}")
print(f"Embedding dimension: {embedding_dim}")

# LOAD DATA: Support Tickets
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

# Display sample ticket - understand what we're working with
print("\n" + "="*80)
print("SAMPLE TICKET:")
print("="*80)
sample = tickets[0]
print(f"ID: {sample['ticket_id']}")
print(f"Title: {sample['title']}")
print(f"Description: {sample['description'][:200]}...")
print(f"Category: {sample['category']}")
print(f"Priority: {sample['priority']}")

# PART 1: Generate Embeddings
print("\n" + "="*80)
print("PART 1: Generating Embeddings")
print("="*80)

ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

# Generate embeddings with Batch call via OpenAI API
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)

# Convert API response to NumPy array for mathematical operations
# Shape: (num_tickets, 1536) - each row is one ticket's embedding
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Generated embeddings with shape: {embeddings.shape}")
print(f"  ({len(tickets)} tickets × {embedding_dim} dimensions)")

# Inspect an embedding
print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])
print("  (These 1536 numbers encode the semantic meaning of the text)")

# PART 2: Compute Similarity Scores
print("\n" + "="*80)
print("PART 2: Computing Similarity Scores")
print("="*80)

# ============================================================================
# Exercise 1: Change the Search Query (Easy)
# ============================================================================
queries = [
    "Users can't login after changing password",
    "Database is running very slowly",
    "Payment failed for international customer",
    "Mobile app keeps crashing",
    "Emails are not being delivered"
]
query = random.choice(queries)
print(f"\nSearch Query: '{query}'")

query_response = client.embeddings.create(input=[query], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")  # (1, 1536)

similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# PART 3: Retrieve Most Similar Tickets
print("\n" + "="*80)
print("PART 3: Finding Most Similar Tickets")
print("="*80)

# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# ============================================================================
top_k = 10

# np.argsort() returns indices that would sort the array (ascending)
# [::-1] reverses to get descending order (highest similarity first)
# [:top_k] takes only the top K results
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nTop {top_k} most similar tickets to query: '{query}'")
print("-" * 80)

# ============================================================================
# Exercise 3: Add a Similarity Threshold (Easy)
# ============================================================================
for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]

    if score < 0.5:
        continue
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")
    print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
    print(f"Description: {ticket['description'][:150]}...")

# PART 4: Visualize Embedding Relationships
print("\n" + "="*80)
print("PART 4: Visualizing Similarity Relationships")
print("="*80)

print("\nEmbeddings capture semantic relationships through similarity scores.")
print("Let's visualize these relationships using exact similarity measurements.\n")

# Create similarity heatmap for top tickets
print("Creating similarity heatmap...")

# Select top matches and a few random others for comparison
# This lets us see: high-similarity pairs vs. unrelated pairs
selected_indices = list(top_indices[:5]) + list(np.random.choice(
    [i for i in range(len(tickets)) if i not in top_indices[:5]], 
    size=min(5, len(tickets) - 5), 
    replace=False
))

# Compute similarity matrix for selected tickets
# This creates a 10x10 matrix of pairwise similarities
selected_embeddings = embeddings[selected_indices]
similarity_matrix = cosine_similarity(selected_embeddings)

# Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT PLOT: Similarity heatmap
# Shows pairwise similarity between all selected tickets
# Green = high similarity, Red = low similarity
im = ax1.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(len(selected_indices)))
ax1.set_yticks(range(len(selected_indices)))

# Label with ticket IDs and categories
labels = [f"{tickets[i]['ticket_id']}\n({tickets[i]['category']})" 
          for i in selected_indices]
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_yticklabels(labels, fontsize=8)

# Add similarity values to cells (so students can read exact numbers)
for i in range(len(selected_indices)):
    for j in range(len(selected_indices)):
        text = ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

ax1.set_title('Similarity Heatmap: What Embeddings Actually Measure\n' + 
             '(Top 5 matches + random others)', fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax1, label='Cosine Similarity')

# RIGHT PLOT: Query similarities bar chart
# Shows how similar each ticket is to the original query
query_similarities = [similarities[i] for i in selected_indices]
colors_bar = ['green' if i < 5 else 'gray' for i in range(len(selected_indices))]

ax2.barh(range(len(selected_indices)), query_similarities, color=colors_bar, alpha=0.7)
ax2.set_yticks(range(len(selected_indices)))
ax2.set_yticklabels([f"{tickets[i]['ticket_id']}" for i in selected_indices], fontsize=9)
ax2.set_xlabel('Similarity to Query', fontweight='bold')
ax2.set_title(f'Similarity Scores for Query:\n"{query}"\n(Green = Top 5 matches)', 
             fontweight='bold', fontsize=11)
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

# Add score labels on bars
for i, score in enumerate(query_similarities):
    ax2.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('embeddings_similarity_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'embeddings_similarity_analysis.png'")
print("\nKEY INSIGHTS FROM THIS VISUALIZATION:")
print("  • Left heatmap: Shows TRUE pairwise similarities in 1536D space")
print("  • Right chart: Query similarity scores (what drives retrieval)")
print("  • High similarity (green) = semantically similar content")
print("  • Low similarity (red) = different topics/meanings")
print("  • These scores are EXACT - they show true relationships in 1536D space!")
plt.show(block=False)

# PART 5: Experiment with Different Queries
print("\n" + "="*80)
print("PART 5: Try Different Queries")
print("="*80)

test_queries = [
    "Database is timing out",
    "Payment not working for foreign customers",
    "App crashes on iPhone",
    "Emails are not being sent"
]

print("\nTesting semantic search with different queries:")
for test_query in test_queries:
    # Generate query embedding
    query_resp = client.embeddings.create(input=[test_query], model=embedding_model)
    query_emb = np.array([query_resp.data[0].embedding])
    
    # Compare to all tickets
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")

# ============================================================================
# Exercise 4: Compare Two Queries (Easy)
# ============================================================================
query1 = "Login authentication failed"
query2 = "Slow database performance"

print("\n" + "="*80)
print("COMPARING TWO QUERIES")
print("="*80)

for q in [query1, query2]:
    response = client.embeddings.create(input=[q], model=embedding_model)
    q_emb = np.array([response.data[0].embedding])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{q}'")
    print(f"  Best match: {tickets[top_idx]['title']}")
    print(f"  Score: {sims[top_idx]:.4f}")

# ============================================================================
# Exercise 5: Test Semantic Understanding (Medium)
# ============================================================================
# These mean the SAME thing but use DIFFERENT words
texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # DIFFERENT topic
]

# Generate embeddings
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

# Calculate all pairwise similarities
similarity_matrix = cosine_similarity(embeddings)

# Print results
print("Similarity Matrix:")
print("-" * 50)
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:  # Only print upper triangle
            sim = similarity_matrix[i][j]
            print(f"{sim:.3f}  '{text1[:30]}...' vs '{text2[:30]}...'")

# ============================================================================
# Exercise 6: Experiment with Different Queries
# ============================================================================
# Load data
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])

def search_with_category(query, category_filter=None, top_k=5):
    """Search tickets, optionally filtering by category"""
    # Get query embedding
    response = client.embeddings.create(input=[query], model=model)
    query_emb = np.array([response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get results with category filter
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]
        
        # FILL IN THIS LINE: Skip if category doesn't match filter
        # Hint: if category_filter is set AND ticket category doesn't match, skip
        if category_filter and ticket['category'] != category_filter:
            continue
        
        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    
    return results

# Test it
print("All categories:")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category:")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

# ============================================================================
# Exercise 7: Batch vs Single Embedding (Medium)
# ============================================================================
texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

# Method 1: SLOW - One API call per text
print("Method 1: Single API calls...")
start = time.time()
for text in texts:
    response = client.embeddings.create(input=[text], model=model)
time_slow = time.time() - start
print(f"  Time: {time_slow:.2f} seconds")

# Method 2: FAST - One API call for all texts
print("\nMethod 2: Batch API call...")
start = time.time()
response = client.embeddings.create(input=texts, model=model)
time_fast = time.time() - start
print(f"  Time: {time_fast:.2f} seconds")

# Compare
print(f"\n✓ Batch is {time_slow/time_fast:.1f}x faster!")
print(f"  Always batch your embeddings in production!")

# ============================================================================
# Bonus Exercise: Similarity Matrix Heatmap (Challenge)
# ============================================================================
# Load first 10 tickets
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)[:10]

# Generate embeddings
texts = [t['title'] for t in tickets]
response = client.embeddings.create(input=texts, model='text-embedding-3-small')
embeddings = np.array([data.embedding for data in response.data])

# Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

# Create heatmap
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')
plt.xticks(range(10), [t['ticket_id'] for t in tickets], rotation=45, ha='right')
plt.yticks(range(10), [t['ticket_id'] for t in tickets])
plt.title('Ticket Similarity Matrix')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')
plt.show()
print("✓ Saved as similarity_heatmap.png")
