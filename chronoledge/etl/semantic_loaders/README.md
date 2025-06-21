# Semantic Phylogeny Tree Engine

## Overview

The Semantic Phylogeny Tree Engine is the core component of ChronoLedge that transforms chronologically ordered events into evolving semantic trees. It processes events through **Anchored Embedding Zones (AEZs)** that dynamically split and evolve based on semantic divergence, creating phylogenetic trees that track how event topics branch and evolve over time.

## Core Concepts

### 🧠 **Anchored Embedding Zones (AEZs)**

**Definition**: AEZs are semantic clusters that group similar events around a central "anchor" embedding. They represent coherent semantic regions in the embedding space.

**Key Properties**:
- **Anchor Embedding**: Central point representing the semantic center of the cluster
- **Event Members**: Collection of events that fall within the cluster's semantic boundary
- **Dynamic Size**: Can grow as similar events are added
- **Split Capability**: Divide into adapted sub-clusters when semantically diverse

**Formation Process**:
```
New Event → Check Existing AEZs → Similar AEZ Found? → Add to AEZ → Check Semantic Drift
                                   ↓ No
                                → Create New AEZ
```

### 🌡️ **Temperature System (Divergence Tracking)**

The temperature system measures semantic coherence within AEZs and predicts when they're ready to split.

#### **Temperature Classifications**:
- **🔥 HOT** (≥50% divergence): Ready to split - high internal semantic diversity
- **🟡 WARM** (35-49% divergence): Approaching split threshold
- **🟠 COOL** (20-34% divergence): Moderate semantic diversity
- **❄️ COLD** (<20% divergence): Highly coherent, stable cluster

#### **Divergence Score Calculation**:
```python
def _calculate_aez_divergence_score(self, aez_id: str) -> Optional[float]:
    """Calculate current divergence score for an AEZ using k-means clustering"""
    embeddings = [event.embedding for event in aez_events]

    # Run K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    
    # Compute similarity between centroids (high similarity = low divergence)
    inter_centroid_similarity = cosine_similarity(
        centroids[0].reshape(1, -1),
        centroids[1].reshape(1, -1)
    )[0][0]
    
    # Convert similarity to divergence (divergence = 1 - similarity)
    divergence_score = 1.0 - inter_centroid_similarity
    
    return divergence_score
```

### 🌿 **Lineage Nodes (Evolutionary Branching)**

**Definition**: Lineage nodes represent evolutionary branch points where AEZs have split, creating new semantic lineages.

**Creation Triggers**:
- AEZ divergence score exceeds split threshold (default 50%)
- Automatic k-means clustering with k=2 applied to split the AEZ
- Original AEZ becomes inactive (ancestor or "fossil"), with two new child AEZs
- Lineage node created to represent the evolutionary split

**Hierarchical Structure**:
```
Domain Root
    ├── Lineage-1 (from AEZ-A split)
    │   ├── AEZ-A1 (specialized cluster)
    │   └── AEZ-A2 (specialized cluster)
    └── Lineage-2 (from AEZ-B split)
        ├── AEZ-B1
        └── Lineage-3 (further evolution)
            ├── AEZ-B1-1
            └── AEZ-B1-2
```

## Technical Architecture

### 🔄 **Chronological Processing Pipeline**

The engine processes events in strict chronological order to maintain temporal evolution patterns:

1. **Date-Based Sorting**: Event files processed from oldest to newest
2. **Sequential Event Processing**: Events within each day processed in order
3. **Cumulative Evolution**: Each event builds upon the existing semantic landscape

### **Evolution Logic**

#### Step 1: Event Ingestion
1. Extract domain from event's `full_category` (first part before " > ")
2. Compute event embedding using SentenceTransformer model
3. Find best matching AEZ in the domain:
   - If similarity > threshold: assign to existing AEZ
   - Else: create new AEZ
4. Update AEZ anchor embedding and event pointers

#### Step 2: Divergence Detection
1. For AEZs with ≥ min_events_for_divergence:
   - Run K-means clustering (k=2) on member embeddings
   - Compute inter-centroid cosine similarity
   - If similarity < divergence_threshold: trigger split

#### Step 3: Lineage Forking
1. Convert splitting AEZ into a lineage node
2. Create two new child AEZs from the clusters
3. Update all pointer relationships
4. Deactivate original AEZ (become an ancestor)

### 🎯 **Similarity and Divergence Thresholds**

**Similarity Threshold** (default: 0.5):
- Determines when an event is similar enough to join an existing AEZ
- Higher values = stricter clustering (more AEZs)
- Lower values = looser clustering (fewer, larger AEZs)

**Divergence Threshold** (default: 0.5 - complementar to similarity):
- Determines when an AEZ is semantically diverse enough to split
- Controls the "temperature" at which AEZs branch into lineages
- Higher values = later splits (larger clusters before branching)

## Domain-Based Separation

### 🌍 **Semantic Domains**

Events are automatically clustered into domains based on their root categories:

**Example Domains**:
- **Politics**: Elections, government policy, international relations
- **Sports**: Games, tournaments, player transfers
- **Science**: Research discoveries, technological advances

**Domain Isolation**:
- Each domain evolves independently
- No cross-domain AEZ inheritance
- Prevents semantic contamination between unrelated topics
- Maintains clean phylogenetic boundaries

## AEZ Lifecycle Management

### 📈 **AEZ Evolution Cycle**

1. **Speciation**: Created when first event doesn't match existing AEZs
2. **Expansion**: Accumulates semantically similar events
3. **Stabilization**: Reaches stable size with coherent semantic focus
4. **Semantic Drift Phase**: Internal diversity increases as new event types join
5. **Semantic Speciation**: Divergence threshold exceeded, AEZ becomes inactive
6. **Ancestral**: Replaced by lineage node with two child AEZs

## Embedding and Similarity Computation

### 🔢 **Sentence Transformer Integration**

**Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384-dimensional embeddings
- **Speed**: Optimized for real-time processing
- **Quality**: Good balance of semantic understanding and performance

**Embedding Generation**:
```python
def generate_embedding(text):
    """Generate semantic embedding for event text"""
    
    # Clean and preprocess text
    cleaned_text = clean_event_text(text)
    
    # Generate embedding using sentence transformer
    embedding = sentence_transformer.encode(cleaned_text)
    
    # Normalize for cosine similarity
    return embedding / np.linalg.norm(embedding)
```

### 📏 **Similarity Metrics**

**Cosine Similarity**: Primary metric for event-to-AEZ matching
```python
def cosine_similarity(embedding_1, embedding_2):
    return np.dot(embedding_1, embedding_2)
```

**Advantages**:
- Direction-focused (semantic meaning) rather than magnitude
- Normalized range [-1, 1] for consistent thresholding
- Robust to text length variations

### 📊 **Tree Metrics**

**Depth**: Maximum levels from domain root to leaf AEZs
**Branching Factor**: Average number of children per lineage node
**Active AEZs**: Currently growing semantic clusters
**Total Lineages**: Number of evolutionary split points
**Evolution Rate**: Frequency of AEZ splits over time

## Performance Optimization

### ⚡ **Computational Efficiency**

**Embedding Caching**: 
- Pre-computed embeddings stored with events
- Avoids re-computation during tree building

**Similarity Search Optimization**:
- Only compare new events against active AEZs in same domain
- Early termination when similarity threshold met

**Memory Management**:
- Inactive AEZ embeddings can be archived
- Streaming processing for large event datasets

### 📈 **Scalability Considerations**

**Domain Parallelization**: 
- Independent domain processing allows parallel execution
- Memory isolation prevents interference

**Incremental Processing**:
- New events can be processed without rebuilding entire tree
- Supports real-time event stream integration