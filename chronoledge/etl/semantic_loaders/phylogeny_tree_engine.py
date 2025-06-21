"""
Semantic Phylogeny Engine

This approach uses only root domains (like "sports", "armed_conflicts_and_attacks") to guide AEZ creation.
Lineages evolve naturally from AEZ splits when semantic divergence is detected.

Key principles:
1. Events are only compared within their root domain
2. AEZs split via k-means when they become semantically diverse
3. AEZ splits create new lineage nodes with two child AEZs
4. Full category paths are stored as metadata only
5. Semantic evolution is tracked naturally through splits
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
import numpy as np
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

@dataclass
class EventNode:
    """Represents a single event in the phylogeny tree"""
    event_id: str
    summary: str
    full_category: str
    domain: str
    date: str
    embedding: Optional[np.ndarray] = None
    aez_id: Optional[str] = None
    lineage_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AEZ:
    """Anchored Embedding Zone - semantic cluster of events"""
    aez_id: str
    domain: str
    event_ids: List[str] = field(default_factory=list)
    anchor_embedding: Optional[np.ndarray] = None
    parent_aez_id: Optional[str] = None
    parent_lineage_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True
    divergence_history: List[Dict[str, Any]] = field(default_factory=list)  # Track divergence evolution
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LineageNode:
    """Represents a node in the phylogenetic tree"""
    lineage_id: str
    domain: str
    node_type: str  # 'root', 'lineage', 'aez'
    name: Optional[str] = None
    parent_lineage_id: Optional[str] = None
    children_aez_ids: List[str] = field(default_factory=list)
    children_lineage_ids: List[str] = field(default_factory=list)  # Child lineages
    ancestor_aez_id: Optional[str] = None  # The AEZ that became this lineage
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticPhylogenyEngine:
    """Main engine for managing semantic evolution and phylogenetic trees"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.5,
                 divergence_threshold: float = 0.5,
                 min_events_for_divergence: int = 3,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the phylogeny engine
        
        Args:
            similarity_threshold: Cosine similarity threshold for event-AEZ matching
            divergence_threshold: Inter-centroid similarity threshold for divergence
            min_events_for_divergence: Minimum events needed for AEZ divergence check
            embedding_model: SentenceTransformer model name for embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.divergence_threshold = divergence_threshold
        self.min_events_for_divergence = min_events_for_divergence
        
        # Storage for all components
        self.events: Dict[str, EventNode] = {}
        self.aezs: Dict[str, AEZ] = {}
        self.lineages: Dict[str, LineageNode] = {}
        self.domain_roots: Dict[str, str] = {}  # domain -> root_lineage_id
        
        # Initialize SentenceTransformer model
        try:
            logger.info(f"Loading SentenceTransformer model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model '{embedding_model}': {e}")
            raise RuntimeError(f"Could not initialize embedding model: {e}")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get SentenceTransformer embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: '{text[:100]}...': {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def _extract_domain(self, full_category: str) -> str:
        """Extract domain (first part) from full category"""
        if ' > ' in full_category:
            return full_category.split(' > ')[0].strip()
        return full_category.strip()
    
    def _ensure_domain_root(self, domain: str) -> str:
        """Ensure domain has a root lineage node"""
        if domain not in self.domain_roots:
            root_id = self._generate_id("lineage")
            root_lineage = LineageNode(
                lineage_id=root_id,
                domain=domain,
                node_type="root",
                name=f"{domain} (Root)"
            )
            self.lineages[root_id] = root_lineage
            self.domain_roots[domain] = root_id
            logger.info(f"Created root lineage {root_id} for domain '{domain}'")
        
        return self.domain_roots[domain]
    
    def _compute_aez_similarity(self, event_embedding: np.ndarray, aez: AEZ) -> float:
        """Compute cosine similarity between event and AEZ anchor"""
        if aez.anchor_embedding is None:
            return 0.0
        
        similarity = cosine_similarity(
            event_embedding.reshape(1, -1),
            aez.anchor_embedding.reshape(1, -1)
        )[0][0]
        
        return similarity
    
    def _update_aez_anchor(self, aez_id: str):
        """Update AEZ anchor embedding as mean of all member events"""
        aez = self.aezs[aez_id]
        if not aez.event_ids:
            return
        
        embeddings = []
        for event_id in aez.event_ids:
            if event_id in self.events and self.events[event_id].embedding is not None:
                embeddings.append(self.events[event_id].embedding)
        
        if embeddings:
            aez.anchor_embedding = np.mean(embeddings, axis=0)
            logger.debug(f"Updated anchor for AEZ {aez_id} with {len(embeddings)} events")
    
    def _calculate_aez_divergence_score(self, aez_id: str) -> Optional[float]:
        """Calculate current divergence score for an AEZ using k-means clustering"""
        aez = self.aezs[aez_id]
        
        # Need at least 2 events to calculate divergence
        if len(aez.event_ids) < 2:
            return None
        
        # Get embeddings for all events in AEZ
        embeddings = []
        for event_id in aez.event_ids:
            if event_id in self.events and self.events[event_id].embedding is not None:
                embeddings.append(self.events[event_id].embedding)
        
        if len(embeddings) < 2:
            return None
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error calculating divergence for AEZ {aez_id}: {e}")
            return None
    
    def _track_aez_divergence(self, aez_id: str, new_event_id: str):
        """Track divergence evolution for an AEZ after adding a new event"""
        aez = self.aezs[aez_id]
        event_count = len(aez.event_ids)
        
        # Calculate current divergence score
        divergence_score = self._calculate_aez_divergence_score(aez_id)
        
        # Create divergence entry
        divergence_entry = {
            'event_count': event_count,
            'divergence_score': divergence_score,
            'timestamp': datetime.now().isoformat(),
            'triggering_event_id': new_event_id,
            'triggering_event_summary': self.events[new_event_id].summary[:100] if new_event_id in self.events else None
        }
        
        # Add to divergence history
        aez.divergence_history.append(divergence_entry)
        
        # Determine temperature based on divergence score
        if divergence_score is not None:
            if divergence_score >= self.divergence_threshold:
                temperature = "🔥 HOT (Ready to split)"
                logger.info(f"AEZ {aez_id} is HOT! Divergence: {divergence_score:.3f} >= {self.divergence_threshold}")
            elif divergence_score >= self.divergence_threshold * 0.7:
                temperature = "🟡 WARM (Approaching split)"
            elif divergence_score >= self.divergence_threshold * 0.4:
                temperature = "🟠 COOL (Moderate diversity)"
            else:
                temperature = "❄️ COLD (Highly coherent)"
            
            divergence_entry['temperature'] = temperature
            
            logger.debug(f"AEZ {aez_id} divergence tracking: {event_count} events, "
                        f"divergence={divergence_score:.3f}, {temperature}")
        else:
            divergence_entry['temperature'] = "⚪ UNKNOWN (Insufficient data)"
    
    def _create_new_aez(self, domain: str, parent_lineage_id: str, event_id: str) -> str:
        """Create a new AEZ for the domain"""
        aez_id = self._generate_id("aez")
        
        aez = AEZ(
            aez_id=aez_id,
            domain=domain,
            event_ids=[event_id],
            parent_lineage_id=parent_lineage_id
        )
        
        self.aezs[aez_id] = aez
        self._update_aez_anchor(aez_id)
        
        # Update lineage children
        if parent_lineage_id in self.lineages:
            self.lineages[parent_lineage_id].children_aez_ids.append(aez_id)
        
        logger.info(f"Created new AEZ {aez_id} in domain '{domain}' under lineage {parent_lineage_id}")
        return aez_id
    
    def _check_aez_divergence(self, aez_id: str) -> bool:
        """Check if AEZ should diverge based on internal clustering"""
        aez = self.aezs[aez_id]
        
        if len(aez.event_ids) < self.min_events_for_divergence:
            return False
        
        # Get embeddings for all events in AEZ
        embeddings = []
        valid_event_ids = []
        
        for event_id in aez.event_ids:
            if event_id in self.events and self.events[event_id].embedding is not None:
                embeddings.append(self.events[event_id].embedding)
                valid_event_ids.append(event_id)
        
        if len(embeddings) < self.min_events_for_divergence:
            return False
        
        # Run K-means clustering with k=2
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            centroids = kmeans.cluster_centers_
            # Compute similarity between centroids
            inter_centroid_similarity = cosine_similarity(
                centroids[0].reshape(1, -1),
                centroids[1].reshape(1, -1)
            )[0][0]

            logger.debug(f"AEZ {aez_id} inter-centroid similarity: {inter_centroid_similarity:.3f}")

            if inter_centroid_similarity < self.divergence_threshold:

                # Store clustering results for later use
                aez.metadata['divergence_clusters'] = {
                    'labels': cluster_labels.tolist(),
                    'valid_event_ids': valid_event_ids,
                    'centroids': centroids.tolist(),
                    'similarity': inter_centroid_similarity
                }
                return True
                
        except Exception as e:
            logger.error(f"Error in divergence check for AEZ {aez_id}: {e}")
        
        return False
    
    def _create_lineage_fork(self, aez_id: str) -> Tuple[str, str]:
        """Create lineage fork and split AEZ into two new AEZs"""
        aez = self.aezs[aez_id]
        
        # Create new lineage node
        lineage_id = self._generate_id("lineage")
        lineage_node = LineageNode(
            lineage_id=lineage_id,
            domain=aez.domain,
            node_type="lineage",
            parent_lineage_id=aez.parent_lineage_id,
            ancestor_aez_id=aez_id
        )
        
        # Get clustering data
        clustering_data = aez.metadata.get('divergence_clusters', {})
        labels = clustering_data.get('labels', [])
        valid_event_ids = clustering_data.get('valid_event_ids', [])
        centroids = clustering_data.get('centroids', [])
        
        # Create two new AEZs
        cluster_0_events = [valid_event_ids[i] for i, label in enumerate(labels) if label == 0]
        cluster_1_events = [valid_event_ids[i] for i, label in enumerate(labels) if label == 1]
        
        # Create first child AEZ
        aez_1_id = self._generate_id("aez")
        aez_1 = AEZ(
            aez_id=aez_1_id,
            domain=aez.domain,
            event_ids=cluster_0_events,
            parent_aez_id=aez_id,
            parent_lineage_id=lineage_id,
            anchor_embedding=np.array(centroids[0]) if centroids else None
        )
        
        # Create second child AEZ
        aez_2_id = self._generate_id("aez")
        aez_2 = AEZ(
            aez_id=aez_2_id,
            domain=aez.domain,
            event_ids=cluster_1_events,
            parent_aez_id=aez_id,
            parent_lineage_id=lineage_id,
            anchor_embedding=np.array(centroids[1]) if centroids else None
        )
        
        # Store new components
        self.lineages[lineage_id] = lineage_node
        self.aezs[aez_1_id] = aez_1
        self.aezs[aez_2_id] = aez_2
        
        # Update lineage children
        lineage_node.children_aez_ids = [aez_1_id, aez_2_id]
        
        # Update parent lineage to point to new lineage instead of old AEZ
        if aez.parent_lineage_id in self.lineages:
            parent_lineage = self.lineages[aez.parent_lineage_id]
            if aez_id in parent_lineage.children_aez_ids:
                parent_lineage.children_aez_ids.remove(aez_id)
                parent_lineage.children_lineage_ids.append(lineage_id)
        
        # Update event pointers
        for event_id in cluster_0_events:
            if event_id in self.events:
                self.events[event_id].aez_id = aez_1_id
                self.events[event_id].lineage_id = lineage_id
        
        for event_id in cluster_1_events:
            if event_id in self.events:
                self.events[event_id].aez_id = aez_2_id
                self.events[event_id].lineage_id = lineage_id
        
        # Deactivate original AEZ
        aez.is_active = False
        
        logger.info(f"Created lineage fork {lineage_id} from AEZ {aez_id}")
        logger.info(f"  - Child AEZ {aez_1_id}: {len(cluster_0_events)} events")
        logger.info(f"  - Child AEZ {aez_2_id}: {len(cluster_1_events)} events")
        
        return aez_1_id, aez_2_id
    
    def ingest_event(self, event_data: Dict[str, Any]) -> str:
        """
        Ingest a single event and process it through the phylogeny system
        
        Args:
            event_data: Event dictionary from JSON
            
        Returns:
            event_id of the processed event
        """
        # Create event node
        event_id = self._generate_id("event")
        domain = self._extract_domain(event_data['full_category'])
        
        event = EventNode(
            event_id=event_id,
            summary=event_data['summary'],
            full_category=event_data['full_category'],
            domain=domain,
            date=event_data['date'],
            metadata=event_data
        )
        
        # Get embedding
        event.embedding = self._get_embedding(event.summary)
        
        # Store event
        self.events[event_id] = event
        
        # Ensure domain root exists
        root_lineage_id = self._ensure_domain_root(domain)
        
        # Step 1: Event Ingestion Logic
        domain_aezs = [aez for aez in self.aezs.values() 
                      if aez.domain == domain and aez.is_active]
        
        if not domain_aezs:
            # Create first AEZ for domain
            aez_id = self._create_new_aez(domain, root_lineage_id, event_id)
            event.aez_id = aez_id
            event.lineage_id = root_lineage_id
        else:
            # Find best matching AEZ
            best_aez = None
            best_similarity = -1
            
            for aez in domain_aezs:
                similarity = self._compute_aez_similarity(event.embedding, aez)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_aez = aez
            
            if best_similarity > self.similarity_threshold:
                # Assign to existing AEZ
                best_aez.event_ids.append(event_id)
                self._update_aez_anchor(best_aez.aez_id)
                
                event.aez_id = best_aez.aez_id
                event.lineage_id = best_aez.parent_lineage_id
                
                # Track divergence evolution after adding the event
                self._track_aez_divergence(best_aez.aez_id, event_id)
                
                logger.debug(f"Assigned event {event_id} to AEZ {best_aez.aez_id} "
                           f"(similarity: {best_similarity:.3f})")
            else:
                # Create new AEZ
                aez_id = self._create_new_aez(domain, root_lineage_id, event_id)
                event.aez_id = aez_id
                event.lineage_id = root_lineage_id
                
                # Track initial divergence for new AEZ (will be None for single event)
                self._track_aez_divergence(aez_id, event_id)
                
                logger.debug(f"Created new AEZ {aez_id} for event {event_id} "
                           f"(best similarity: {best_similarity:.3f})")
        
        # Step 2: Check for divergence in assigned AEZ
        if event.aez_id and self._check_aez_divergence(event.aez_id):
            logger.info(f"Triggering divergence for AEZ {event.aez_id}")
            self._create_lineage_fork(event.aez_id)
        
        logger.info(f"Ingested event {event_id} in domain '{domain}'")
        return event_id
    
    def process_chronological_events(self, data_dir: str = "data/events"):
        """Process all events from data directory in chronological order"""
        events_dir = Path(data_dir)
        
        # Get all event files and sort chronologically
        event_files = sorted(events_dir.glob("*.json"))
        
        logger.info(f"Processing {len(event_files)} event files chronologically")
        
        for file_path in event_files:
            logger.info(f"Processing {file_path.name}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            events = data.get('events', [])
            for event_data in events:
                try:
                    self.ingest_event(event_data)
                except Exception as e:
                    logger.error(f"Error processing event in {file_path.name}: {e}")
                    continue
    
    def get_aez_temperature_summary(self, aez_id: str) -> Dict[str, Any]:
        """Get temperature and divergence summary for an AEZ"""
        if aez_id not in self.aezs:
            return {}
        
        aez = self.aezs[aez_id]
        history = aez.divergence_history
        
        if not history:
            return {
                'current_temperature': "⚪ UNKNOWN",
                'current_divergence': None,
                'trend': "No data",
                'history_length': 0
            }
        
        latest = history[-1]
        current_divergence = latest.get('divergence_score')
        current_temperature = latest.get('temperature', "⚪ UNKNOWN")
        
        # Calculate trend if we have multiple measurements
        trend = "Stable"
        if len(history) >= 3:
            recent_scores = [h.get('divergence_score') for h in history[-3:] if h.get('divergence_score') is not None]
            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[0] + 0.1:
                    trend = "📈 Heating up"
                elif recent_scores[-1] < recent_scores[0] - 0.1:
                    trend = "📉 Cooling down"
                else:
                    trend = "📊 Stable"
        
        return {
            'current_temperature': current_temperature,
            'current_divergence': current_divergence,
            'trend': trend,
            'history_length': len(history),
            'divergence_history': history[-10:]  # Last 10 measurements
        }
    
    def get_domain_temperature_overview(self, domain: str) -> Dict[str, Any]:
        """Get temperature overview for all AEZs in a domain"""
        domain_aezs = [a for a in self.aezs.values() if a.domain == domain and a.is_active]
        
        temperature_counts = {
            'hot': 0,     # Ready to split
            'warm': 0,    # Approaching split
            'cool': 0,    # Moderate diversity
            'cold': 0,    # Highly coherent
            'unknown': 0  # Insufficient data
        }
        
        hot_aezs = []
        warm_aezs = []
        
        for aez in domain_aezs:
            temp_summary = self.get_aez_temperature_summary(aez.aez_id)
            temp = temp_summary.get('current_temperature', '⚪ UNKNOWN')
            
            if '🔥 HOT' in temp:
                temperature_counts['hot'] += 1
                hot_aezs.append({
                    'aez_id': aez.aez_id,
                    'event_count': len(aez.event_ids),
                    'divergence': temp_summary.get('current_divergence')
                })
            elif '🟡 WARM' in temp:
                temperature_counts['warm'] += 1
                warm_aezs.append({
                    'aez_id': aez.aez_id,
                    'event_count': len(aez.event_ids),
                    'divergence': temp_summary.get('current_divergence')
                })
            elif '🟠 COOL' in temp:
                temperature_counts['cool'] += 1
            elif '❄️ COLD' in temp:
                temperature_counts['cold'] += 1
            else:
                temperature_counts['unknown'] += 1
        
        return {
            'domain': domain,
            'total_active_aezs': len(domain_aezs),
            'temperature_counts': temperature_counts,
            'hot_aezs': hot_aezs,
            'warm_aezs': warm_aezs,
            'split_probability': 'High' if hot_aezs else 'Medium' if warm_aezs else 'Low'
        }

    def get_domain_summary(self, domain: str) -> Dict[str, Any]:
        """Get summary statistics for a domain"""
        domain_events = [e for e in self.events.values() if e.domain == domain]
        domain_aezs = [a for a in self.aezs.values() if a.domain == domain]
        domain_lineages = [l for l in self.lineages.values() if l.domain == domain]
        
        active_aezs = [a for a in domain_aezs if a.is_active]
        
        # Add temperature overview
        temp_overview = self.get_domain_temperature_overview(domain)
        
        return {
            'domain': domain,
            'total_events': len(domain_events),
            'total_aezs': len(domain_aezs),
            'active_aezs': len(active_aezs),
            'total_lineages': len(domain_lineages),
            'root_lineage_id': self.domain_roots.get(domain),
            'temperature_overview': temp_overview
        }
    
    def get_phylogeny_summary(self) -> Dict[str, Any]:
        """Get complete phylogeny summary"""
        return {
            'total_events': len(self.events),
            'total_aezs': len(self.aezs),
            'active_aezs': len([a for a in self.aezs.values() if a.is_active]),
            'total_lineages': len(self.lineages),
            'domains': list(self.domain_roots.keys()),
            'domain_summaries': {domain: self.get_domain_summary(domain) 
                               for domain in self.domain_roots.keys()}
        }
    
    def save_phylogeny_state(self, output_path: str):
        """Save complete phylogeny state to JSON"""
        state = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'similarity_threshold': self.similarity_threshold,
                'divergence_threshold': self.divergence_threshold,
                'min_events_for_divergence': self.min_events_for_divergence
            },
            'events': {eid: {
                'event_id': e.event_id,
                'summary': e.summary,
                'full_category': e.full_category,
                'domain': e.domain,
                'date': e.date,
                'aez_id': e.aez_id,
                'lineage_id': e.lineage_id,
                'metadata': e.metadata
            } for eid, e in self.events.items()},
            'aezs': {aid: {
                'aez_id': a.aez_id,
                'domain': a.domain,
                'event_ids': a.event_ids,
                'parent_aez_id': a.parent_aez_id,
                'parent_lineage_id': a.parent_lineage_id,
                'created_at': a.created_at,
                'is_active': a.is_active,
                'divergence_history': a.divergence_history,
                'metadata': a.metadata
            } for aid, a in self.aezs.items()},
            'lineages': {lid: {
                'lineage_id': l.lineage_id,
                'domain': l.domain,
                'node_type': l.node_type,
                'name': l.name,
                'parent_lineage_id': l.parent_lineage_id,
                'children_aez_ids': l.children_aez_ids,
                'children_lineage_ids': l.children_lineage_ids,
                'ancestor_aez_id': l.ancestor_aez_id,
                'created_at': l.created_at,
                'metadata': l.metadata
            } for lid, l in self.lineages.items()},
            'domain_roots': self.domain_roots,
            'summary': self.get_phylogeny_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved phylogeny state to {output_path}")