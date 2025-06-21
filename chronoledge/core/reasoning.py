"""
Reasoning Module for ChronoLedge

This module provides AI-powered analysis of phylogenetic data including:
- AEZ cluster summarization
- Evolutionary path explanation  
- Free-form question answering
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from chronoledge.core.config import settings

logger = logging.getLogger(__name__)

class PhylogenyReasoner:
    """AI-powered reasoning engine for phylogenetic analysis"""
    
    def __init__(self):
        self.provider = settings.REASONING_PROVIDER
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate AI client based on configuration"""
        if self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"Initialized OpenAI client with model: {settings.OPENAI_MODEL}")
            except ImportError:
                logger.error("OpenAI package not installed. Install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        
        elif self.provider == "claude":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.CLAUDE_API_KEY)
                logger.info(f"Initialized Claude client with model: {settings.CLAUDE_MODEL}")
            except ImportError:
                logger.error("Anthropic package not installed. Install with: pip install anthropic")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Claude client: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported reasoning provider: {self.provider}")
    
    def _call_model(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """Call the configured AI model with the given messages"""
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    max_tokens=max_tokens or settings.OPENAI_MAX_TOKENS,
                    temperature=settings.OPENAI_TEMPERATURE
                )
                return response.choices[0].message.content
            
            elif self.provider == "claude":
                # Format messages for Claude API
                system_message = None
                formatted_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        formatted_messages.append(msg)
                
                response = self._client.messages.create(
                    model=settings.CLAUDE_MODEL,
                    max_tokens=max_tokens or settings.CLAUDE_MAX_TOKENS,
                    temperature=settings.CLAUDE_TEMPERATURE,
                    system=system_message,
                    messages=formatted_messages
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error calling {self.provider} model: {e}")
            raise
    
    def analyze_aez_cluster(self, aez_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and summarize an AEZ cluster
        
        Args:
            aez_data: Complete AEZ data including events, temperature, etc.
            
        Returns:
            Analysis results with summary, insights, and recommendations
        """
        try:
            # Prepare AEZ information
            aez_info = {
                "aez_id": aez_data.get("id", "unknown"),
                "event_count": aez_data.get("event_count", 0),
                "temperature": aez_data.get("temperature", {}),
                "events": aez_data.get("event_details", [])[:10],  # Limit to 10 events
                "sample_events": aez_data.get("sample_events", [])
            }
            
            # Create analysis prompt
            system_prompt = """You are an expert in semantic phylogeny analysis. You analyze Anchored Embedding Zones (AEZs) - semantic clusters of related events that evolve over time.

Your task is to provide insightful analysis of AEZ clusters, focusing on:
1. Semantic coherence and themes
2. Evolution potential based on temperature
3. Key events and patterns
4. Splitting probability and timeline

Provide clear, actionable insights for researchers studying semantic evolution."""

            user_prompt = f"""Analyze this AEZ cluster:

**AEZ Information:**
- ID: {aez_info['aez_id']}
- Event Count: {aez_info['event_count']}
- Temperature: {aez_info['temperature'].get('current_temperature', 'Unknown')}
- Divergence Score: {aez_info['temperature'].get('current_divergence', 'N/A')}
- Trend: {aez_info['temperature'].get('trend', 'Unknown')}

**Sample Events:**
{json.dumps(aez_info['events'], indent=2)}

Please provide:
1. **Semantic Summary**: What themes/topics does this cluster represent?
2. **Evolution Analysis**: Based on temperature, what's the splitting probability?
3. **Key Insights**: What patterns do you observe?
4. **Recommendations**: What should researchers watch for?

Keep the analysis concise but insightful."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            analysis = self._call_model(messages)
            
            return {
                "status": "success",
                "analysis_type": "aez_cluster",
                "aez_id": aez_info["aez_id"],
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "metadata": {
                    "event_count": aez_info["event_count"],
                    "temperature": aez_info["temperature"],
                    "model_used": settings.OPENAI_MODEL if self.provider == "openai" else settings.CLAUDE_MODEL
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing AEZ cluster: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_type": "aez_cluster"
            }
    
    def explain_evolutionary_path(self, path_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain an evolutionary path and sibling differences for an AEZ
        
        Args:
            path_data: Complete evolutionary context including target AEZ, siblings, and path
            
        Returns:
            Explanation of how the AEZ differs from its siblings and evolutionary context
        """
        try:
            target_aez = path_data.get("target_aez", {})
            siblings = path_data.get("siblings", [])
            evolutionary_path = path_data.get("evolutionary_path", [])
            ancestor_aez = path_data.get("ancestor_aez", {})
            split_context = path_data.get("split_context", {})
            
            system_prompt = """You are an expert in semantic phylogeny and evolutionary divergence analysis. You specialize in explaining how semantic clusters (AEZs) differentiate from their siblings after evolutionary splits.

Your task is to analyze semantic divergence between sibling AEZs, focusing on:
1. **Semantic Differentiation**: How the target AEZ differs from its siblings
2. **Divergence Triggers**: What themes/topics caused the original split
3. **Evolutionary Context**: The path that led to this specialization
4. **Comparative Analysis**: Unique characteristics of each sibling cluster

Provide clear, insightful analysis that helps researchers understand semantic evolution patterns."""

            # Build detailed comparison content
            target_events_text = "\n".join([f"- {event['summary'][:150]}..." for event in target_aez.get("events", [])[:3]])
            
            siblings_text = ""
            for i, sibling in enumerate(siblings):
                sibling_events = "\n".join([f"  - {event['summary'][:100]}..." for event in sibling.get("events", [])[:2]])
                siblings_text += f"""
**Sibling {i+1}: AEZ-{sibling['aez_id'][-8:]}**
- Event Count: {sibling['event_count']}
- Temperature: {sibling.get('temperature', {}).get('current_temperature', 'Unknown')}
- Sample Events:
{sibling_events}
"""

            user_prompt = f"""Analyze the evolutionary differentiation of this AEZ cluster:

**TARGET AEZ ANALYSIS:**
- AEZ ID: {target_aez['aez_id']}
- Domain: {target_aez['domain']}
- Event Count: {target_aez['event_count']}
- Temperature: {target_aez.get('temperature', {}).get('current_temperature', 'Unknown')}
- Sample Events:
{target_events_text}

**SIBLING AEZs (Same Lineage):**
{siblings_text}

**EVOLUTIONARY CONTEXT:**
- Total Siblings: {split_context.get('total_siblings', 'Unknown')}
- Parent Lineage: {split_context.get('parent_lineage_id', 'Unknown')}
- Original AEZ: {ancestor_aez.get('aez_id', 'Unknown') if ancestor_aez else 'Unknown'}

**EVOLUTIONARY PATH:**
{json.dumps(evolutionary_path, indent=2)}

Please provide:
1. **Semantic Differentiation**: How does the target AEZ differ semantically from its siblings?
2. **Split Analysis**: What divergent themes caused the original cluster to split into these groups?
3. **Specialization**: What unique semantic niche does the target AEZ occupy?
4. **Evolutionary Insights**: What does this split reveal about semantic evolution in this domain?

Focus on the semantic differences and evolutionary significance of this divergence."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            analysis = self._call_model(messages)
            
            return {
                "status": "success",
                "analysis_type": "evolutionary_path",
                "target_aez_id": target_aez.get("aez_id"),
                "sibling_count": len(siblings),
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "metadata": {
                    "target_aez": target_aez.get("aez_id"),
                    "siblings_analyzed": len(siblings),
                    "path_length": len(evolutionary_path),
                    "domain": target_aez.get("domain"),
                    "model_used": settings.OPENAI_MODEL if self.provider == "openai" else settings.CLAUDE_MODEL
                }
            }
            
        except Exception as e:
            logger.error(f"Error explaining evolutionary path: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_type": "evolutionary_path"
            }
    
    def answer_free_question(self, question: str, context_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Answer free-form questions about phylogenetic data
        
        Args:
            question: User's question
            context_data: Optional context data (current domain, visible nodes, etc.)
            
        Returns:
            Answer and insights
        """
        try:
            # TODO: Implement comprehensive free-form question answering
            # This will include domain-specific context and data integration
            
            context_info = ""
            if context_data:
                context_info = f"\n**Current Context:**\n{json.dumps(context_data, indent=2)}\n"
            
            system_prompt = """You are an expert assistant for ChronoLedge, a semantic phylogeny analysis system. You help researchers understand:

- Anchored Embedding Zones (AEZs): Semantic clusters of related events
- Phylogenetic trees: Evolutionary structure showing how topics diverge
- Temperature system: Semantic "heat" indicating splitting probability
- Evolutionary patterns: How semantic meanings evolve over time

Provide helpful, accurate answers based on phylogenetic principles and the available data. If you need more specific data to answer fully, suggest what information would be helpful."""

            user_prompt = f"""Question: {question}{context_info}

Please provide a helpful answer based on phylogenetic analysis principles and the available context. If the question requires specific data that isn't provided, explain what information would be needed."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            analysis = self._call_model(messages)
            
            return {
                "status": "success",
                "analysis_type": "free_question",
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "metadata": {
                    "has_context": context_data is not None,
                    "model_used": settings.OPENAI_MODEL if self.provider == "openai" else settings.CLAUDE_MODEL
                }
            }
            
        except Exception as e:
            logger.error(f"Error answering free question: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_type": "free_question"
            }

# Global reasoner instance
reasoner = PhylogenyReasoner() 