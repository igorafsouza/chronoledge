"""
Main entry point for the ChronoLedge application.
"""

import logging
from pathlib import Path
from fastapi import FastAPI, Request, APIRouter, Query, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
import asyncio
import requests
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from chronoledge.core.config import settings, LOGS_DIR
from chronoledge.core.reasoning import reasoner
from chronoledge.etl.extractors.wikimedia import WikimediaExtractor
from chronoledge.etl.extractors.data_manager import EventDataManager
from chronoledge.etl.semantic_loaders.phylogeny_tree_engine import SemanticPhylogenyEngine

# Configure logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="ChronoLedge",
    description="A time-series knowledge graph system with LLM-based reasoning",
    version="0.1.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="chronoledge/frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="chronoledge/frontend/templates")

etl_router = APIRouter(prefix="/api/etl", tags=["ETL"])
query_router = APIRouter(prefix="/api/query", tags=["Semantic Query"])
phylogeny_router = APIRouter(prefix="/api/phylogeny", tags=["Phylogeny"])
chat_router = APIRouter(prefix="/api/chat", tags=["Chat"])
reasoning_router = APIRouter(prefix="/api/reasoning", tags=["AI Reasoning"])

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    phylogeny_data: Optional[Dict[str, Any]] = None

class AEZAnalysisRequest(BaseModel):
    aez_data: Dict[str, Any]

class EvolutionaryPathRequest(BaseModel):
    path_data: Dict[str, Any]

class FreeQuestionRequest(BaseModel):
    question: str
    context_data: Optional[Dict[str, Any]] = None

# Initialize components
data_manager = EventDataManager()

# Global pipeline state for progress tracking
current_pipeline_progress = []

# Global phylogeny engine instance
phylogeny_engine = None
phylogeny_state = {
    "last_processed": None,
    "summary": None,
    "domains": []
}

# === ETL ENDPOINTS ===

@etl_router.get("/wikimedia/current-events")
async def get_wikimedia_current_events(date: Optional[str] = Query(None, description="Date in YYYY_Month_DD format")):
    """Fetch current events from Wikimedia API."""
    try:
        async with WikimediaExtractor() as extractor:
            if date:
                # Extract specific date if not already extracted
                if data_manager.is_date_extracted(date):
                    logger.info(f"Date {date} already extracted, loading from cache")
                    cached_data = data_manager.load_events_for_date(date)
                    return {
                        "status": "success",
                        "source": "cache",
                        "date": date,
                        "events": cached_data.get("events", []) if cached_data else []
                    }
                else:
                    events = await extractor.get_current_events(date)
                    if events:
                        data_manager.save_events(events, date, "date")
                    return {"status": "success", "source": "fresh", "date": date, "events": events}
            else:
                # Extract current events (today + unextracted recent dates)
                # First, get the current page to see available dates
                response = requests.get(extractor.SCRAPEAPI_ENDPOINT, params={
                    "api_key": settings.SCRAPEAPI_KEY,
                    "url": extractor.PORTAL_URL
                })
                response.raise_for_status()
                
                # Get available dates from the current page
                available_dates = extractor.get_available_dates_from_current(response.text)
                logger.info(f"Found {len(available_dates)} available dates on current page")
                
                # Get dates that haven't been extracted yet
                unextracted_dates = data_manager.get_unextracted_dates(available_dates)
                logger.info(f"Found {len(unextracted_dates)} unextracted dates: {unextracted_dates}")
                
                all_events = []
                extracted_count = 0
                
                # Extract events for each unextracted date
                for date_id in unextracted_dates[:5]:  # Limit to 5 most recent dates
                    try:
                        events = await extractor.get_current_events(date_id)
                        if events:
                            data_manager.save_events(events, date_id, "current")
                            all_events.extend(events)
                            extracted_count += 1
                            logger.info(f"Extracted and saved {len(events)} events for {date_id}")
                    except Exception as e:
                        logger.error(f"Error extracting events for {date_id}: {str(e)}")
                
                return {
                    "status": "success", 
                    "source": "current", 
                    "events": all_events,
                    "dates_processed": extracted_count,
                    "total_available_dates": len(available_dates),
                    "unextracted_dates": len(unextracted_dates)
                }
                
    except Exception as e:
        logger.error(f"Error extracting events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@etl_router.get("/wikimedia/month-events")
async def extract_month_events(
    year: int = Query(..., description="Year (e.g., 2025)"),
    month: str = Query(..., description="Month name (e.g., 'June')")
):
    """Extract all events for a specific month."""
    try:
        async with WikimediaExtractor() as extractor:
            events_by_date = await extractor.get_month_events(year, month)
            
            total_events = 0
            new_dates = 0
            
            # Save events for each date
            for date_id, events in events_by_date.items():
                if not data_manager.is_date_extracted(date_id):
                    data_manager.save_events(events, date_id, "monthly")
                    new_dates += 1
                total_events += len(events)
            
            # Flatten all events for response
            all_events = []
            for events in events_by_date.values():
                all_events.extend(events)
            
            return {
                "status": "success",
                "year": year,
                "month": month,
                "total_events": total_events,
                "new_dates_extracted": new_dates,
                "total_dates": len(events_by_date),
                "events": all_events
            }
            
    except Exception as e:
        logger.error(f"Error extracting month events: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@etl_router.get("/extraction-status")
async def get_extraction_status():
    """Get the current extraction status and history."""
    try:
        summary = data_manager.get_extraction_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"Error getting extraction status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@etl_router.get("/semantic-pipeline/status")
async def get_semantic_pipeline_status():
    """Get the current semantic pipeline status."""
    try:
        # Check if phylogeny engine components exist
        import os
        from pathlib import Path
        
        data_dir = Path("data")
        output_dir = Path("output")
        
        # Check for event files
        events_dir = data_dir / "events"
        event_files = []
        events_count = 0
        if events_dir.exists():
            event_files = sorted([f.name for f in events_dir.glob("*.json")])
            events_count = len(event_files)
        
        # Check phylogeny engine files
        phylogeny_state_file = output_dir / "phylogeny_state.json"
        
        phylogeny_status = {
            "lineage_nodes_file_exists": False,
            "aezs_file_exists": False, 
            "events_file_exists": events_count > 0,
            "graph_file_exists": False
        }
        
        # Check if phylogeny state file exists (this contains all the necessary data)
        if phylogeny_state_file.exists():
            phylogeny_status["lineage_nodes_file_exists"] = True
            phylogeny_status["aezs_file_exists"] = True
            phylogeny_status["graph_file_exists"] = True
        
        return {
            "status": "success",
            "pipeline_status": {
                "components": {
                    "phylogeny_engine": phylogeny_status
                },
                "data_status": {
                    "events_count": events_count,
                    "event_files": event_files
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@etl_router.post("/semantic-pipeline/run")
async def run_semantic_pipeline(
    background_tasks: BackgroundTasks,
    force_rebuild: bool = Query(False, description="Force rebuild of existing phylogeny data")
):
    """Run the semantic phylogeny pipeline."""
    try:
        import time
        from datetime import datetime
        
        # Clear existing engine if force rebuild
        global phylogeny_engine
        if force_rebuild:
            phylogeny_engine = None
            logger.info("Force rebuild requested - clearing existing phylogeny engine")
        
        start_time = time.time()
        
        # Load/create phylogeny engine and process data
        logger.info("Starting phylogeny pipeline...")
        engine = load_or_create_phylogeny_engine()
        
        # Get summary
        summary = engine.get_phylogeny_summary() if engine else {}
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Enhanced summary with additional stats
        enhanced_summary = {
            **summary,
            "total_domains": len(summary.get("domains", [])),
            "total_lineage_nodes": summary.get("total_lineages", 0),
            "graph_nodes": summary.get("total_aezs", 0) + summary.get("total_lineages", 0),
            "graph_edges": summary.get("total_lineages", 0)  # Approximate
        }
        
        return {
            "status": "success",
            "message": "Phylogeny pipeline completed successfully",
            "pipeline_results": {
                "phylogeny_summary": enhanced_summary,
                "duration_seconds": duration,
                "force_rebuild": force_rebuild,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error running semantic pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@etl_router.get("/semantic-pipeline/progress") 
async def get_pipeline_progress():
    """Get current pipeline progress (simplified for now)."""
    try:
        # For now, return a simple progress update
        # In a full implementation, you'd track actual progress
        return {
            "status": "success",
            "last_update": {
                "stage": "completed",
                "status": "completed", 
                "progress_percent": 100.0,
                "message": "Pipeline ready - check status for details"
            }
        }
    except Exception as e:
        logger.error(f"Error getting pipeline progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# === PHYLOGENY ENDPOINTS ===

def load_or_create_phylogeny_engine():
    """Load phylogeny engine with existing data or create new one"""
    global phylogeny_engine, phylogeny_state
    
    if phylogeny_engine is None:
        logger.info("Initializing phylogeny engine...")
        phylogeny_engine = SemanticPhylogenyEngine()
        
        # Try to load existing phylogeny state
        state_file = Path("output/phylogeny_state.json")
        if state_file.exists():
            try:
                logger.info("Loading existing phylogeny state...")
                # For now, we'll process events fresh each time
                # In a production system, you'd want to save/restore the full engine state
                pass
            except Exception as e:
                logger.warning(f"Could not load phylogeny state: {e}")
        
        # Process available events
        try:
            logger.info("Processing events for phylogeny analysis...")
            phylogeny_engine.process_chronological_events("data/events")
            
            # Update global state
            phylogeny_state["summary"] = phylogeny_engine.get_phylogeny_summary()
            phylogeny_state["domains"] = phylogeny_state["summary"].get("domains", [])
            phylogeny_state["last_processed"] = "now"
            
            logger.info(f"Phylogeny processing complete. Found {len(phylogeny_state['domains'])} domains.")
            
            state_file = Path("output/phylogeny_state.json")
            phylogeny_engine.save_phylogeny_state(str(state_file))

        except Exception as e:
            logger.error(f"Error processing phylogeny: {e}")
            phylogeny_state["summary"] = {
                "total_events": 0,
                "total_aezs": 0,
                "active_aezs": 0,
                "total_lineages": 0,
                "domains": [],
                "domain_summaries": {}
            }
    
    return phylogeny_engine

@phylogeny_router.get("/data")
async def get_phylogeny_data():
    """Get overall phylogeny data"""
    try:
        load_or_create_phylogeny_engine()
        return {
            "status": "success",
            "last_processed": phylogeny_state["last_processed"],
            **phylogeny_state["summary"]
        }
    except Exception as e:
        logger.error(f"Error getting phylogeny data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@phylogeny_router.get("/tree/{domain}")
async def get_domain_tree(domain: str):
    """Get tree structure for a specific domain (phylogenetic structure only)"""
    try:
        engine = load_or_create_phylogeny_engine()
        
        # Get domain data
        domain_lineages = [l for l in engine.lineages.values() if l.domain == domain]
        domain_aezs = [a for a in engine.aezs.values() if a.domain == domain]
        
        if not domain_lineages and not domain_aezs:
            return {"nodes": []}
        
        # Convert to tree structure for D3 (phylogenetic structure only)
        nodes = []
        
        # Add domain root
        nodes.append({
            "id": f"domain-{domain}",
            "label": domain,
            "type": "domain",
            "parent": None,
            "domain": domain
        })
        
        # Add lineage nodes
        for lineage in domain_lineages:
            nodes.append({
                "id": lineage.lineage_id,
                "label": f"L-{lineage.lineage_id[-6:]}",
                "type": "lineage", 
                "parent": lineage.parent_lineage_id or f"domain-{domain}",
                "domain": domain,
                "children_count": len(lineage.children_aez_ids) + len(lineage.children_lineage_ids)
            })
        
        # Add AEZ nodes with enhanced event information (only active AEZs for clean visualization)
        for aez in domain_aezs:
            # Skip inactive AEZs - they've split and are replaced by lineage nodes
            if not aez.is_active:
                continue
                
            sample_events = []
            event_details = []
            
            if aez.event_ids:
                # Get sample events for quick display (first 5)
                for event_id in aez.event_ids[:5]:
                    if event_id in engine.events:
                        event = engine.events[event_id]
                        sample_events.append(event.summary)
                
                # Get ALL event details for inspector
                for event_id in aez.event_ids:
                    if event_id in engine.events:
                        event = engine.events[event_id]
                        event_details.append({
                            "id": event_id,
                            "summary": event.summary,
                            "date": getattr(event, 'date', None),
                            "category": getattr(event, 'category', None)
                        })
            
            # Get temperature information
            temp_summary = engine.get_aez_temperature_summary(aez.aez_id)
            
            nodes.append({
                "id": aez.aez_id,
                "label": f"AEZ-{aez.aez_id[-6:]}",
                "type": "aez",
                "parent": aez.parent_lineage_id or f"domain-{domain}",
                "active": aez.is_active,
                "event_count": len(aez.event_ids),
                "sample_events": sample_events,
                "event_details": event_details,  # Full event details for inspector
                "temperature": temp_summary  # Temperature and divergence data
            })
        
        return {"nodes": nodes}
        
    except Exception as e:
        logger.error(f"Error getting tree for domain {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@phylogeny_router.get("/evolutionary-path/{aez_id}")
async def get_aez_evolutionary_path(aez_id: str):
    """Get evolutionary path and sibling analysis for a specific AEZ"""
    try:
        engine = load_or_create_phylogeny_engine()
        
        if aez_id not in engine.aezs:
            raise HTTPException(status_code=404, detail="AEZ not found")
        
        target_aez = engine.aezs[aez_id]
        
        # Get evolutionary path from root to this AEZ
        path = []
        current_lineage_id = target_aez.parent_lineage_id
        
        # Trace back to domain root
        while current_lineage_id and current_lineage_id in engine.lineages:
            lineage = engine.lineages[current_lineage_id]
            path.insert(0, {
                "id": lineage.lineage_id,
                "type": "lineage",
                "name": lineage.name,
                "created_at": lineage.created_at,
                "ancestor_aez_id": lineage.ancestor_aez_id
            })
            current_lineage_id = lineage.parent_lineage_id
        
        # Get sibling AEZs (those sharing the same parent lineage)
        siblings = []
        if target_aez.parent_lineage_id and target_aez.parent_lineage_id in engine.lineages:
            parent_lineage = engine.lineages[target_aez.parent_lineage_id]
            
            for sibling_aez_id in parent_lineage.children_aez_ids:
                if sibling_aez_id != aez_id and sibling_aez_id in engine.aezs:
                    sibling_aez = engine.aezs[sibling_aez_id]
                    
                    # Get all sibling events for comparison
                    sibling_events = []
                    for event_id in sibling_aez.event_ids:
                        if event_id in engine.events:
                            event = engine.events[event_id]
                            sibling_events.append({
                                "id": event_id,
                                "summary": event.summary,
                                "date": getattr(event, 'date', None),
                                "category": getattr(event, 'category', None)
                            })
                    
                    siblings.append({
                        "aez_id": sibling_aez.aez_id,
                        "event_count": len(sibling_aez.event_ids),
                        "is_active": sibling_aez.is_active,
                        "events": sibling_events,
                        "temperature": engine.get_aez_temperature_summary(sibling_aez.aez_id)
                    })
        
        # Get target AEZ detailed information
        target_events = []
        for event_id in target_aez.event_ids:
            if event_id in engine.events:
                event = engine.events[event_id]
                target_events.append({
                    "id": event_id,
                    "summary": event.summary,
                    "date": getattr(event, 'date', None),
                    "category": getattr(event, 'category', None)
                })
        
        # Get the original AEZ that split (ancestor)
        ancestor_info = None
        if path and path[-1]["ancestor_aez_id"]:
            ancestor_aez_id = path[-1]["ancestor_aez_id"]
            if ancestor_aez_id in engine.aezs:
                ancestor_aez = engine.aezs[ancestor_aez_id]
                ancestor_info = {
                    "aez_id": ancestor_aez.aez_id,
                    "event_count": len(ancestor_aez.event_ids),
                    "created_at": ancestor_aez.created_at,
                    "is_active": ancestor_aez.is_active
                }
        
        return {
            "target_aez": {
                "aez_id": target_aez.aez_id,
                "domain": target_aez.domain,
                "event_count": len(target_aez.event_ids),
                "is_active": target_aez.is_active,
                "events": target_events,
                "temperature": engine.get_aez_temperature_summary(aez_id)
            },
            "evolutionary_path": path,
            "siblings": siblings,
            "ancestor_aez": ancestor_info,
            "split_context": {
                "parent_lineage_id": target_aez.parent_lineage_id,
                "total_siblings": len(siblings) + 1,  # Include target AEZ
                "domain": target_aez.domain
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting evolutionary path for AEZ {aez_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === CHAT ENDPOINTS ===

@chat_router.post("/phylogeny")
async def chat_about_phylogeny(chat_message: ChatMessage):
    """Chat with AI about phylogeny patterns"""
    try:
        # For now, return a simple response
        # In a real implementation, you'd integrate with an LLM here
        message = chat_message.message.lower()
        
        # Simple pattern matching for demonstration
        if "domain" in message:
            if phylogeny_state["domains"]:
                response = f"I can see {len(phylogeny_state['domains'])} domains in the data: {', '.join(phylogeny_state['domains'])}. "
                response += "Each domain represents a different category of events that evolve independently in the phylogenetic tree."
            else:
                response = "No domains are currently available. You may need to process some event data first."
        
        elif "evolution" in message or "split" in message:
            summary = phylogeny_state.get("summary", {})
            total_aezs = summary.get("total_aezs", 0)
            active_aezs = summary.get("active_aezs", 0)
            splits = total_aezs - active_aezs
            
            if splits > 0:
                response = f"I can see evidence of semantic evolution! There have been {splits} AEZ splits, "
                response += f"indicating that {splits} semantic clusters have divided into more specialized sub-clusters. "
                response += "This suggests the topics are becoming more semantically diverse over time."
            else:
                response = "I don't see any major evolutionary splits yet. This could mean the events are either too similar semantically, or there isn't enough data to trigger divergence."
        
        elif "lineage" in message:
            summary = phylogeny_state.get("summary", {})
            total_lineages = summary.get("total_lineages", 0)
            response = f"The phylogenetic tree currently has {total_lineages} lineage nodes. "
            response += "Lineages represent evolutionary branch points where semantic clusters have split into more specialized groups."
        
        elif "help" in message or "what" in message:
            response = """I can help you understand phylogenetic patterns in your data! I can explain:
            
            🌳 **Domains**: Different categories of events (e.g., "Politics", "Science")
            🔹 **AEZs (Anchored Embedding Zones)**: Semantic clusters of similar events
            🌿 **Lineages**: Evolutionary branch points where topics split
            📈 **Evolution**: How semantic topics diverge and specialize over time
            
            Try asking about specific domains, evolution patterns, or lineage structures!"""
        
        else:
            response = "I'd be happy to help you explore the phylogenetic patterns! You can ask me about domains, evolution, lineages, or specific patterns you notice in the trees."
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in phylogeny chat: {e}")
        raise HTTPException(status_code=500, detail="Sorry, there was an error processing your question.")

# === REASONING ENDPOINTS ===

@reasoning_router.post("/analyze-aez")
async def analyze_aez_cluster(request: AEZAnalysisRequest):
    """Analyze and summarize an AEZ cluster using AI reasoning"""
    try:
        analysis = reasoner.analyze_aez_cluster(request.aez_data)
        return analysis
    except Exception as e:
        logger.error(f"Error in AEZ analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@reasoning_router.post("/explain-path")
async def explain_evolutionary_path(request: EvolutionaryPathRequest):
    """Explain an evolutionary path through the phylogenetic tree using AI reasoning"""
    try:
        analysis = reasoner.explain_evolutionary_path(request.path_data)
        return analysis
    except Exception as e:
        logger.error(f"Error in evolutionary path explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Path explanation failed: {str(e)}")

@reasoning_router.post("/free-question")
async def answer_free_question(request: FreeQuestionRequest):
    """Answer free-form questions about phylogenetic data using AI reasoning"""
    try:
        analysis = reasoner.answer_free_question(request.question, request.context_data)
        return analysis
    except Exception as e:
        logger.error(f"Error answering free question: {e}")
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

@reasoning_router.get("/status")
async def get_reasoning_status():
    """Get the status of the reasoning module"""
    try:
        return {
            "status": "active",
            "provider": settings.REASONING_PROVIDER,
            "model": settings.OPENAI_MODEL if settings.REASONING_PROVIDER == "openai" else settings.CLAUDE_MODEL,
            "available_endpoints": [
                "/analyze-aez - Analyze AEZ clusters",
                "/explain-path - Explain evolutionary paths", 
                "/free-question - Answer free-form questions"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting reasoning status: {e}")
        return {"status": "error", "error": str(e)}

# === MAIN APP ROUTES ===

@app.get("/")
async def root(request: Request):
    """Root endpoint - Adventure page."""
    return templates.TemplateResponse("adventure.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ChronoLedge"}

@app.get("/etl", response_class=HTMLResponse)
async def etl_page(request: Request):
    """Render the ETL management page."""
    return templates.TemplateResponse("etl.html", {"request": request})

@app.get("/adventure", response_class=HTMLResponse)
async def adventure_page(request: Request):
    """Render the phylogeny adventure page."""
    return templates.TemplateResponse("adventure.html", {"request": request})



app.include_router(etl_router)
app.include_router(query_router)
app.include_router(phylogeny_router)
app.include_router(chat_router)
app.include_router(reasoning_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chronoledge.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 