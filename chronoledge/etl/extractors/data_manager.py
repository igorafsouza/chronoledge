"""
Data manager for storing and tracking extracted events.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class EventDataManager:
    """Manages storage and tracking of extracted events to prevent duplicates."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.events_dir = self.data_dir / "events"
        self.metadata_file = self.data_dir / "extraction_metadata.json"
        
        # Create directories if they don't exist
        self.events_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load extraction metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            "extracted_dates": [],
            "last_extraction": None,
            "extraction_history": []
        }
    
    def _save_metadata(self):
        """Save extraction metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def is_date_extracted(self, date_id: str) -> bool:
        """Check if a specific date has already been extracted.
        
        Args:
            date_id: Date in format 'YYYY_Month_DD' (e.g., '2025_June_17')
        """
        return date_id in self.metadata.get("extracted_dates", [])
    
    def get_unextracted_dates(self, available_dates: List[str]) -> List[str]:
        """Get list of dates that haven't been extracted yet.
        
        Args:
            available_dates: List of available date IDs
            
        Returns:
            List of date IDs that haven't been extracted
        """
        extracted = set(self.metadata.get("extracted_dates", []))
        return [date for date in available_dates if date not in extracted]
    
    def save_events(self, events: List[Dict], date_id: str, extraction_type: str = "daily"):
        """Save events for a specific date.
        
        Args:
            events: List of event dictionaries
            date_id: Date ID (e.g., '2025_June_17')
            extraction_type: Type of extraction ('daily', 'monthly', 'current')
        """
        # Create filename based on date
        filename = self.events_dir / f"{date_id}.json"
        
        # Save events
        with open(filename, 'w') as f:
            json.dump({
                "date_id": date_id,
                "extraction_type": extraction_type,
                "extraction_timestamp": datetime.now().isoformat(),
                "event_count": len(events),
                "events": events
            }, f, indent=2)
        
        # Update metadata
        if date_id not in self.metadata["extracted_dates"]:
            self.metadata["extracted_dates"].append(date_id)
        
        self.metadata["last_extraction"] = datetime.now().isoformat()
        self.metadata["extraction_history"].append({
            "timestamp": datetime.now().isoformat(),
            "date_id": date_id,
            "event_count": len(events),
            "extraction_type": extraction_type
        })
        
        self._save_metadata()
        logger.info(f"Saved {len(events)} events for {date_id}")
    
    def load_events_for_date(self, date_id: str) -> Optional[Dict]:
        """Load events for a specific date.
        
        Args:
            date_id: Date ID (e.g., '2025_June_17')
            
        Returns:
            Dictionary with events data or None if not found
        """
        filename = self.events_dir / f"{date_id}.json"
        if filename.exists():
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    
    def load_events_for_month(self, year: int, month: str) -> List[Dict]:
        """Load all events for a specific month.
        
        Args:
            year: Year (e.g., 2025)
            month: Month name (e.g., 'June')
            
        Returns:
            List of all events for the month
        """
        month_prefix = f"{year}_{month}_"
        all_events = []
        
        for filename in self.events_dir.glob(f"{month_prefix}*.json"):
            with open(filename, 'r') as f:
                data = json.load(f)
                all_events.extend(data.get("events", []))
        
        return all_events
    
    def get_extraction_summary(self) -> Dict:
        """Get summary of extraction history."""
        return {
            "total_dates_extracted": len(self.metadata.get("extracted_dates", [])),
            "last_extraction": self.metadata.get("last_extraction"),
            "extracted_dates": sorted(self.metadata.get("extracted_dates", [])),
            "recent_extractions": self.metadata.get("extraction_history", [])[-10:]
        } 