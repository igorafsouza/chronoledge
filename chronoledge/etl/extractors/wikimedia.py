"""
Wikimedia API data extractor.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
import re

from chronoledge.core.config import settings

logger = logging.getLogger(__name__)

class WikimediaExtractor:
    """Extracts current events data from Wikimedia API.
    
    TODO: Some events are still not being captured (~20%), particularly:
    - Events under "Iran-Israel War" on June 17, 2025
    - Need to investigate why certain nested events are missed
    - Currently capturing >80% which is acceptable for POC
    """
    
    BASE_URL = "https://en.wikipedia.org/w/api.php"
    SCRAPEAPI_ENDPOINT = "http://api.scraperapi.com?"
    PORTAL_URL = "https://en.wikipedia.org/wiki/Portal:Current_events"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_events(self, date: Optional[str] = None) -> List[Dict]:
        """
        Fetch current events from Wikipedia Current Events portal using ScrapeAPI.
        Args:
            date: Optional date string in 'YYYY_Month_DD' format (e.g., '2025_June_17'). If None, fetches today.
        Returns:
            List of event dictionaries
        """
        try:
            if date:
                url = f"{self.PORTAL_URL}#{date}"
            else:
                url = self.PORTAL_URL
            
            logger.info(f"Fetching events from URL: {url}")
            
            params = {
                "api_key": settings.SCRAPEAPI_KEY,
                "url": url
            }
            
            # Use synchronous requests since we're mixing sync/async
            response = requests.get(self.SCRAPEAPI_ENDPOINT, params=params)
            response.raise_for_status()
            
            logger.info(f"ScrapeAPI response status: {response.status_code}")
            logger.info(f"Response length: {len(response.text)} characters")
            
            soup = BeautifulSoup(response.text, "html.parser")
            events = []
            
            # Strategy 1: Look for specific date section if date is provided
            if date:
                date_section = soup.find("div", {"id": date, "class": "current-events-main vevent"})
                if date_section:
                    logger.info(f"Found specific date section for {date}")
                    events = self._parse_date_section(date_section, date)
                    logger.info(f"Extracted {len(events)} events for specific date")
                    return events
                else:
                    logger.warning(f"Could not find date section for {date}")
            
            # Strategy 2: Look for any current-events-main sections (recent days)
            date_sections = soup.find_all("div", class_="current-events-main vevent")
            logger.info(f"Found {len(date_sections)} date sections")
            
            for section in date_sections[:3]:  # Limit to 3 most recent days
                section_id = section.get("id", "unknown")
                logger.info(f"Processing section: {section_id}")
                section_events = self._parse_date_section(section, section_id)
                events.extend(section_events)
                
                if len(events) >= 20:  # Limit total events
                    break
            
            # Strategy 3: Fallback - look for "Topics in the news" section
            if not events:
                logger.info("Trying fallback: Topics in the news")
                topics_section = soup.find('h2', string=lambda text: text and 'Topics in the news' in text)
                if topics_section:
                    logger.info("Found Topics in the news section")
                    # Get the next sibling elements
                    current = topics_section.find_next_sibling()
                    while current and current.name != 'h2':
                        if current.name == 'ul':
                            for li in current.find_all('li', recursive=False):
                                text = li.get_text(strip=True)
                                links = self._extract_links_from_element(li)
                                if text and len(text) > 10:
                                    events.append({
                                        "summary": text,
                                        "category": "Topics in the news",
                                        "date": "current",
                                        "links": links,
                                        "level": 0,
                                        "source": "topics_fallback"
                                    })
                        current = current.find_next_sibling()
                else:
                    logger.warning("No Topics in the news section found")
            
            logger.info(f"Extracted {len(events)} total events")
            print(f"DEBUG: Extracted {len(events)} events")  # Debug print
            
            # Return a simple test event if no events found
            if not events:
                events = [{
                    "summary": "Test event - no events found in parsing",
                    "category": "Debug",
                    "date": "unknown",
                    "links": [],
                    "source": "debug"
                }]
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching current events: {str(e)}")
            print(f"DEBUG ERROR: {str(e)}")  # Debug print
            return [{
                "summary": f"Error: {str(e)}",
                "source": "error"
            }]
    
    def _parse_date_section(self, section, date_id: str) -> List[Dict]:
        """Parse a specific date section and extract events by category."""
        events = []
        
        # Get the date title
        date_title_elem = section.find("span", class_="summary")
        date_title = date_title_elem.get_text(strip=True) if date_title_elem else date_id
        
        # Find the content div
        content_div = section.find("div", class_="current-events-content description")
        if not content_div:
            logger.warning(f"No content div found for {date_id}")
            return events
        
        current_category = "General"
        
        # Process all children of the content div
        for element in content_div.children:
            if hasattr(element, 'name'):
                # Check if this is a category header (bold paragraph)
                if element.name == 'p' and element.find('b'):
                    category_text = element.get_text(strip=True)
                    if category_text:
                        current_category = category_text
                        logger.info(f"Found category: {current_category}")
                
                # Process event lists
                elif element.name == 'ul':
                    category_events = self._parse_event_list(element, current_category, date_title)
                    events.extend(category_events)
        
        return events
    
    def _parse_event_list(self, ul_element, category: str, date: str) -> List[Dict]:
        """Parse a list of events under a category, handling hierarchical structure properly."""
        events = []
        
        for li in ul_element.find_all('li', recursive=False):
            # Process this list item with empty topic path initially
            event_data = self._process_list_item(li, category, date, [], 0)
            if event_data:
                events.extend(event_data)
        
        return events
    
    def _process_list_item(self, li_element, category: str, date: str, topic_path: List[str], level: int = 0) -> List[Dict]:
        """Process a single list item, handling nested structure and tracking topic hierarchy."""
        events = []
        
        # Get the direct text content of this li (not including nested ul content)
        direct_text = self._get_direct_text_only(li_element)
        
        # Debug logging for specific events
        if 'tim kaine' in direct_text.lower() or 'television headquarters' in direct_text.lower() or ('air strikes' in direct_text.lower() and 'television' in direct_text.lower()):
            logger.info(f"DEBUG: Processing text at level {level}: '{direct_text[:80]}...'")
            logger.info(f"DEBUG: Current topic path: {topic_path}")
        
        # Extract links from this specific li
        links = self._extract_links_from_element(li_element)
        
        # Find nested ul elements
        nested_uls = li_element.find_all('ul', recursive=False)
        
        # Determine if this is a topic header or an actual event
        is_topic_header = False
        if direct_text and len(direct_text.strip()) > 5:
            # Check if this looks like a topic header
            has_event_indicators = self._contains_event_indicators(direct_text)
            has_source_citations = self._contains_source_citations(direct_text)
            is_short = len(direct_text.strip()) < 100  # Increased threshold slightly
            has_nested = bool(nested_uls)
            
            # Additional checks for topic headers
            text_lower = direct_text.lower().strip()
            
            # Common topic header patterns
            is_conflict_topic = any(term in text_lower for term in [
                'crisis', 'war', 'conflict', 'invasion', 'strikes', 'operation', 'relations'
            ]) and has_nested and not has_source_citations
            
            is_geographic_topic = any(term in text_lower for term in [
                'eastern', 'western', 'northern', 'southern', 'middle east', 'region'
            ]) and has_nested and not has_source_citations
            
            # Check for country/entity relationships (e.g., "Iran–Israel War")
            is_relationship_topic = ('–' in direct_text or '-' in direct_text) and has_nested and not has_source_citations
            
            # Check for parenthetical lists (e.g., "(list of airstrikes)")
            has_list_parenthetical = '(list of' in text_lower and has_nested
            
            # If it's very short and has nested content, likely a topic
            is_short_with_nested = len(direct_text.strip()) < 50 and has_nested and not has_source_citations
            
            is_topic_header = (
                (is_short and has_nested and not has_event_indicators and not has_source_citations) or
                is_conflict_topic or
                is_geographic_topic or
                is_relationship_topic or
                has_list_parenthetical or
                is_short_with_nested
            )
            
            # Debug logging for specific events
            if 'tim kaine' in direct_text.lower() or 'television headquarters' in direct_text.lower() or ('air strikes' in direct_text.lower() and 'television' in direct_text.lower()):
                logger.info(f"DEBUG: has_event_indicators={has_event_indicators}, has_source_citations={has_source_citations}")
                logger.info(f"DEBUG: is_short={is_short}, has_nested={has_nested}")
                logger.info(f"DEBUG: is_topic_header={is_topic_header}")
                logger.info(f"DEBUG: text length={len(direct_text.strip())}")
        
        # Build the current topic path
        current_topic_path = topic_path.copy()
        if is_topic_header and direct_text:
            topic_name = direct_text.strip()
            # Clean up topic name (remove extra info in parentheses for cleaner categories)
            if '(' in topic_name and ')' in topic_name:
                # Keep the part before the first parenthesis for cleaner topic names
                clean_topic = topic_name.split('(')[0].strip()
                if clean_topic:
                    topic_name = clean_topic
            current_topic_path.append(topic_name)
        
        # If this is an actual event (not just a topic header), create an event
        # Use the inherited topic_path (not current_topic_path) for events
        if direct_text and len(direct_text.strip()) > 10 and not is_topic_header:
            # For events, use the inherited topic path from parent
            event_topic_path = topic_path.copy()
            
            # Create the full topic category
            full_category = category
            if event_topic_path:
                full_category = f"{category} > {' > '.join(event_topic_path)}"
            
            # Extract temporal references
            temporal_info = self._extract_temporal_references(direct_text)
            
            event = {
                "summary": direct_text.strip(),
                "category": category,
                "subcategory": " > ".join(event_topic_path) if event_topic_path else None,
                "full_category": full_category,
                "date": date,
                "reporting_date": date,  # When it was reported
                "temporal_info": temporal_info,  # Information about when it actually happened
                "links": links,
                "level": level,
                "topic_path": event_topic_path,
                "source": "parsed_section"
            }
            events.append(event)
            
            # Debug logging for specific events
            if 'tim kaine' in direct_text.lower() or 'television headquarters' in direct_text.lower() or ('air strikes' in direct_text.lower() and 'television' in direct_text.lower()):
                logger.info(f"DEBUG: Created event with full_category: {full_category}")
        
        # Process nested items with the updated topic path (for topic headers)
        if nested_uls:
            for nested_ul in nested_uls:
                for nested_li in nested_ul.find_all('li', recursive=False):
                    # Recursively process nested items with the current topic path
                    nested_events = self._process_list_item(nested_li, category, date, current_topic_path, level + 1)
                    events.extend(nested_events)
        
        return events
    
    def _contains_event_indicators(self, text: str) -> bool:
        """Check if text contains indicators that suggest it's an actual event."""
        event_indicators = [
            # Violence/conflict
            'kills', 'killed', 'dies', 'died', 'attacks', 'attacked', 'launches', 'launched',
            'strikes', 'struck', 'bombing', 'bombed', 'explosion', 'exploded', 'fire', 'fired',
            'shot', 'shooting', 'hit', 'hits', 'target', 'targeted', 'injures', 'injured',
            'wounded', 'casualties', 'victims', 'destroy', 'destroyed', 'damage', 'damaged',
            'intercept', 'intercepted', 'crash', 'crashed', 'collision',
            
            # Political/legal
            'announces', 'announced', 'says', 'said', 'reports', 'reported', 'declares', 'declared',
            'introduces', 'introduced', 'legislation', 'bill', 'vote', 'voted', 'election', 'elected',
            'arrest', 'arrested', 'detained', 'charges', 'charged', 'sentenced', 'convicted',
            'meeting', 'meets', 'summit', 'conference', 'agreement', 'treaty', 'deal',
            
            # Numbers/statistics (often indicate events)
            'people', 'person', 'soldiers', 'civilians', 'employees', 'officials', 'minister',
            'president', 'senator', 'general', 'commander', 'at least', 'more than', 'up to',
            
            # Accidents/disasters
            'accident', 'incident', 'emergency', 'evacuated', 'evacuation', 'rescue', 'rescued',
            'collide', 'collision', 'sinks', 'sunk', 'fire', 'flood', 'earthquake', 'storm',
            
            # Time indicators (suggest specific events)
            'yesterday', 'today', 'tonight', 'morning', 'afternoon', 'evening', 'overnight',
            'hours', 'days ago', 'weeks ago', 'months ago',

            # Adverbs Indicators
            'who', 'what', 'when', 'where', 'why', 'how', 'which', 'that', 'these', 'those',
            'also', 'further', 'more', 'less', 'most', 'least', 'better', 'worse', 'more', 'less',
            'from', 'to', 'in', 'of', 'on', 'at', 'by', 'for', 'with', 'without', 'about', 'above',
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in event_indicators)
    
    def _contains_source_citations(self, text: str) -> bool:
        """Check if text contains news source citations."""
        # Check for parenthetical citations like (Reuters), (BBC News), etc.
        import re
        
        # Pattern to match parenthetical citations
        parenthetical_pattern = r'\([^)]+\)'
        matches = re.findall(parenthetical_pattern, text)
        
        # Common news sources
        news_sources = [
            'reuters', 'bbc', 'cnn', 'al jazeera', 'new york times', 'nyt',
            'washington post', 'associated press', 'ap', 'bloomberg', 'guardian',
            'nos', 'afp', 'abc', 'nbc', 'cbs', 'fox', 'npr', 'pbs', 'wsj',
            'wall street journal', 'financial times', 'ft', 'economist',
            'us congress', 'congress.gov', 'eastleigh voice'
        ]
        
        # Check if any parenthetical matches contain news sources
        for match in matches:
            match_lower = match.lower()
            if any(source in match_lower for source in news_sources):
                return True
        
        # Also check for URLs
        url_indicators = ['http://', 'https://', 'www.', '.com', '.org', '.gov']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in url_indicators)
    
    def _get_direct_text_only(self, element):
        """Get only the direct text of an element, excluding nested ul/li content."""
        # Clone the element to avoid modifying the original
        import copy
        temp_element = copy.copy(element)
        
        # Remove all nested ul elements to get only the direct text
        for nested_ul in temp_element.find_all('ul'):
            nested_ul.decompose()
        
        # Get the text content
        text = temp_element.get_text(separator=' ', strip=True)
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_links_from_element(self, element):
        """Extract links only from the direct content of an element, not nested ul."""
        import copy
        temp_element = copy.copy(element)
        
        # Remove nested ul elements
        for nested_ul in temp_element.find_all('ul'):
            nested_ul.decompose()
        
        # Now extract links from the remaining content
        links = []
        for a_tag in temp_element.find_all('a', href=True):
            href = a_tag.get('href')
            text = a_tag.get_text(strip=True)
            
            # Skip edit/history/watch links
            if any(skip in href for skip in ['action=edit', 'action=history', 'action=watch']):
                continue
            
            # Classify link type
            link_type = "external" if href.startswith('http') else "internal"
            
            # For internal Wikipedia links, make them absolute
            if link_type == "internal" and href.startswith('/wiki/'):
                href = f"https://en.wikipedia.org{href}"
            
            links.append({
                "text": text,
                "url": href,
                "type": link_type
            })
        
        return links
    
    async def get_event_details(self, title: str) -> Dict:
        """
        Fetch detailed information about a specific event.
        
        Args:
            title: The title of the event
            
        Returns:
            Dictionary containing event details
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context manager.")
        
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|categories|links",
            "titles": title,
            "exintro": True,
            "explaintext": True
        }
        
        try:
            async with self.session.get(self.BASE_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    if pages:
                        page_id = next(iter(pages))
                        return pages[page_id]
                else:
                    logger.error(f"Failed to fetch event details: {response.status}")
        
        except Exception as e:
            logger.error(f"Error fetching event details: {str(e)}")
            raise
        
        return {} 

    def _extract_temporal_references(self, text: str) -> Dict[str, any]:
        """Extract temporal references from event text to identify when the event actually occurred."""
        temporal_refs = {
            'yesterday': -1,
            'the day before': -1,
            'two days ago': -2,
            'three days ago': -3,
            'last week': -7,
            'last month': -30,
            'earlier today': 0,
            'this morning': 0,
            'this afternoon': 0,
            'tonight': 0,
            'overnight': 0,
        }
        
        text_lower = text.lower()
        
        for ref, days_offset in temporal_refs.items():
            if ref in text_lower:
                return {
                    'has_temporal_reference': True,
                    'temporal_reference': ref,
                    'days_offset': days_offset,
                    'is_follow_up': days_offset < 0  # Events referring to past are follow-ups
                }
        
        return {
            'has_temporal_reference': False,
            'temporal_reference': None,
            'days_offset': 0,
            'is_follow_up': False
        } 

    async def get_month_events(self, year: int, month: str) -> Dict[str, List[Dict]]:
        """
        Fetch all events for a specific month from Wikipedia.
        
        Args:
            year: Year (e.g., 2025)
            month: Month name (e.g., 'May', 'June')
            
        Returns:
            Dictionary mapping date IDs to lists of events
        """
        # Construct month URL
        month_url = f"https://en.wikipedia.org/wiki/Portal:Current_events/{month}_{year}"
        
        logger.info(f"Fetching events for {month} {year} from: {month_url}")
        
        params = {
            "api_key": settings.SCRAPEAPI_KEY,
            "url": month_url
        }
        
        try:
            response = requests.get(self.SCRAPEAPI_ENDPOINT, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all date sections in the month page
            date_sections = soup.find_all("div", class_="current-events-main vevent")
            logger.info(f"Found {len(date_sections)} date sections in {month} {year}")
            
            events_by_date = {}
            
            for section in date_sections:
                section_id = section.get("id", "")
                if section_id:
                    logger.info(f"Processing date section: {section_id}")
                    events = self._parse_date_section(section, section_id)
                    if events:
                        events_by_date[section_id] = events
            
            return events_by_date
            
        except Exception as e:
            logger.error(f"Error fetching month events for {month} {year}: {str(e)}")
            return {}
    
    def get_available_dates_from_current(self, html_content: str) -> List[str]:
        """Extract available date IDs from the current events page.
        
        Args:
            html_content: HTML content of the page
            
        Returns:
            List of date IDs found on the page
        """
        soup = BeautifulSoup(html_content, "html.parser")
        date_sections = soup.find_all("div", class_="current-events-main vevent")
        
        date_ids = []
        for section in date_sections:
            section_id = section.get("id", "")
            if section_id and re.match(r'\d{4}_\w+_\d{1,2}', section_id):
                date_ids.append(section_id)
        
        return date_ids 
