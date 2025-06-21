# Wikipedia Current Events Extractor

## Overview

The Wikipedia Current Events extractor is a data extraction system that fetches and parses event data from Wikipedia's "Current events" portal. It uses **ScrapeAPI** for reliable web scraping to fetch Wikipedia pages, then applies custom parsing logic to convert HTML content into structured JSON events. The system processes real-world events from Wikipedia's carefully curated and categorized event listings, making them available for semantic analysis and phylogenetic processing in ChronoLedge.

## Features

### 🔄 **Multiple Extraction Modes**
- **Current Events**: Extract today's events and catch up on recent unprocessed dates
- **Specific Date**: Extract events from a particular date (format: `YYYY_Month_DD`)
- **Month Extraction**: Bulk extract an entire month of events

### 📊 **Data Processing**
- **Structured Parsing**: Converts Wikipedia markup into clean, structured JSON
- **Category Detection**: Automatically identifies event categories (domains) (Politics, Sports, Science, etc.)
- **Temporal Analysis**: Extracts and processes temporal references within events
- **Duplicate Prevention**: Tracks extracted dates to avoid reprocessing

### 🌐 **Web Scraping Architecture (using ScrapeAPI)**
- **ScrapeAPI Integration**: Uses ScrapeAPI service for reliable Wikipedia page fetching
- **Robust HTTP Handling**: Built-in retry logic and error handling through ScrapeAPI
- **Anti-bot Detection**: ScrapeAPI handles bot detection and IP rotation automatically
- **Custom HTML Parsing**: In-house parsing logic converts raw HTML to structured data

## How It Works

### 1. **Page Fetching with ScrapeAPI**
The extractor uses ScrapeAPI to fetch Wikipedia's "Current events" portal pages:
- **Portal URL Pattern**: `https://en.wikipedia.org/wiki/Portal:Current_events/[Date]`
- **Date Format**: Uses Wikipedia's specific date format (e.g., "2025_June_16")
- **ScrapeAPI Request**: Sends URL to ScrapeAPI service for reliable page retrieval
- **HTML Response**: Receives raw HTML content from Wikipedia pages

### 2. **Custom HTML Parsing Logic**

#### **Two-Stage Process**
```
ScrapeAPI Fetch → Raw HTML → Custom Parser → Structured JSON Events
```

#### **HTML Content Analysis**
- **DOM Parsing**: Processes HTML structure to identify content sections
- **Event Category Detection**: Identifies event categories (domains) from HTML structure
- **Content Block Extraction**: Isolates individual event descriptions from Wikipedia markup

#### **Text Processing & Structuring**
- **Wikipedia Markup Removal**: Strips Wikipedia-specific formatting and links
- **Content Cleaning**: Removes HTML tags and standardizes text encoding
- **Event Structuring**: Converts parsed content into structured JSON objects
- **Category Assignment**: Maps Wikipedia sections to semantic domains

### 3. **Temporal Information Processing**
- **Date Extraction**: Identifies explicit dates within event text
- **Temporal References**: Processes relative time expressions ("yesterday", "this week")
- **Event Dating**: Associates events with their occurrence dates
- **Chronological Ordering**: Maintains temporal sequence for phylogenetic analysis

### 4. **Data Structuring**
Each extracted event becomes a structured JSON object:
```json
{
  "id": "unique_event_identifier",
  "summary": "Clean event description",
  "category": "Primary category",
  "full_category": "Detailed category path",
  "temporal_info": {
    "temporal_reference": "extracted_date_info",
    "confidence": "high|medium|low"
  },
  "source_url": "wikipedia_page_url",
  "extraction_date": "when_extracted",
  "metadata": {
    "section": "wikipedia_section",
    "processing_notes": "any_special_handling"
  }
}
```

## Usage Examples

### Current Events Extraction
```python
# Extract today's events and recent unprocessed dates
response = await get_wikimedia_current_events()
```

### Specific Date Extraction
```python
# Extract events from June 16, 2025
response = await get_wikimedia_current_events(date="2025_June_16")
```

### Month Extraction
```python
# Extract all events from June 2025
response = await extract_month_events(year=2025, month="June")
```

## API Endpoints

### `/api/etl/wikimedia/current-events`
- **Method**: GET
- **Parameters**: 
  - `date` (optional): Specific date in format `YYYY_Month_DD`
- **Returns**: Events for today or specified date

### `/api/etl/wikimedia/month-events`
- **Method**: GET
- **Parameters**:
  - `year`: Year (e.g., 2025)
  - `month`: Month name (e.g., "June")
- **Returns**: All events for the specified month

### `/api/etl/extraction-status`
- **Method**: GET
- **Returns**: Summary of extraction history and status

## File Organization

### Output Structure
```
data/
├── events/
│   ├── 2025_June_14.json    # Daily event files
│   ├── 2025_June_15.json
│   └── ...
├── event_metadata.jsonl     # Processing metadata
└── extraction_metadata.json # Extraction tracking
```

### Metadata Tracking
- **Extraction History**: Tracks which dates have been processed
- **Processing Stats**: Event counts, categories, success rates
- **Error Logging**: Failed extractions and retry information

## Technical Implementation

### Core Components

#### **WikimediaExtractor Class** (`wikimedia.py`)
- **ScrapeAPI Integration**: Handles requests to ScrapeAPI service
- **HTML Processing**: Custom parsing logic for Wikipedia page structure
- **Content Extraction**: Converts HTML content blocks to structured events
- **Text Cleaning**: Removes Wikipedia markup and formatting

#### **DataManager Class** (`data_manager.py`)
- File management and storage
- Metadata tracking
- Duplicate detection

### Two-Layer Architecture
1. **Fetching Layer**: ScrapeAPI handles web scraping challenges (bot detection, IP rotation, etc.)
2. **Parsing Layer**: Custom logic processes HTML content into structured JSON events

### Error Handling
- **ScrapeAPI Errors**: Handles service errors and rate limiting from ScrapeAPI
- **HTML Parsing Errors**: Graceful degradation when Wikipedia structure changes
- **Data Validation**: Content quality checks before saving structured events

### Performance Considerations
- **Caching**: Avoids re-extracting existing dates
- **Batch Processing**: Efficient month-wide extractions
- **Memory Management**: Processes events incrementally

## Integration with ChronoLedge

### Phylogenetic Pipeline
The extracted events feed directly into ChronoLedge's semantic phylogeny system:

1. **Chronological Processing**: Events are processed in temporal order
2. **Domain Clustering**: Events are grouped by semantic domains
3. **AEZ Formation**: Similar events form Anchored Embedding Zones
4. **Evolution Tracking**: Semantic divergence creates phylogenetic trees

### Data Quality
- **Structured Output**: Consistent JSON format for downstream processing
- **Rich Metadata**: Temporal and categorical information preserved
- **Source Traceability**: Links back to original Wikipedia content

## Configuration

### Extraction Settings
- **Date Ranges**: Configurable start/end dates for bulk processing
- **Category Filters**: Option to focus on specific event types
- **Quality Thresholds**: Minimum content requirements

### Storage Options
- **File Format**: JSON with optional compression
- **Naming Convention**: Date-based file organization
- **Backup Strategy**: Configurable data retention

## Monitoring and Maintenance

### Health Checks
- **Extraction Success Rates**: Monitor processing efficiency
- **Data Quality Metrics**: Track content completeness
- **Source Availability**: Verify Wikipedia accessibility

### Troubleshooting
- **Missing Dates**: Check Wikipedia availability for specific dates
- **ScrapeAPI Issues**: Monitor ScrapeAPI service status and quota usage
- **Format Changes**: Wikipedia occasionally updates portal structure requiring parser updates
- **HTML Parsing Failures**: Custom parser may need updates if Wikipedia structure changes

## Best Practices

### Ethical Data Extraction
- **ScrapeAPI Service**: Uses professional scraping service that handles rate limiting
- **Wikipedia Compliance**: ScrapeAPI manages respectful request patterns
- **Service Terms**: Follows both ScrapeAPI and Wikipedia usage guidelines

### Data Management
- **Regular Backups**: Protect against data loss
- **Version Control**: Track changes in extraction logic
- **Quality Assurance**: Regular validation of extracted content