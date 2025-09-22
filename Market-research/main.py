from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import os
import logging
from io import StringIO
import threading
from datetime import datetime
import uuid

from scraper import scrape_competitors
from analyzer import ContentAnalyzer
from config import COMPETITORS, OUTPUT_CONFIG, MODEL_NAME

# Set up logging
import logging
from io import StringIO
import threading
from datetime import datetime

# Create a custom log handler to capture logs in memory
class InMemoryLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.lock = threading.Lock()
        
    def emit(self, record):
        with self.lock:
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                'level': record.levelname,
                'message': self.format(record),
                'module': record.name
            }
            self.logs.append(log_entry)
            # Keep only last 1000 log entries to prevent memory issues
            if len(self.logs) > 1000:
                self.logs = self.logs[-1000:]
    
    def get_logs(self):
        with self.lock:
            return self.logs.copy()
    
    def clear_logs(self):
        with self.lock:
            self.logs.clear()

# Create global log handler
log_handler = InMemoryLogHandler()
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Keep console logging
        log_handler  # Add our custom handler
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Research Assistant API")

# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory storage for research tasks
research_tasks = {}

class ResearchRequest(BaseModel):
    competitors: Optional[Dict[str, dict]] = None
    model_name: Optional[str] = None

class ResearchStatus(BaseModel):
    task_id: str
    status: str
    message: str

class ResearchResult(BaseModel):
    task_id: str
    status: str
    digest: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None
    scraped_content: Optional[Dict[str, str]] = None  # Add scraped content field
    scraped_urls: Optional[Dict[str, str]] = None  # Add URLs being scraped

# Add root endpoint to serve the main page
@app.get("/")
async def root():
    return FileResponse('static/index.html')

# Add health endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Market Research Assistant API"}

# Add logs endpoint for frontend
@app.get("/logs")
async def get_logs():
    """Get backend logs with detailed scraping information"""
    try:
        logs = log_handler.get_logs()
        return {
            "logs": logs,
            "total_count": len(logs),
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Error fetching logs: {str(e)}"}

@app.delete("/logs")
async def clear_logs():
    """Clear all logs"""
    try:
        log_handler.clear_logs()
        return {"message": "Logs cleared successfully"}
    except Exception as e:
        return {"error": f"Error clearing logs: {str(e)}"}

@app.get("/historical-data")
async def get_historical_data():
    """Get historical data from the JSON file"""
    try:
        historical_file_path = "historical_data.json"
        if not os.path.exists(historical_file_path):
            return {"error": "Historical data file not found", "data": []}
        
        with open(historical_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Process the data to make it more readable
        if isinstance(data, list) and len(data) > 0:
            # The data appears to be a single long string in a list
            content = data[0] if data else ""
            
            # Try to parse and structure the content
            sections = []
            
            # Split by common patterns to identify sections
            parts = content.split("Learn more")
            
            for i, part in enumerate(parts):
                if part.strip():
                    # Clean up the text
                    clean_part = part.strip()
                    if clean_part:
                        # Try to identify section titles
                        title = ""
                        content_text = clean_part
                        
                        # Look for common section patterns
                        if "Data Strategy" in clean_part:
                            title = "Data Strategy & Engineering"
                        elif "Business Intelligence" in clean_part:
                            title = "Business Intelligence & Analytics"
                        elif "Data Governance" in clean_part:
                            title = "Data Governance"
                        elif "BI Migration" in clean_part:
                            title = "BI Migration & Modernization"
                        elif "Cloud Transformation" in clean_part:
                            title = "Cloud Transformation"
                        elif "Industrial Internet" in clean_part or "IIoT" in clean_part:
                            title = "Industrial Internet of Things (IIoT)"
                        elif "Artificial Intelligence" in clean_part:
                            title = "Artificial Intelligence"
                        elif "Salesforce" in clean_part:
                            title = "Salesforce Implementation"
                        elif "Evaluation" in clean_part:
                            title = "Evaluation"
                        elif "Experience" in clean_part:
                            title = "Experience"
                        elif "Kick-Start" in clean_part:
                            title = "Kick-Start"
                        else:
                            title = f"Section {i+1}"
                        
                        sections.append({
                            "id": i,
                            "title": title,
                            "content": content_text[:500] + "..." if len(content_text) > 500 else content_text,
                            "full_content": content_text
                        })
            
            return {
                "success": True,
                "total_sections": len(sections),
                "sections": sections,
                "raw_data_length": len(content),
                "last_updated": datetime.now().isoformat()
            }
        else:
            return {
                "success": True,
                "total_sections": 0,
                "sections": [],
                "raw_data": data,
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error reading historical data: {str(e)}")
        return {"error": f"Error reading historical data: {str(e)}", "data": []}

def run_research(task_id: str, competitors: Dict[str, dict], model_name: str):
    """Run the market research process with detailed logging"""
    try:
        logger.info(f"🚀 Starting research task {task_id}")
        logger.info(f"📊 Model: {model_name}")
        logger.info(f"🎯 Competitors: {list(competitors.keys())}")
        
        research_tasks[task_id]["status"] = "running"
        research_tasks[task_id]["message"] = "Initializing scraping process..."
        
        # Store URLs being scraped
        scraped_urls = {name: config["url"] for name, config in competitors.items()}
        research_tasks[task_id]["scraped_urls"] = scraped_urls
        
        logger.info(f"🌐 URLs to scrape:")
        for name, url in scraped_urls.items():
            logger.info(f"  • {name}: {url}")
        
        research_tasks[task_id]["message"] = "Scraping competitor websites..."
        logger.info("🔍 Starting web scraping process...")
        
        # Step 1: Scrape competitor websites
        scraped_data = scrape_competitors(competitors)
        
        if not scraped_data or not any(scraped_data.values()):
            logger.error("❌ No content found on competitor websites")
            raise ValueError("No content found on competitor websites.")
        
        # Log scraping results
        logger.info("✅ Scraping completed successfully!")
        for name, content in scraped_data.items():
            if content:
                word_count = len(content.split())
                char_count = len(content)
                logger.info(f"  • {name}: {word_count} words, {char_count} characters")
            else:
                logger.warning(f"  • {name}: No content scraped")
        
        # Store scraped content in task data
        research_tasks[task_id]["scraped_content"] = scraped_data
        
        # Load historical data (if exists)
        logger.info("📚 Loading historical data...")
        historical_data = []
        if os.path.exists("historical_data.json"):
            try:
                with open("historical_data.json", "r") as f:
                    historical_data = json.load(f)
                logger.info(f"✅ Loaded {len(historical_data)} historical entries")
            except Exception as e:
                logger.error(f"❌ Error loading historical data: {e}")
        else:
            logger.info("ℹ️ No historical data found")
        
        research_tasks[task_id]["message"] = "Analyzing content with AI..."
        logger.info("🤖 Starting AI content analysis...")
        
        # Step 2: Create ContentAnalyzer instance
        analyzer = ContentAnalyzer(model_name)
        logger.info(f"✅ ContentAnalyzer initialized with model: {model_name}")
        
        # Step 3: Generate digest
        logger.info("📝 Generating market research digest...")
        digest = analyzer.generate_market_digest(
            scraped_data=scraped_data,
            historical_data=historical_data
        )
        logger.info("✅ Market digest generated successfully")
        
        # Step 4: Save digest to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_CONFIG['filename_prefix']}_{timestamp}.txt"
        
        if OUTPUT_CONFIG.get("save_to_file", True):
            directory = OUTPUT_CONFIG.get("directory", "digests")
            os.makedirs(directory, exist_ok=True)
            filepath = os.path.join(directory, filename)
            
            logger.info(f"💾 Saving digest to: {filepath}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(digest)
            
            research_tasks[task_id]["filename"] = filepath
            logger.info("✅ Digest saved successfully")
        
        # Save current data as historical for next run
        try:
            logger.info("💾 Saving data for historical analysis...")
            with open("historical_data.json", "w") as f:
                json.dump(list(scraped_data.values()), f)
            logger.info("✅ Historical data saved")
        except Exception as e:
            logger.error(f"❌ Error saving historical data: {e}")
        
        research_tasks[task_id]["status"] = "completed"
        research_tasks[task_id]["digest"] = digest
        research_tasks[task_id]["message"] = "Research completed successfully"
        
        logger.info(f"🎉 Research task {task_id} completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"💥 Error in research task {task_id}: {e}")
        research_tasks[task_id]["status"] = "failed"
        research_tasks[task_id]["error"] = str(e)
        research_tasks[task_id]["message"] = f"Research failed: {str(e)}"

@app.post("/start-research", response_model=ResearchStatus)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """Start a new market research task"""
    task_id = str(uuid.uuid4())
    
    # Use provided values or defaults from config
    competitors = request.competitors or COMPETITORS
    model_name = request.model_name or MODEL_NAME
    
    # Initialize task status
    research_tasks[task_id] = {
        "status": "pending",
        "message": "Task queued",
        "digest": None,
        "filename": None,
        "error": None,
        "scraped_content": None,
        "scraped_urls": None
    }
    
    # Add background task
    background_tasks.add_task(run_research, task_id, competitors, model_name)
    
    return ResearchStatus(
        task_id=task_id,
        status="pending",
        message="Research task started"
    )

@app.get("/research-status/{task_id}", response_model=ResearchStatus)
async def get_research_status(task_id: str):
    """Get the status of a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = research_tasks[task_id]
    return ResearchStatus(
        task_id=task_id,
        status=task["status"],
        message=task["message"]
    )

@app.get("/research-result/{task_id}", response_model=ResearchResult)
async def get_research_result(task_id: str):
    """Get the result of a research task"""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = research_tasks[task_id]
    return ResearchResult(
        task_id=task_id,
        status=task["status"],
        digest=task.get("digest"),
        filename=task.get("filename"),
        error=task.get("error"),
        scraped_content=task.get("scraped_content"),  # Include scraped content
        scraped_urls=task.get("scraped_urls")  # Include scraped URLs
    )

@app.get("/research-tasks")
async def list_research_tasks():
    """List all research tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "message": task["message"]
            }
            for task_id, task in research_tasks.items()
        ]
    }

# Add endpoint to get default competitors
@app.get("/default-competitors")
async def get_default_competitors():
    """Get the default competitors configuration"""
    return COMPETITORS

# Add startup dependency check
def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import sentencepiece
        logger.info("✅ SentencePiece library found")
    except ImportError:
        missing_deps.append("sentencepiece")
        logger.error("❌ SentencePiece library not found")
    
    try:
        from transformers import T5Tokenizer
        # Try to create a tokenizer to test if SentencePiece works with transformers
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        logger.info("✅ T5Tokenizer working correctly")
    except Exception as e:
        missing_deps.append("T5Tokenizer")
        logger.error(f"❌ T5Tokenizer failed: {e}")
    
    if missing_deps:
        error_msg = f"""
🚨 DEPENDENCY ERROR: Missing required libraries: {', '.join(missing_deps)}

🔧 TO FIX THIS ISSUE:
1. Run: python fix_sentencepiece.py
2. Or run: fix_sentencepiece.bat
3. Or manually: pip install --upgrade sentencepiece transformers

📖 For more info: https://github.com/google/sentencepiece#installation
        """
        logger.error(error_msg)
        return False, error_msg
    
    return True, "All dependencies are available"

@app.on_event("startup")
async def startup_event():
    """Check dependencies on startup"""
    logger.info("🚀 Starting Market Research Assistant...")
    deps_ok, message = check_dependencies()
    if not deps_ok:
        logger.error("⚠️ Server started with missing dependencies")
        logger.error("🌐 Web interface will be available, but research tasks will fail")
    else:
        logger.info("✅ All dependencies verified - Ready to go!")

@app.get("/dependency-check")
async def dependency_check():
    """Check if all dependencies are properly installed"""
    deps_ok, message = check_dependencies()
    return {
        "dependencies_ok": deps_ok,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/research-tasks")
async def list_research_tasks():
    """List all research tasks"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "message": task["message"]
            }
            for task_id, task in research_tasks.items()
        ]
    }

# Add endpoint to get default competitors
@app.get("/default-competitors")
async def get_default_competitors():
    """Get the default competitors configuration"""
    return COMPETITORS

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external connections
    uvicorn.run(app, host="127.0.0.1", port=8000)