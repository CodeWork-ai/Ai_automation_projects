import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import sys
import os
import logging

# Set up logging with explicit flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force all handlers to flush immediately
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        class FlushingStream:
            def __init__(self, stream):
                self.stream = stream
            def write(self, data):
                self.stream.write(data)
                self.stream.flush()
            def flush(self):
                self.stream.flush()
        handler.stream = FlushingStream(handler.stream)

def scrape_company(url, content_selectors, exclude_selectors, company_name):
    """Scrape a company website using dynamic scraping"""
    logger.info(f"🏢 Starting scrape for {company_name}")
    logger.info(f"🔗 URL: {url}")
    text = scrape_dynamic(url, content_selectors, exclude_selectors, company_name)
    return text

def scrape_dynamic(url, content_selectors, exclude_selectors, company_name):
    """Optimized dynamic scraping with better content extraction"""
    logger.info(f"🌐 Initializing browser for {company_name}...")
    
    # Set up Chrome options
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    logger.info(f"🚀 Launching browser and navigating to {url}...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    driver.get(url)
    logger.info(f"⏳ Waiting for page to load...")
    time.sleep(3)
    
    logger.info(f"📄 Page loaded. Title: {driver.title}")
    logger.info(f"🔍 Extracting content using selectors...")
    
    # Set up Chrome options
    options = Options()
    options.headless = False  # Set to True for production
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    # Initialize WebDriver
    logger.info(f"   → Initializing Chrome WebDriver...")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        logger.info(f"   → Navigating to URL: {url}")
        driver.get(url)
        
        # Wait for page to load
        logger.info(f"   → Waiting for page elements to load...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        logger.info(f"   → Page elements loaded successfully")
        
        # Strategic scrolling for lazy loading
        logger.info(f"   → Performing strategic scrolling...")
        time.sleep(3)  # Initial wait for page to stabilize
        
        # Get page height
        page_height = driver.execute_script("return document.body.scrollHeight")
        logger.info(f"     → Page height: {page_height}px")
        
        # Scroll to bottom
        logger.info(f"     → Scrolling to bottom...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        
        # Scroll back to top
        logger.info(f"     → Scrolling back to top...")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        
        # Click on key expandable elements
        logger.info(f"     → Clicking key expandable elements...")
        expand_selectors = [
            ".read-more", ".expand", ".more", ".show-more",
            ".accordion-trigger", ".tab-trigger", ".toggle"
        ]
        
        for selector in expand_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                logger.info(f"       - Found {len(elements)} elements with '{selector}'")
                for element in elements[:3]:  # Limit to first 3 elements
                    try:
                        driver.execute_script("arguments[0].click();", element)
                        time.sleep(1)
                    except Exception as e:
                        logger.debug(f"         - Click failed: {str(e)}")
            except Exception as e:
                logger.debug(f"       - Selector error: {str(e)}")
        
        # Special handling for GWC Data AI
        if "GWC" in company_name:
            logger.info(f"   → Applying GWC Data AI specific optimizations...")

            # Wait for key content containers
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "main, .main, #main, section, article"))
                )
                logger.info("     → GWC main containers detected")
            except Exception as e:
                logger.warning(f"     → Could not detect main containers: {str(e)}")

            # Extra incremental scrolling to trigger lazy load
            logger.info(f"     → Performing incremental scrolling for GWC...")
            scroll_height = driver.execute_script("return document.body.scrollHeight")
            for i in range(1, 6):
                driver.execute_script(f"window.scrollTo(0, {scroll_height * i // 6});")
                time.sleep(2)

            # Expand GWC-specific sections
            gwc_selectors = [
                ".accordion", ".accordion-item", ".accordion-trigger",
                ".tab", ".tab-link", ".tab-trigger",
                ".read-more", ".show-more", ".expand"
            ]
            for selector in gwc_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.info(f"       - Found {len(elements)} expandable elements with '{selector}'")
                    for element in elements[:5]:  # limit to 5 clicks
                        try:
                            driver.execute_script("arguments[0].scrollIntoView();", element)
                            time.sleep(1)
                            driver.execute_script("arguments[0].click();", element)
                            time.sleep(2)
                        except Exception as e:
                            logger.debug(f"         - Click failed: {str(e)}")
                except Exception as e:
                    logger.debug(f"       - Selector error: {str(e)}")

            # Final wait to let content load
            logger.info(f"     → Waiting for final GWC content to load...")
            time.sleep(4)
        
        # Special handling for CodeWork AI (untouched)
        if "CodeWork" in company_name:
            logger.info(f"   → Applying CodeWork AI specific optimizations...")
            
            # Wait for dynamic content
            logger.info(f"     → Waiting for dynamic content...")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "main, .main, #main, .content"))
            )
            
            # Additional scrolling
            logger.info(f"     → Performing additional scrolling...")
            for i in range(1, 4):
                scroll_position = page_height * i // 3
                driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(2)
            
            # Click on CodeWork specific elements
            logger.info(f"     → Clicking CodeWork AI specific elements...")
            codework_selectors = [
                ".service", ".feature", ".solution",
                ".ai-service", ".automation", ".intelligence",
                "[class*='ai']", "[class*='automation']"
            ]
            
            for selector in codework_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.info(f"       - Found {len(elements)} elements with '{selector}'")
                    for element in elements[:2]:  # Limit to first 2 elements
                        try:
                            driver.execute_script("arguments[0].scrollIntoView();", element)
                            time.sleep(1)
                            driver.execute_script("arguments[0].click();", element)
                            time.sleep(1)
                        except Exception as e:
                            logger.debug(f"         - Interaction failed: {str(e)}")
                except Exception as e:
                    logger.debug(f"       - Selector error: {str(e)}")
            
            # Final wait for content
            logger.info(f"     → Final wait for content...")
            time.sleep(3)
        
        # Final scroll to ensure all content is loaded
        logger.info(f"   → Final scroll to ensure all content is loaded...")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
        
        # Get page source and parse with BeautifulSoup
        logger.info(f"   → Getting page source...")
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        logger.info(f"   → Page source obtained. Length: {len(page_source)} characters")
        
        # Remove unwanted elements
        logger.info(f"   → Removing unwanted elements...")
        for selector in exclude_selectors:
            elements = soup.select(selector)
            logger.info(f"     - Removed {len(elements)} elements with '{selector}'")
            for element in elements:
                element.decompose()
        
        # Extract main content with improved logic
        logger.info(f"   → Extracting main content...")
        text = ""
        
        # Try to get content from main areas first
        main_selectors = ["main", ".main", "#main", "article", ".content", ".content-wrapper"]
        for selector in main_selectors:
            content = soup.select_one(selector)
            if content:
                content_text = content.get_text(strip=True, separator=' ')
                if len(content_text) > 500:
                    text = content_text
                    logger.info(f"     - Found main content with '{selector}': {len(text)} chars")
                    break
        
        # If no main content, extract from all meaningful elements
        if not text:
            logger.info(f"     - No main content found. Extracting from all elements...")
            content_elements = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'article', 'section'])
            text_parts = []
            
            for elem in content_elements:
                elem_text = elem.get_text(strip=True)
                if 50 < len(elem_text) < 1000:  # Filter out very short/long elements
                    text_parts.append(elem_text)
            
            text = ' '.join(text_parts)
            logger.info(f"     - Extracted from elements: {len(text)} chars")
        
        # If still no substantial content, get all text
        if len(text) < 200:
            logger.info(f"     - Insufficient content. Getting all text...")
            text = soup.get_text(strip=True)
            logger.info(f"     - Extracted all text: {len(text)} chars")
        
        logger.info(f"✓ Dynamic scraping successful for {company_name}. Extracted {len(text)} characters.")
        logger.info(f"   → Content preview: {text[:200]}...")
        return text
        
    except Exception as e:
        logger.error(f"✗ Error in dynamic scraping for {company_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""
    finally:
        # Keep browser open for a few seconds so you can see the result
        logger.info(f"   → Keeping browser open for 5 seconds...")
        time.sleep(5)
        
        try:
            logger.info(f"   → Closing WebDriver...")
            driver.quit()
            logger.info(f"   → WebDriver closed successfully")
        except:
            logger.error(f"✗ Error closing WebDriver")

def scrape_competitors(competitors):
    """Scrape all competitor websites with detailed logging"""
    logger.info("🎯 Starting competitor scraping process...")
    logger.info(f"📋 Total competitors to scrape: {len(competitors)}")
    
    scraped_data = {}
    
    for i, (company_name, config) in enumerate(competitors.items(), 1):
        logger.info(f"📍 Progress: {i}/{len(competitors)} - Processing {company_name}")
        
        try:
            content = scrape_company(
                config["url"],
                config.get("content_selectors", []),
                config.get("exclude_selectors", []),
                company_name
            )
            scraped_data[company_name] = content
            
            if content:
                logger.info(f"✅ {company_name}: Scraping successful")
            else:
                logger.warning(f"⚠️ {company_name}: No content found")
                
        except Exception as e:
            logger.error(f"💥 {company_name}: Scraping failed - {str(e)}")
            scraped_data[company_name] = ""
    
    successful_scrapes = sum(1 for content in scraped_data.values() if content)
    logger.info(f"📊 Scraping summary: {successful_scrapes}/{len(competitors)} successful")
    
    return scraped_data
    
    if results[name]:
        logger.info(f"✓ Successfully scraped {name}. Content length: {len(results[name])} characters")
    else:
        logger.info(f"✗ Failed to scrape {name}")
        
        logger.info(f"Completed scraping {name}.")
        # Force flush after each competitor
        for handler in logging.getLogger().handlers:
            handler.flush()
    
    logger.info(f"{'*'*70}")
    logger.info(f"SCRAPING PROCESS COMPLETED")
    logger.info(f"{'*'*70}")
    
    return results
