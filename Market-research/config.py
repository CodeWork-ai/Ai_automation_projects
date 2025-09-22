# Competitors configuration
COMPETITORS = {
    "GWC Data AI": {
        "url": "https://gwcdata.ai/",
        "content_selectors": [
            "main", 
            ".content", 
            ".post-content", 
            ".blog-content",
            ".news-content",
            ".article-content",
            ".entry-content",
            ".main-content",
            ".page-content",
            ".site-content",
            "article",
            ".section",
            ".container",
            ".wrapper",
            ".text-block",
            ".info-block",
            ".description",
            ".about-section",
            ".services-section",
            ".partnership-section",
            ".hero-section",
            ".feature-section",
            ".solution-section"
        ],
        "exclude_selectors": [
            "nav", 
            "footer", 
            ".sidebar", 
            ".ads",
            ".social-links",
            ".menu",
            ".navigation",
            ".header",
            ".footer",
            ".copyright",
            ".comments",
            ".related",
            ".share",
            ".newsletter",
            ".popup",
            ".modal",
            ".cookie-notice",
            ".form",
            ".button",
            ".icon",
            ".meta",
            ".tags",
            ".breadcrumb"
        ]
    },
    "CodeWork AI": {
        "url": "https://codework.ai/",
        "content_selectors": [
            "main", 
            ".content", 
            ".post-content", 
            ".blog-content",
            ".news-content",
            ".article-content",
            ".entry-content",
            ".main-content",
            ".page-content",
            ".site-content",
            "article",
            ".section",
            ".container",
            ".wrapper",
            ".hero",
            ".about",
            ".services",
            ".text-block",
            ".info-block",
            ".description",
            ".feature-section"
        ],
        "exclude_selectors": [
            "nav", 
            "footer", 
            ".sidebar", 
            ".ads",
            ".social-links",
            ".menu",
            ".navigation",
            ".header",
            ".footer",
            ".copyright",
            ".comments",
            ".related",
            ".share",
            ".newsletter",
            ".popup",
            ".modal",
            ".cookie-notice",
            ".form",
            ".button",
            ".icon",
            ".meta",
            ".tags",
            ".breadcrumb"
        ]
    }
}

# File output settings
OUTPUT_CONFIG = {
    "save_to_file": True,
    "directory": "digests",
    "filename_prefix": "market_research_digest"
}

# Single model for all tasks
MODEL_NAME = "google/flan-t5-base"

# Analysis settings
CATEGORIES = ["Product Launch", "Pricing Change", "Partnership", "Marketing", "Research Update", "AI Innovation"]
TREND_THRESHOLD = 0.8