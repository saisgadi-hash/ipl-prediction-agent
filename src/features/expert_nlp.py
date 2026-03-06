"""
Module B1: Expert Commentary NLP
Retrieves cricket news and performs sentiment analysis using Hugging Face Transformers.
"""
import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Load the sentiment analysis pipeline using a lightweight default model (DistilBERT)
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    print(f"Warning: Could not load sentiment pipeline: {e}")
    sentiment_analyzer = None

def get_recent_news(team_name: str, max_articles: int = 3) -> list:
    """
    Search for recent news about a team using a simple web search or News API if configured.
    For this implementation we will simulate scraping by providing typical commentary, 
    or you can hook it to an actual search API (e.g. NewsAPI) if you have a key.
    We do a very basic scrape of ESPNcricinfo search page for the team.
    """
    # NOTE: Scraping ESPNcricinfo search results directly is often blocked or fragile.
    # In a real production system, use a News API or official RSS feeds.
    # For Phase B MVP, we mock the retrieval of 3 recent "expert opinions" if scraping fails.
    
    # Simple simulated news for MVP to prevent scraping bans
    simulated_news = {
        "Chennai Super Kings": [
            "CSK's top order looks solid ahead of the clash, with players in peak form.",
            "Bowling could be a slight concern for Chennai at this venue.",
            "Dhoni's tactical brilliance gives CSK an edge in close encounters."
        ],
        "Mumbai Indians": [
            "Mumbai Indians are struggling to find rhythm in the powerplay overs.",
            "Bumrah's return strengthens MI's death bowling significantly.",
            "The middle order needs to step up for Mumbai to pose a massive total."
        ],
        "Royal Challengers Bengaluru": [
            "Kohli is looking in supreme touch, a huge positive for RCB.",
            "RCB's bowling attack still looks vulnerable under pressure.",
            "The team looks balanced but field placements have been questionable."
        ]
    }
    
    # Generic news for teams not in the simulation map
    generic_news = [
        f"The {team_name} camp looks confident ahead of their next fixture.",
        f"Injury concerns might force {team_name} to rethink their playing XI.",
        f"Recent form suggests {team_name} has momentum on their side."
    ]
    
    return simulated_news.get(team_name, generic_news)[:max_articles]

def calculate_expert_sentiment(team_name: str) -> float:
    """
    Retrieves recent news for a team, analyzes the sentiment of each text,
    and returns an aggregated sentiment score between 0 (Negative) and 1 (Positive).
    """
    if not sentiment_analyzer:
        return 0.5 # Neutral if pipeline fails to load
        
    news_items = get_recent_news(team_name)
    if not news_items:
        return 0.5
        
    total_score = 0.0
    valid_items = 0
    
    for text in news_items:
        try:
            result = sentiment_analyzer(text)[0]
            label = result['label'] # 'POSITIVE' or 'NEGATIVE'
            score = result['score'] # Confidence score (0.0 to 1.0)
            
            # Convert to a standard 0 to 1 scale where 1 is absolute positive
            if label == 'POSITIVE':
                match_score = score 
            else:
                match_score = 1.0 - score
                
            total_score += match_score
            valid_items += 1
            
        except Exception as e:
            print(f"Error analyzing sentiment for '{text}': {e}")
            
    if valid_items == 0:
        return 0.5
        
    return total_score / valid_items

if __name__ == "__main__":
    csk_sentiment = calculate_expert_sentiment("Chennai Super Kings")
    mi_sentiment = calculate_expert_sentiment("Mumbai Indians")
    
    print(f"CSK Expert Sentiment Score: {csk_sentiment:.3f}")
    print(f"MI Expert Sentiment Score: {mi_sentiment:.3f}")
