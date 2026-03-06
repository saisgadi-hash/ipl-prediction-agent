"""
Module B2: LLM Integration for Match Analysis
Uses Gemini, DeepSeek, or Anthropic APIs to generate tactical insights and natural language match previews.
"""
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# We will implement a fallback mechanism: Gemini -> DeepSeek -> Anthropic (if available)
def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Call Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY safely ignored")
        
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        # Combine system and user prompt for Gemini
        combined_prompt = f"{system_prompt}\n\nTask:\n{user_prompt}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=combined_prompt,
        )
        return response.text
    except Exception as e:
        raise Exception(f"Gemini API Error: {e}")

def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    """Call DeepSeek API using openai-compatible client."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY missing")
        
    try:
        # We can use requests directly for DeepSeek since it's simple REST
        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        raise Exception(f"DeepSeek API Error: {e}")

def call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Call Anthropic (Claude) API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY missing")
        
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text
    except Exception as e:
        raise Exception(f"Anthropic API Error: {e}")

# Simple caching mechanism to avoid redundant API calls
_match_analysis_cache = {}

def get_llm_match_analysis(team1: str, team2: str, match_context: dict, cache_ttl: int = 3600) -> str:
    """
    Generates a natural language match preview and tactical analysis.
    Tries Gemini first, then DeepSeek, then Anthropic.
    Caches the response for `cache_ttl` seconds.
    """
    cache_key = f"{team1}_vs_{team2}"
    
    # Check cache
    if cache_key in _match_analysis_cache:
        cached_item = _match_analysis_cache[cache_key]
        if time.time() - cached_item['time'] < cache_ttl:
            return cached_item['data']
            
    system_prompt = (
        "You are an expert cricket analyst summarizing an upcoming IPL match. "
        "Keep the analysis concise, insightful, and focused on tactical matchups, recent form, and venue characteristics. "
        "Provide a short paragraph on what to watch out for."
    )
    
    # Format the context for the LLM
    context_str = "\n".join([f"- {k}: {v}" for k, v in match_context.items()])
    user_prompt = f"Analyze this upcoming match between {team1} and {team2} using the following statistical context:\n{context_str}"
    
    analysis = "Match analysis unavailable (all APIs failed or missing keys)."
    
    # Fallback Mechanism: Gemini -> DeepSeek -> Anthropic
    try:
        analysis = call_gemini(system_prompt, user_prompt)
    except Exception as e1:
        print(f"Primary LLM (Gemini) failed: {e1}. Trying DeepSeek...")
        try:
            analysis = call_deepseek(system_prompt, user_prompt)
        except Exception as e2:
            print(f"Fallback LLM (DeepSeek) failed: {e2}. Trying Anthropic...")
            try:
                analysis = call_anthropic(system_prompt, user_prompt)
            except Exception as e3:
                print(f"All LLMs failed. Final error: {e3}")
                
    # Cache the successful result (or the fallback message if all failed)
    _match_analysis_cache[cache_key] = {'data': analysis, 'time': time.time()}
    
    return analysis

if __name__ == "__main__":
    print("Testing B2 Module LLM Integration (Gemini/DeepSeek fallback) ...")
    mock_context = {
        "team1_recent_win_pct": 0.8,
        "team2_recent_win_pct": 0.4,
        "venue": "M A Chidambaram Stadium, Chennai",
        "head_to_head_team1_wins": 12,
        "head_to_head_team2_wins": 8
    }
    
    try:
        result = get_llm_match_analysis("Chennai Super Kings", "Mumbai Indians", mock_context)
        print("\n--- LLM Match Analysis ---")
        print(result)
        print("--------------------------")
    except Exception as e:
        print(f"Test failed: {e}")
