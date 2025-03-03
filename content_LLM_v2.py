import os
import json
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import requests
import time
from dotenv import load_dotenv
from typing import Tuple, Dict, List, Union
from dateutil.relativedelta import relativedelta
import re
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

# Load environment variables
load_dotenv()
# Define required API keys and their environment variable names
REQUIRED_ENV_VARS = {
    'OPENAI_API_KEY': None,
    'CLAUDE_API_KEY': None,
    'DEEPSEEK_API_KEY': None,
    'GEMINI_API_KEY': None,
    'DEEPSEEK_API_URL': None
}


# Define available models and their configurations
AVAILABLE_MODELS = {
    "GPT-4 Latest": {
        "id": "gpt-4-0125-preview",
        "provider": "openai",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "max_tokens": 4096,
        "supports_streaming": True,
        "requires_key": "OPENAI_API_KEY"
    },
    "Claude 3.5 Sonnet": {
        "id": "claude-3-sonnet-20240229",
        "provider": "anthropic",
        "api_url": "https://api.anthropic.com/v1/messages",
        "max_tokens": 4096,
        "supports_streaming": True,
        "requires_key": "CLAUDE_API_KEY"
    },
    "DeepSeek Chat v3": {
        "id": "deepseek-chat",
        "provider": "deepseek",
        "api_url": None,  # Will be filled from env var
        "max_tokens": 4096,
        "supports_streaming": True,
        "requires_key": "DEEPSEEK_API_KEY"
    },
    "DeepSeek Reasoner": {
        "id": "deepseek-reasoner",
        "provider": "deepseek",
        "api_url": None,  # Will be filled from env var
        "max_tokens": 4096,
        "supports_streaming": True,
        "requires_key": "DEEPSEEK_API_KEY"
    },
    "Gemini 1.5 Pro": {
        "id": "gemini-1.5-pro",
        "provider": "google",
        "api_url": "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro",
        "max_tokens": 4096,
        "supports_streaming": True,
        "requires_key": "GEMINI_API_KEY"
    }
}


def load_api_keys():
    """Load and validate required API keys from environment variables."""
    missing_vars = []
    
    for var_name in REQUIRED_ENV_VARS:
        value = os.getenv(var_name)
        if value:
            REQUIRED_ENV_VARS[var_name] = value
        else:
            missing_vars.append(var_name)
    
    return missing_vars

def check_api_keys():
    """Check API keys availability and display status in sidebar."""
    missing_vars = load_api_keys()
    
    st.sidebar.subheader("🔑 API Status")
    
    if missing_vars:
        st.sidebar.error("Missing API Keys:")
        for var in missing_vars:
            st.sidebar.warning(f"❌ {var} not found")
        return False
    
    st.sidebar.success("✅ All API keys loaded")
    return True

def get_available_models():
    """Get list of available models based on API keys"""
    # Load API keys first
    load_api_keys()
    
    available_models = {}
    
    # Check each model's requirements
    for model_name, model_info in AVAILABLE_MODELS.items():
        required_key = model_info['requires_key']
        
        # Special handling for DeepSeek models which need URL too
        if model_info['provider'] == 'deepseek':
            if (REQUIRED_ENV_VARS['DEEPSEEK_API_KEY'] and 
                REQUIRED_ENV_VARS['DEEPSEEK_API_URL']):
                # Update the API URL from env var
                model_info = model_info.copy()  # Make a copy to avoid modifying original
                model_info['api_url'] = REQUIRED_ENV_VARS['DEEPSEEK_API_URL']
                available_models[model_name] = model_info
        # For other models, just check their required key
        elif REQUIRED_ENV_VARS[required_key]:
            available_models[model_name] = model_info
    
    return available_models


if 'current_api' not in st.session_state:
    st.session_state['current_api'] = None

if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None

if 'available_models' not in st.session_state:
    st.session_state['available_models'] = {}

if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = None

if 'translated_content' not in st.session_state:
    st.session_state['translated_content'] = None

# Current date for context
CURRENT_DATE = datetime.now()
CURRENT_YEAR = CURRENT_DATE.year
CURRENT_MONTH = CURRENT_DATE.strftime('%B')
DATE_CONTEXT = f"{CURRENT_MONTH} {CURRENT_YEAR}"

# Added date validation patterns
DATE_PATTERNS = [
    r'\b\d{4}\b',  # Year
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',  # Month Year
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'  # DD/MM/YY or DD/MM/YYYY
]

    
# Supported languages remain the same
SUPPORTED_LANGUAGES = {
    'zh-CN': 'Chinese Simplified',
    'zh-TW': 'Chinese Traditional',
    'ar': 'Arabic',
    'vi': 'Vietnamese',
    'th': 'Thai'
}

def validate_dates(content: str) -> bool:
    """
    Validate that dates in content are current or recent
    Returns True if dates are valid, False otherwise
    """
    for pattern in DATE_PATTERNS:
        dates = re.findall(pattern, content)
        for date_str in dates:
            try:
                if '/' in date_str:
                    date = datetime.strptime(date_str, '%m/%d/%y' if len(date_str) < 10 else '%m/%d/%Y')
                else:
                    date = datetime.strptime(date_str, '%B %Y' if len(date_str) > 6 else '%Y')
                
                # Check if date is within acceptable range (last 3 months to current)
                if date < CURRENT_DATE - timedelta(days=90) or date > CURRENT_DATE:
                    return False
            except ValueError:
                continue
    return True


def get_major_market_movers() -> list:
    """Get the day's biggest market-moving news across different categories."""
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Market movement keywords with current context
    market_terms = [
        f'(surge OR plunge OR rally OR tumble) AND date > {current_date}',
        f'(bullish OR bearish OR volatility) AND date > {current_date}',
        f'(new highs OR new lows OR record) AND date > {current_date}',
        f'(outlook OR forecast OR guidance) AND date > {current_date}'
    ]
    
    # Market categories - unchanged
    categories = [
        'forex OR "currency market" OR USDCAD OR EURUSD OR GBPUSD',
        'stocks OR equity OR "stock market" OR indices',
        'commodities OR gold OR oil OR metals',
        'crypto OR bitcoin OR ethereum'
    ]
    
    # Create combined queries with date context
    queries = []
    for market in categories:
        for term in market_terms:
            queries.append(f'({market}) AND {term}')
    
    all_articles = []
    for query in queries:
        try:
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 3,
                'from': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'domains': 'reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,marketwatch.com'
            }
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params=params,
                headers={'Authorization': f'Bearer {os.getenv("NEWS_API_KEY")}'}
            )
            if response.status_code == 200:
                data = response.json()
                all_articles.extend(data.get('articles', []))
        except Exception as e:
            print(f"Error fetching market movers: {e}")
            continue
    
    return all_articles

@st.cache_data(ttl=3600)
def fetch_market_context(topic: str) -> Tuple[str, bool]:
    """Fetch relevant market data with current timestamp."""
    ticker_map = {
        "gold": "GC=F",
        "oil": "CL=F", 
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "forex": "EURUSD=X",
        "silver": "SI=F",
        "stocks": "^GSPC",
        "market": "^GSPC",
        "trading": "^VIX"
    }
    
    market_context = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    for keyword, ticker in ticker_map.items():
        if keyword in topic.lower():
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period="1mo")
                latest_data = history.tail(3)
                market_context.append(
                    f"Latest {keyword.capitalize()} data ({ticker}) as of {current_time}:\n"
                    f"{latest_data.to_string()}\n"
                )
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue
    
    return "\n".join(market_context) if market_context else "", bool(market_context)

def get_relevant_examples(topic: str, learning_context: dict) -> str:
    """Get relevant examples based on topic keywords."""
    examples = []
    keywords = set(topic.lower().split())
    
    # Add current date context
    current_context = {
        'year': str(CURRENT_YEAR),
        'month': CURRENT_MONTH,
        'prevYear': str(CURRENT_YEAR - 1)
    }
    
    for key, content in learning_context.items():
        if not isinstance(content, dict) or not content.get('is_good_example'):
            continue
            
        example_content = content.get('content', '')
        
        # Update any dates in example content
        for old_year in range(2020, CURRENT_YEAR):
            example_content = example_content.replace(str(old_year), current_context['year'])
        
        score = sum(keyword in example_content.lower() for keyword in keywords)
        if score > 0:
            examples.append((example_content, score))
    
    examples.sort(key=lambda x: x[1], reverse=True)
    return "\n".join(ex[0] for ex in examples[:2])

def verify_news_accuracy(topic: str, api_key: str) -> dict:
    """Enhanced news verification with strict date validation."""
    if not api_key:
        return {'error': 'NEWS_API_KEY not configured'}

    try:
        # Extract key terms from topic
        topic_terms = topic.lower().split()
        query_parts = []
        
        # Add market-related terms if present
        market_terms = {'forex', 'stocks', 'gold', 'oil', 'crypto', 'currency', 'commodities'}
        market_matches = [term for term in topic_terms if term in market_terms]
        if market_matches:
            query_parts.extend(market_matches)
        
        # Add company/asset specific terms
        company_terms = [term for term in topic_terms if term.isupper() and len(term) >= 2]
        if company_terms:
            query_parts.extend(company_terms)
        
        # Combine terms with date restriction
        query = ' OR '.join(query_parts) if query_parts else topic
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'from': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
            'to': current_date,
            'domains': 'reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,marketwatch.com'
        }
        
        # Get topic-specific articles
        response = requests.get(
            "https://newsapi.org/v2/everything", 
            params=params,
            headers={'Authorization': f'Bearer {api_key}'}
        )
        
        if response.status_code != 200:
            return {'error': f'API request failed with status {response.status_code}'}
            
        topic_articles = response.json().get('articles', [])
        
        # Get broader market context
        market_movers = get_major_market_movers()
        
        all_news = []
        seen_titles = set()
        
        # Process articles with date validation
        for article in topic_articles + market_movers:
            if article['title'] not in seen_titles:
                pub_date = datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d')
                if pub_date > datetime.now() - timedelta(days=3):
                    article['category'] = '📊 Topic Analysis'
                    all_news.append(article)
                    seen_titles.add(article['title'])
        
        if not all_news:
            return {'error': 'No recent articles found'}
        
        return {
            'latest_date': all_news[0]['publishedAt'],
            'key_facts': [f"{article['category']}: {article['title']} (as of {article['publishedAt'][:10]})" 
                         for article in all_news[:5]],
            'sources': {article['source']['name'] for article in all_news},
            'is_current': True,
            'descriptions': [article['description'] for article in all_news[:3]]
        }
        
    except Exception as e:
        return {'error': str(e)}
    
def load_resources() -> bool:
    """Load all required JSON resources."""
    resources = {
        'learning_context': 'learning_summaries.json',
        'style_guide': 'pps_style_guide.json',
        'content_rules': 'pps_content_rules.json'
    }

    success = True
    for key, file in resources.items():
        try:
            if key not in st.session_state:
                with open(file, 'r') as f:
                    st.session_state[key] = json.load(f)
                st.sidebar.success(f"✅ {key.replace('_', ' ').title()}")
        except Exception as e:
            st.sidebar.error(f"❌ {file} not found")
            print(f"Error loading {file}: {str(e)}")
            success = False

    return success
    
def get_date_range(months_back: int) -> str:
    """Generate a properly formatted date range string."""
    end_date = datetime.now()
    start_date = end_date - relativedelta(months=months_back)
    
    # Ensure chronological order
    if start_date > end_date:
        start_date, end_date = end_date, start_date
        
    return f"{start_date.strftime('%B %Y')} – {end_date.strftime('%B %Y')}"


def consolidate_date_references(content: str) -> str:
    """Consolidate multiple 'as of' date references into a single statement."""
    if not content:
        return content
        
    CURRENT_DATE = datetime.now()
    date_str = CURRENT_DATE.strftime('%B %d, %Y')
    
    # Count occurrences of "as of" date references
    as_of_pattern = rf'as of {date_str}'
    count = len(re.findall(as_of_pattern, content, re.IGNORECASE))
    
    if count > 1:
        # Add a single date reference at the start if multiple exist
        prefix = f"Note: All market data and prices referenced are as of {date_str}.\n\n"
        
        # Remove all other "as of" date references
        content = re.sub(as_of_pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up any double spaces or awkward punctuation from removals
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\s+([.,])', r'\1', content)
        content = re.sub(r'\(\s*\)', '', content)
        
        # Add the prefix
        content = prefix + content
    
    return content.strip()

def update_content_dates(content: str) -> str:
    """Update date references and ranges."""
    if not content:
        return content

    # First fix chronologically incorrect date ranges
    date_range_pattern = r'([A-Za-z]+\s+\d{4})\s*[-–]\s*([A-Za-z]+\s+\d{4})'
    
    def fix_date_range(match):
        try:
            date1 = datetime.strptime(match.group(1), '%B %Y')
            date2 = datetime.strptime(match.group(2), '%B %Y')
            
            # If dates are in wrong order, swap them
            if date1 > date2:
                return f"{date2.strftime('%B %Y')} – {date1.strftime('%B %Y')}"
            return match.group(0)  # Keep original if order is correct
            
        except ValueError:
            return match.group(0)  # Keep original if dates can't be parsed
    
    content = re.sub(date_range_pattern, fix_date_range, content)
    
    # Then consolidate date references
    content = consolidate_date_references(content)
    
    return content


def make_api_request(messages: List[dict], model_name: str, max_tokens: int, max_retries: int = 3) -> str:
    """Unified API request handler for all models"""
    if not messages or not model_name:
        raise ValueError("Messages and model_name are required")

    model_info = st.session_state.get('available_models', {}).get(model_name)
    if not model_info:
        raise ValueError(f"Model information not found for {model_name}")

    provider = model_info['provider']
    
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                # Format messages for Claude
                formatted_messages = []
                system_message = None
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                # Construct Claude payload
                payload = {
                    "model": model_info['id'],
                    "max_tokens": max_tokens,
                    "messages": formatted_messages
                }
                
                # Add system message if present
                if system_message:
                    payload["system"] = system_message
                
                headers = {
                    "x-api-key": REQUIRED_ENV_VARS['CLAUDE_API_KEY'],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                
                # Make request to Claude API
                response = requests.post(
                    model_info['api_url'],
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                
                # Print debug info
                print(f"Claude Request URL: {model_info['api_url']}")
                print(f"Claude Response Status: {response.status_code}")
                print(f"Claude Response Headers: {response.headers}")
                
                if response.status_code != 200:
                    error_data = response.json()
                    error_message = error_data.get('error', {}).get('message', 'Unknown error')
                    raise Exception(f"Claude API error: {error_message}")
                
                response_data = response.json()
                
                # Extract content from Claude response
                if not response_data.get('content'):
                    raise ValueError(f"No content in Claude response: {response_data}")
                
                if not isinstance(response_data['content'], list):
                    raise ValueError(f"Unexpected content format in Claude response: {response_data}")
                
                content = response_data['content'][0].get('text')
                if not content:
                    raise ValueError("Empty content in Claude response")
                
                st.session_state['current_api'] = model_name
                return content

            elif provider == "openai":
                payload = {
                    "model": model_info['id'],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "response_format": { "type": "text" }
                }
                headers = {
                    "Authorization": f"Bearer {REQUIRED_ENV_VARS['OPENAI_API_KEY']}",
                    "Content-Type": "application/json"
                }
                url = model_info['api_url']
                
                response = requests.post(url, json=payload, headers=headers, timeout=60)
                response_data = response.json()
                
                if response_data.get('choices'):
                    return response_data['choices'][0]['message']['content']
                
            elif provider == "deepseek":
                payload = {
                    "model": model_info['id'],
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                headers = {
                    "Authorization": f"Bearer {REQUIRED_ENV_VARS['DEEPSEEK_API_KEY']}",
                    "Content-Type": "application/json"
                }
                url = model_info['api_url']
                
            elif provider == "google":
                payload = {
                    "contents": [{"parts": [{"text": msg["content"]} for msg in messages]}],
                    "generation_config": {
                        "max_output_tokens": max_tokens,
                        "temperature": 0.7
                    }
                }
                headers = {"Content-Type": "application/json"}
                url = f"{model_info['api_url']}:generateContent?key={REQUIRED_ENV_VARS['GEMINI_API_KEY']}"
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Make the API request
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
            except requests.exceptions.RequestException as e:
                raise Exception(f"API request failed: {str(e)}")

            # Check if response exists and has content
            if not response or not response.content:
                raise ValueError(f"Empty response received from {provider}")

            # Parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response from {provider}: {str(e)}")

            # Check response status
            if response.status_code != 200:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"API request failed with status {response.status_code}: {error_message}")

            # Update session state with current API
            st.session_state['current_api'] = model_name

            # Extract content based on provider
            try:
                if provider == "openai":
                    if not response_data.get('choices'):
                        raise ValueError("No choices in response")
                    content = response_data['choices'][0].get('message', {}).get('content')
                    
                elif provider == "anthropic":
                    if not response_data.get('content'):
                        raise ValueError("No content in response")
                    content = response_data['content'][0].get('text')
                    
                elif provider == "deepseek":
                    if not response_data.get('choices'):
                        raise ValueError("No choices in response")
                    content = response_data['choices'][0].get('message', {}).get('content')
                    
                elif provider == "google":
                    if not response_data.get('candidates'):
                        raise ValueError("No candidates in response")
                    content = response_data['candidates'][0].get('content', {}).get('parts', [{}])[0].get('text')
                
                # Validate content
                if content is None:
                    raise ValueError(f"No content returned from {provider}")
                
                return content

            except (KeyError, IndexError) as e:
                raise ValueError(f"Error extracting content from {provider} response: {str(e)}")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Final attempt failed: {str(e)}")
            time.sleep(2 * (attempt + 1))
            continue
    
    raise Exception("Max retries reached without successful response")

def format_content(content: str, is_short_form: bool = False) -> str:
    """Format content to ensure consistent styling and line breaks."""
    if not content:
        return content
        
    # Replace any existing multiple line breaks with a placeholder
    content = re.sub(r'\n\s*\n', '__DOUBLE_BREAK__', content)
    content = re.sub(r'\n', '__SINGLE_BREAK__', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.replace('__DOUBLE_BREAK__', '\n\n')
    content = content.replace('__SINGLE_BREAK__', '\n')
    
    # Bold numbers and currencies
    numbers_pattern = r'(\d+\.?\d*%?(?:\s*(?:USD|EUR|GBP|JPY|points?|pips?|[A-Z]{3})){0,1})'
    content = re.sub(numbers_pattern, lambda m: f"**{m.group(1)}**" if '**' not in m.group(1) else m.group(1), content)
    
    # Bold dates - handling each part separately to avoid group reference errors
    months = r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    date_pattern = f"{months}\\s+\\d{{1,2}},?\\s+\\d{{4}}"
    content = re.sub(date_pattern, lambda m: f"**{m.group(0)}**" if '**' not in m.group(0) else m.group(0), content)
    
    # Ensure proper list formatting
    content = re.sub(r'(?m)^[•●]\s*', '\n• ', content)  # Handle different bullet types
    content = re.sub(r'(?m)^(\d+)\.\s*', lambda m: f"\n{m.group(1)}. ", content)
    
    if is_short_form:
        content = content.replace('\n\n', ' ')
        content = content.replace('\n', ' ')
        content = re.sub(r'\s+', ' ', content)
    
    return content.strip()

def generate_content(topic: str, market_data: str, context: str, content_type: str, max_length: int = 800, focus_area: str = "") -> str:
    """Generate content using the selected model."""
    try:
        # Determine if generating short-form content
        is_short_form = max_length <= 100

        # Current date context
        current_date_context = f"""
IMPORTANT - CURRENT DATE CONTEXT:
- Current Date: {CURRENT_DATE.strftime('%B %d, %Y')}
- All market analysis must be from this current perspective
- Any price levels must be qualified with "as of {CURRENT_DATE.strftime('%B %d, %Y')}"
- Reference only events from the past 3 months
- Ensure all timeframes are relative to {CURRENT_DATE.strftime('%B %Y')}

Format Requirements:
1. Use '**text**' for bold emphasis (e.g., **February 14, 2025**)
2. Bold all dates, numbers, and key metrics
3. Use double line breaks between paragraphs
4. Use proper bullet points (•) for lists
5. Preserve all formatting and line breaks
"""

        # Load necessary context based on content type
        if not is_short_form:
            style_guide = st.session_state['style_guide']
            content_rules = st.session_state['content_rules']
            good_examples = get_relevant_examples(topic, st.session_state['learning_context'])
        
        news_info = verify_news_accuracy(topic, os.getenv("NEWS_API_KEY"))

        # Prepare focus area emphasis
        focus_emphasis = ""
        if focus_area.strip():
            if is_short_form:
                focus_emphasis = f"Emphasize {focus_area} while maintaining clarity"
            else:
                focus_emphasis = f"""
Special Focus Requirements:
1. Emphasize {focus_area} throughout the content
2. Provide detailed analysis and insights specifically related to {focus_area}
3. Include specific examples and implications related to {focus_area}
4. Ensure at least 40% of the content focuses on {focus_area}
5. Connect other aspects of the analysis back to {focus_area} where relevant"""

        # Prepare news context
        if 'error' in news_info:
            news_context = f"Focus on current market conditions and technical analysis as of {CURRENT_DATE.strftime('%B %d, %Y')}"
        else:
            if is_short_form:
                news_context = news_info['key_facts'][0] if news_info['key_facts'] else ""
            else:
                news_context = "Current Market Environment:\n\n"
                for fact in news_info['key_facts'][:5]:
                    news_context += f"{fact}\n\n"
                if news_info.get('descriptions'):
                    news_context += "Detailed Market Impact:\n"
                    for desc in news_info['descriptions'][:2]:
                        if desc:
                            news_context += f"• {desc}\n"

        if is_short_form:
            # Special prompt for short-form content
            prompt = f"""
{current_date_context}

Generate a complete and concise {max_length}-word market update about {topic}.

Requirements:
1. MUST be exactly {max_length} words
2. MUST be a complete thought/analysis
3. MUST end with a clear market implication or actionable insight
4. Include ONE specific price level with current date
5. Keep the message focused and impactful
6. Bold all key numbers, dates, and metrics using **text** format

Format:
- Key market fact with specific data point
- Brief supporting context
- Clear market implication

Current Market Context:
{news_context}

Market Data:
{market_data if market_data else 'Use current market conditions'}

Example Format:
"**Bitcoin** breaks above **$50,000** (as of **February 14, 2025**) on ETF inflow surge. Institutional demand drives **15%** weekly gain with strong volume support. Watch **$52,000** resistance for potential breakout signal."

Focus: {focus_emphasis if focus_emphasis else 'Key market drivers and implications'}

Note: Message must be complete and end with clear takeaway."""

        elif content_type == "educational":
            prompt = f"""
{current_date_context}

Generate educational content about {topic} for traders and investors.

Learning Objectives:
1. Explain key concepts and fundamentals clearly
2. Provide practical examples and applications
3. Break down complex topics into digestible segments
4. Include relevant market examples where applicable
5. Target length: {max_length} words

Current Market Context:
{news_context}

{focus_emphasis}

Technical Concepts to Cover:
{'Focus on fundamental concepts and practical applications'}

Style Guidelines:
{json.dumps(style_guide, indent=2)}

Content Rules:
{json.dumps(content_rules, indent=2)}

Good Examples:
{good_examples}"""

        else:  # market_analysis
            prompt = f"""
{current_date_context}

Generate sophisticated market analysis about {topic}.

Current Market Context:
{news_context}

Technical Market Data:
{market_data if market_data else 'Consider broader market conditions and indicators'}

{focus_emphasis}

Content Requirements:
1. Start with the most significant market-moving events and their impact
2. Include analysis of how major global events are affecting markets
3. Consider technological developments if relevant
4. Provide specific data points and price movements where applicable
5. Target length: {max_length} words
6. Use proper paragraph breaks and formatting

Style Guidelines:
{json.dumps(style_guide, indent=2)}

Content Rules:
{json.dumps(content_rules, indent=2)}

Good Examples:
{good_examples}"""

        # Set appropriate system role
        if is_short_form:
            system_role = "You are a market analyst specializing in concise, complete market updates. Never cut off mid-sentence or leave thoughts incomplete."
        else:
            system_role = (
                "You are an experienced educator specializing in financial markets." if content_type == "educational"
                else "You are an expert financial analyst specializing in market analysis."
            )

        system_role += f"\nYou are writing this content on {CURRENT_DATE.strftime('%B %d, %Y')}. Use this as your reference point."

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]
        
        # Get selected model
        selected_model = st.session_state.get('current_model')
        model_info = AVAILABLE_MODELS[selected_model]
        
        # Adjust token limit based on content type
        max_tokens = min(max_length * (6 if is_short_form else 4), model_info['max_tokens'])
        
        with st.spinner(f"🤖 Generating with {selected_model}..."):
            try:
                content = make_api_request(
                    messages=messages,
                    model_name=selected_model,
                    max_tokens=max_tokens
                )
                
                if content:
                    # Process the content through the formatter
                    content = format_content(content, is_short_form)
                    
                    # For short content, ensure it's complete
                    if is_short_form:
                        # Split into sentences and ensure completion
                        sentences = content.split('.')
                        complete_content = []
                        word_count = 0
                        
                        for sentence in sentences:
                            if not sentence.strip():
                                continue
                            
                            sentence_words = len(sentence.split())
                            if word_count + sentence_words <= max_length:
                                complete_content.append(sentence.strip())
                                word_count += sentence_words
                            else:
                                break
                        
                        content = '. '.join(complete_content)
                        if not content.endswith('.'):
                            content += '.'
                
                return content
                
            except Exception as e:
                st.error(f"Generation error with {selected_model}: {str(e)}")
                return f"Error occurred: {str(e)}"
                
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        print(f"Detailed error: {str(e)}")
        return f"Error occurred: {str(e)}"

def translate_content(content: str, target_language: str) -> str:
    """Translate content using available LLM models."""
    if not content:
        return "No content to translate"

    # Translation prompt with guidelines
    translation_prompt = f"""
Translate the following content into {SUPPORTED_LANGUAGES[target_language]}.

Translation Requirements:
1. Preserve all trading terms and technical indicators in English:
   - Currency pairs (e.g., USD/JPY)
   - Technical indicators (RSI, MACD, etc.)
   - Chart patterns (bullish flag, etc.)
   - Numbers and percentages
   - Timeframes (1H, 4H, 1D)
   - Special tokens (<extra_id_X>)

2. Maintain formatting:
   - Keep paragraph breaks
   - Preserve bullet points
   - Keep special characters
   - Maintain bold/italic markers (**)

3. Translate naturally while keeping financial terms accurate.

Content to translate:
{content}

Please provide a natural {SUPPORTED_LANGUAGES[target_language]} translation that preserves all technical terms and formatting."""

    # Set up messages for the API request
    messages = [
        {
            "role": "system",
            "content": f"You are an expert financial translator specializing in {SUPPORTED_LANGUAGES[target_language]}."
        },
        {
            "role": "user",
            "content": translation_prompt
        }
    ]

    try:
        # Use the current selected model
        selected_model = st.session_state.get('current_model')
        if not selected_model:
            raise ValueError("No model selected for translation")

        model_info = AVAILABLE_MODELS[selected_model]
        max_tokens = min(len(content) * 4, model_info['max_tokens'])

        # Use the same API request function as content generation
        with st.spinner(f"Translating to {SUPPORTED_LANGUAGES[target_language]}..."):
            translated_content = make_api_request(
                messages=messages,
                model_name=selected_model,
                max_tokens=max_tokens
            )

            if translated_content:
                # Clean up any doubled spaces but preserve special tokens
                translated_content = re.sub(r'(?<!extra_id_)\s+', ' ', translated_content)
                return translated_content.strip()
            else:
                raise ValueError("Empty translation received")

    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Translation failed: {str(e)}"

def main():
    """Main application function."""
    st.title("Market Content Generator")

    # Check API keys first
    if not check_api_keys():
        st.error("⚠️ Some required API keys are missing. Please check your .env file.")
        st.stop()

    if not load_resources():
        st.error("Missing required files")
        return

    # Get available models
    available_models = get_available_models()

    # Model selection in sidebar
    if available_models:
        st.sidebar.subheader("🤖 Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose Language Model:",
            options=list(available_models.keys()),
            help="Select which AI model to use for content generation"
        )
        
        # Show model info
        model_info = available_models[selected_model]
        st.sidebar.info(f"""
        Selected Model: {selected_model}
        Provider: {model_info['provider'].title()}
        Max Tokens: {model_info['max_tokens']}
        """)
        
        # Store model information in session state
        st.session_state['current_model'] = selected_model
        st.session_state['available_models'] = available_models
    else:
        st.error("No models available. Please check your API keys.")
        st.stop()

    # Create tabs for content generation and translation
    tab1, tab2 = st.tabs(["📝 Generate Content", "🌐 Translate"])

    # Content Generation Tab

        
    with tab1:
        content_type = st.radio(
            "Select Content Type:",
            ["Market Analysis", "Educational Content"],
            help="Choose the type of content you want to generate"
        )

        col1, col2 = st.columns(2)

        with col1:
            topic = st.text_input(
                "Topic:",
                placeholder="Enter market analysis topic...",
                help="Enter any market-related topic for analysis"
            )

        with col2:
            focus_area = st.text_input(
                "Focus Area (Optional):",
                placeholder="E.g., technical analysis, fundamentals, risks...",
                help="Specify what aspects to emphasize in the content"
            )

        content_type = "market_analysis" if content_type == "Market Analysis" else "educational"

        # Add toggle for short-form content
        is_short_form = st.checkbox(
            "Enable short-form content (tweets/captions)",
            value=False,
            help="Toggle for tweet-length content generation"
        )

        # Set slider parameters based on content type
        if is_short_form:
            max_length = st.slider(
                "Word Count",
                min_value=20,
                max_value=100,
                value=30,
                step=5,
                help="20-50 words recommended for tweets, up to 100 for captions"
            )
            if max_length <= 50:
                st.info("📱 Generating social media content (tweet/caption length)")
        else:
            max_length = st.slider(
                "Word Count",
                min_value=300,
                max_value=2000,
                value=800,
                step=100,
                help="300+ words recommended for full articles"
            )

        if st.button("Generate Content", type="primary") and topic.strip():
            with st.spinner("🔄 Analyzing markets and preparing content..."):
                context = get_relevant_examples(topic, st.session_state['learning_context'])
                market_data, has_market_data = fetch_market_context(topic)

                content = generate_content(
                    topic=topic,
                    market_data=market_data,
                    context=context,
                    content_type=content_type,
                    max_length=max_length,
                    focus_area=focus_area
                )

                if content:
                    # Validate and update dates in content
                    content = update_content_dates(content)

                    # Validate market data currency
                    market_data_current = bool(market_data and
                                         CURRENT_DATE.strftime('%Y-%m-%d') in market_data)

                    if not market_data_current and has_market_data:
                        st.warning("⚠️ Market data may not be current. Please verify the latest figures.")

                    # Store in session state
                    st.session_state['generated_content'] = content

                    # Display which API was used
                    api_used = st.session_state.get('current_api', 'Unknown')
                    st.info(f"📢 Content generated using {api_used} API")

                    # Display the content
                    st.markdown("### Generated Content:")
                    st.markdown(content)

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        file_type = "tweet" if max_length <= 50 else "market_analysis"
                        st.download_button(
                            "📥 Download Content",
                            data=content,
                            file_name=f"{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.info("Switch to the Translation tab to translate this content")

    # Translation Tab
    
    with tab2:
        st.subheader("Content Translation")

        # Option to paste custom content
        use_custom = st.checkbox(
            "Use custom content",
            help="Check this to translate your own content instead of previously generated content"
        )

        if use_custom:
            content_to_translate = st.text_area(
                "Enter content to translate:",
                height=200,
                placeholder="Paste your content here..."
            )
        else:
            content_to_translate = st.session_state.get('generated_content', None)
            if content_to_translate and isinstance(content_to_translate, str):
                st.markdown("**Content to Translate:**")
                st.markdown(content_to_translate)
            else:
                st.warning("🔍 No content available. Please generate content first or use custom content.")
                content_to_translate = ""

        # Language selection
        target_language = st.selectbox(
            "Select target language:",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda x: SUPPORTED_LANGUAGES[x]
        )

        if st.button("Translate", type="primary") and content_to_translate:
            try:
                translated_content = translate_content(content_to_translate, target_language)

                if translated_content and isinstance(translated_content, str):
                    # Store translation in session state
                    st.session_state['translated_content'] = translated_content

                    # Display translation
                    st.markdown(f"### {SUPPORTED_LANGUAGES[target_language]} Translation:")
                    st.markdown(translated_content)

                    # Download option
                    download_translation = translated_content.encode('utf-8')
                    st.download_button(
                        "📥 Download Translation",
                        data=download_translation,
                        file_name=f"translated_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Translation error: {str(e)}")
                
        elif st.button("Translate"):
            st.error("⚠️ Please provide content to translate.")
            
            
if __name__ == "__main__":
    main()