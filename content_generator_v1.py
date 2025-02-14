import os
import json
import streamlit as st
from datetime import datetime
import yfinance as yf
import requests
import time
from dotenv import load_dotenv
from typing import Tuple, Dict, List, Union

# Load environment variables
load_dotenv()
deepseek_api = os.getenv("DEEPSEEK_API_KEY")
deepseek_url = os.getenv("DEEPSEEK_API_URL")
openai_api = os.getenv("OPENAI_API_KEY")
openai_url = "https://api.openai.com/v1/chat/completions"


if 'current_api' not in st.session_state:
    st.session_state['current_api'] = None

if 'generated_content' not in st.session_state:
    st.session_state['generated_content'] = None

if 'translated_content' not in st.session_state:
    st.session_state['translated_content'] = None

# Keep only one definition of SUPPORTED_LANGUAGES
SUPPORTED_LANGUAGES = {
    'zh-CN': 'Chinese Simplified',
    'zh-TW': 'Chinese Traditional',
    'ar': 'Arabic',
    'vi': 'Vietnamese',
    'th': 'Thai'
}


def load_resources():
    """Load all required JSON resources."""
    resources = {
        'learning_context': 'learning_summaries.json',
        'style_guide': 'pps_style_guide.json', 
        'content_rules': 'pps_content_rules.json'
    }
    return all(load_resource(key, file) for key, file in resources.items())

def load_resource(key: str, file: str) -> bool:
    """Load a single resource file into session state."""
    try:
        if key not in st.session_state:
            with open(file, 'r') as f:
                st.session_state[key] = json.load(f)
            st.sidebar.success(f"✅ {key.replace('_', ' ').title()}")
        return True
    except Exception as e:
        st.sidebar.error(f"❌ {file} not found")
        return False

def determine_content_type(topic: str) -> str:
    """Determine the type of content based on topic keywords."""
    market_keywords = {'trading', 'market', 'forex', 'stock', 'gold', 'crypto'}
    educational_keywords = {'guide', 'learn', 'how to', 'basics', 'introduction'}
    
    topic_lower = topic.lower()
    if any(keyword in topic_lower for keyword in market_keywords):
        return "market_analysis"
    elif any(keyword in topic_lower for keyword in educational_keywords):
        return "educational"
    return "general_content"

def get_major_market_movers() -> list:
    """Get the day's biggest market-moving news across different categories."""
    # Market movement keywords
    market_terms = [
        '(surge OR plunge OR rally OR tumble OR soar OR slump)',
        '(bullish OR bearish OR volatility)',
        '(new highs OR new lows OR record high OR record low)',
        '(outlook OR forecast OR guidance)'
    ]
    
    # Market categories
    categories = [
        'forex OR "currency market" OR USDCAD OR EURUSD OR GBPUSD',
        'stocks OR equity OR "stock market" OR indices',
        'commodities OR gold OR oil OR metals',
        'crypto OR bitcoin OR ethereum'
    ]
    
    # Create combined queries
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
        except:
            continue
    
    return all_articles

def get_global_events() -> list:
    """Get major global events affecting markets."""
    # Event categories
    event_categories = {
        'Policy & Regulation': [
            'tariffs OR trade policy OR trade war',
            'antitrust OR regulation OR regulatory',
            'monetary policy OR interest rates OR central bank',
            'sanctions OR export controls'
        ],
        'Geopolitical': [
            'geopolitical tension OR diplomatic relations',
            'international conflict OR military action',
            'trade relations OR economic cooperation',
            'political uncertainty OR election impact'
        ],
        'Economic': [
            'economic data OR economic indicators',
            'inflation OR deflation OR stagflation',
            'recession OR economic growth',
            'employment OR labor market'
        ],
        'Corporate': [
            'earnings OR financial results',
            'merger OR acquisition OR takeover',
            'bankruptcy OR restructuring',
            'corporate investigation OR probe'
        ],
        'Technology': [
            'AI OR artificial intelligence',
            'semiconductor OR chip shortage',
            'cybersecurity OR data privacy',
            'tech regulation OR tech policy'
        ]
    }
    
    all_articles = []
    for category, queries in event_categories.items():
        for query in queries:
            try:
                params = {
                    'q': query,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 2,
                    'domains': 'reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,marketwatch.com'
                }
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    headers={'Authorization': f'Bearer {os.getenv("NEWS_API_KEY")}'}
                )
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    for article in articles:
                        article['category'] = category  # Add category to article
                    all_articles.extend(articles)
            except:
                continue
    
    return all_articles

def verify_news_accuracy(topic: str, api_key: str) -> dict:
    """Enhanced news verification including major market movers and global events."""
    if not api_key:
        return {'error': 'NEWS_API_KEY not configured'}

    try:
        # Extract key terms from topic for better news matching
        topic_terms = topic.lower().split()
        
        # Build a smarter query based on topic terms
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
            
        # Add event terms
        event_terms = {'tariff', 'probe', 'tension', 'ban', 'policy', 'regulation'}
        event_matches = [term for term in topic_terms if term in event_terms]
        if event_matches:
            query_parts.extend(event_matches)
        
        # Combine all terms or use original topic if no specific terms found
        query = ' OR '.join(query_parts) if query_parts else topic
        
        params = {
            'q': query,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'domains': 'reuters.com,bloomberg.com,cnbc.com,finance.yahoo.com,marketwatch.com'
        }
        
        # Get topic-specific articles
        topic_articles = requests.get(
            "https://newsapi.org/v2/everything", 
            params=params
        ).json().get('articles', [])
        
        # Get broader market context
        market_movers = get_major_market_movers()
        global_events = get_global_events()
        
        all_news = []
        seen_titles = set()
        
        # Add topic-specific articles first
        for article in topic_articles:
            if article['title'] not in seen_titles:
                article['category'] = '📊 Topic Analysis'
                all_news.append(article)
                seen_titles.add(article['title'])
        
        # Add relevant market movers
        for article in market_movers:
            if article['title'] not in seen_titles:
                article['category'] = '🔴 Market Move'
                all_news.append(article)
                seen_titles.add(article['title'])
        
        # Add relevant global events
        for article in global_events:
            if article['title'] not in seen_titles:
                article['category'] = f'🌍 {article.get("category", "Global Impact")}'
                all_news.append(article)
                seen_titles.add(article['title'])
        
        if not all_news:
            return {'error': 'No articles found'}
        
        return {
            'latest_date': all_news[0]['publishedAt'],
            'key_facts': [f"{article['category']}: {article['title']}" for article in all_news[:5]],
            'sources': {article['source']['name'] for article in all_news},
            'is_current': True,
            'descriptions': [article['description'] for article in all_news[:3]]
        }
        
    except Exception as e:
        return {'error': str(e)}

@st.cache_data(ttl=3600)
def fetch_market_context(topic: str) -> Tuple[str, bool]:
    """Fetch relevant market data."""
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
    for keyword, ticker in ticker_map.items():
        if keyword in topic.lower():
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period="1mo")
                latest_data = history.tail(3)
                market_context.append(
                    f"Latest {keyword.capitalize()} data ({ticker}):\n{latest_data.to_string()}\n"
                )
            except Exception:
                continue
    
    return "\n".join(market_context) if market_context else "", bool(market_context)

def get_relevant_examples(topic: str, learning_context: dict) -> str:
    """Get relevant examples based on topic keywords."""
    examples = []
    keywords = set(topic.lower().split())
    
    for key, content in learning_context.items():
        if not isinstance(content, dict) or not content.get('is_good_example'):
            continue
            
        example_content = content.get('content', '')
        score = sum(keyword in example_content.lower() for keyword in keywords)
        if score > 0:
            examples.append((example_content, score))
    
    examples.sort(key=lambda x: x[1], reverse=True)
    return "\n".join(ex[0] for ex in examples[:2])

def make_api_request(url: str, payload: dict, api_key: str, is_openai: bool = False, max_retries: int = 3) -> Union[dict, str]:
    """Enhanced API request handler with OpenAI support"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            
            if not response.content:
                raise ValueError("Empty response received")
            
            response_data = response.json()
            
            # Update current API in session state
            st.session_state.current_api = "OpenAI" if is_openai else "DeepSeek"
            
            if is_openai:
                # Handle OpenAI response format
                if 'choices' in response_data and response_data['choices']:
                    return response_data
            else:
                # Handle DeepSeek response format
                return response_data
                
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            st.warning(f"Request failed (attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(2 * (attempt + 1))
        except (ValueError, json.JSONDecodeError) as e:
            if attempt == max_retries - 1:
                raise
            st.warning(f"Invalid response (attempt {attempt + 1}/{max_retries}), retrying...")
            time.sleep(2 * (attempt + 1))
            
    raise Exception("Max retries reached without successful response")

def generate_content(topic: str, market_data: str, context: str, content_type: str, max_length: int = 800, focus_area: str = "") -> str:
    """Generate content using the DeepSeek API with OpenAI fallback."""
    try:
        style_guide = st.session_state['style_guide']
        content_rules = st.session_state['content_rules']
        good_examples = get_relevant_examples(topic, st.session_state['learning_context'])
        news_info = verify_news_accuracy(topic, os.getenv("NEWS_API_KEY"))

        # Determine if generating social media content
        is_social = max_length <= 50

        # Prepare focus area emphasis if provided
        focus_emphasis = ""
        if focus_area.strip():
            if is_social:
                focus_emphasis = f"Emphasize {focus_area} in the tweet/caption while maintaining engagement"
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
            news_context = "Focus on current market conditions and technical analysis"
        else:
            if is_social:
                # Simplified news context for social media
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

        # Prepare educational content prompt
        if content_type == "educational":
            prompt = f"""
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

        # Prepare market analysis prompt
        elif content_type == "market_analysis":
            prompt = f"""
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

Style Guidelines:
{json.dumps(style_guide, indent=2)}

Content Rules:
{json.dumps(content_rules, indent=2)}

Good Examples:
{good_examples}"""

        # Prepare social media prompt
        else:
            prompt = f"""
Generate a compelling {max_length}-word tweet/caption about {topic} for social media.

Tweet Structure Requirements:
1. Must be exactly {max_length} words with a complete thought
2. Follow this structure:
   - Start with the key insight/observation
   - Add supporting detail or context
   - End with a clear conclusion or market implication
3. Format:
   - Main message first
   - Market impact or implication
   - Relevant $cashtags
   - Essential #hashtags last

Example Tweet Structures:
✅ "Gold hits new highs as US-China tensions escalate. Safe-haven demand surges amid global uncertainty. $GLD $XAU #Gold #Trading"
✅ "EURUSD breaks key resistance on hawkish ECB stance. Technical setup suggests further upside potential. $EURUSD #Forex"

{focus_emphasis}

Market Context:
{news_context}

Style Guidelines for Social:
- Be concise and impactful
- Use professional but engaging tone
- Add market-specific tags (e.g., $BTC, $EURUSD)
- Keep it factual and insight-focused

Content Rules:
{json.dumps(content_rules, indent=2)}"""

        system_role = (
            "You are a social media expert specializing in financial markets content." if is_social
            else "You are an experienced educator specializing in financial markets." if content_type == "educational"
            else "You are an expert financial analyst specializing in market analysis."
        )

        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]

        # Try DeepSeek first
        try:
            deepseek_payload = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7 if not is_social else 0.8,
                "max_tokens": max_length * 4,
                "top_p": 0.95,
                "stream": False
            }
            
            with st.spinner("🤖 Generating with DeepSeek..."):
                response_data = make_api_request(deepseek_url, deepseek_payload, deepseek_api)
                
                if isinstance(response_data, dict):
                    if 'choices' in response_data and response_data['choices']:
                        return response_data['choices'][0]['message']['content']
                    elif 'response' in response_data:
                        return response_data['response']
                    elif 'content' in response_data:
                        return response_data['content']
                
        except Exception as deepseek_error:
            st.warning("⚠️ DeepSeek API unavailable, falling back to OpenAI...")
            
            try:
                # Fallback to OpenAI
                openai_payload = {
                    "model": "gpt-4",
                    "messages": messages,
                    "temperature": 0.7 if not is_social else 0.8,
                    "max_tokens": max_length * 4,
                }
                
                with st.spinner("🤖 Generating with OpenAI..."):
                    response_data = make_api_request(
                        openai_url, 
                        openai_payload, 
                        openai_api,
                        is_openai=True
                    )
                    
                    if isinstance(response_data, dict) and 'choices' in response_data:
                        return response_data['choices'][0]['message']['content']
                    
            except Exception as openai_error:
                raise Exception(f"Both APIs failed. DeepSeek error: {str(deepseek_error)}, OpenAI error: {str(openai_error)}")
        
        return "Error: Unable to generate content with either API"

    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        print(f"Detailed error: {str(e)}")
        return f"Error occurred: {str(e)}"

def translate_content(content: str, target_language: str) -> str:
    """Translate content using the enhanced TRCEI template with fallback."""
    if not content:
        return "No content to translate"
        
    translation_prompt = f"""
TRCEI Template for Language Translator

Task: Translate the given text into {SUPPORTED_LANGUAGES[target_language]}, ensuring the translation feels natural and conversational. Prioritize relatability over excessive formality, and use direct English terms where they are commonly understood in financial contexts.

Role: You are a professional linguist and native speaker of {SUPPORTED_LANGUAGES[target_language]} with expertise in financial markets and trading terminology.

Context: The translated content is for traders and financial market participants who need clear, accurate market information. The translation should feel fluent and culturally appropriate while maintaining technical accuracy.

Examples:
Good Example (Simple and Natural):
Original: "Market Analysis: USD/JPY breaks key resistance level at 150.00, supported by rising US yields. Technical indicators suggest bullish momentum. $USDJPY #Forex"
Translation should:
- Keep technical terms like "USD/JPY" unchanged
- Maintain all numbers and levels (150.00)
- Keep cashtags ($USDJPY) and hashtags (#Forex) in original form
- Use natural language for market concepts

Bad Example (Overly Formal):
- Avoid overly complex translations of common trading terms
- Don't translate technical indicators into complex local terms
- Avoid formal language when casual terms are commonly used

Instructions:
1. Translation Guidelines:
   - Keep all trading platform terms in English
   - Maintain all numerical data exactly as shown
   - Preserve market symbols, cashtags, and hashtags
   - Use commonly understood financial terms
   - Maintain paragraph structure and formatting

2. Specific Requirements:
   - Keep all chart patterns in English (e.g., "bullish flag", "double top")
   - Maintain technical indicators in English (RSI, MACD, etc.)
   - Keep currency pairs and market names unchanged
   - Preserve all percentages and decimal points
   - Keep time frames in original format (1H, 4H, 1D, etc.)

3. Formatting Requirements:
   - Preserve all bullet points and lists
   - Maintain paragraph breaks
   - Keep any special characters or symbols
   - Preserve any emphasis (bold, italic) markers

Content to translate:
{content}

Please provide the translation in {SUPPORTED_LANGUAGES[target_language]}, ensuring it follows all the above guidelines while maintaining natural flow and readability."""

    messages = [
        {
            "role": "system", 
            "content": "You are an expert financial translator with native fluency in the target language and deep knowledge of financial markets and trading terminology."
        },
        {
            "role": "user", 
            "content": translation_prompt
        }
    ]

    # Try DeepSeek first
    try:
        deepseek_payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": len(content) * 3,  # Increased token limit
            "top_p": 0.95,
            "stream": False
        }
        
        with st.spinner(f"🤖 Translating to {SUPPORTED_LANGUAGES[target_language]} with DeepSeek..."):
            response_data = make_api_request(deepseek_url, deepseek_payload, deepseek_api)
            
            if isinstance(response_data, dict):
                # Handle DeepSeek response format
                if 'choices' in response_data and response_data['choices']:
                    if 'message' in response_data['choices'][0]:
                        return response_data['choices'][0]['message']['content']
                    elif 'content' in response_data['choices'][0]:
                        return response_data['choices'][0]['content']
                elif 'response' in response_data:
                    return response_data['response']
                elif 'content' in response_data:
                    return response_data['content']
                else:
                    raise ValueError("Unexpected DeepSeek response format")
            
    except Exception as deepseek_error:
        st.warning("⚠️ DeepSeek API unavailable for translation, falling back to OpenAI...")
        print(f"DeepSeek error: {str(deepseek_error)}")  # For debugging
        
        try:
            # Fallback to OpenAI
            openai_payload = {
                "model": "gpt-4",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": len(content) * 3,  # Increased token limit
            }
            
            with st.spinner(f"🤖 Translating to {SUPPORTED_LANGUAGES[target_language]} with OpenAI..."):
                response_data = make_api_request(
                    openai_url, 
                    openai_payload, 
                    openai_api,
                    is_openai=True
                )
                
                if isinstance(response_data, dict) and 'choices' in response_data:
                    return response_data['choices'][0]['message']['content']
                else:
                    raise ValueError("Unexpected OpenAI response format")
                
        except Exception as openai_error:
            print(f"OpenAI error: {str(openai_error)}")  # For debugging
            raise Exception(f"Translation failed. DeepSeek error: {str(deepseek_error)}, OpenAI error: {str(openai_error)}")
    
    return "Error: Unable to translate content with either API"

def main():
    """Main application function."""
    st.title("Market Content Generator")
    
    if not load_resources():
        st.error("Missing required files")
        return

    # Reset API status at start of each run
    st.session_state['current_api'] = None

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
                market_data, _ = fetch_market_context(topic)
                
                content = generate_content(
                    topic=topic,
                    market_data=market_data,
                    context=context,
                    content_type=content_type,
                    max_length=max_length,
                    focus_area=focus_area
                )

                if content:
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
            if content_to_translate:
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
            with st.spinner(f"🌐 Translating to {SUPPORTED_LANGUAGES[target_language]}..."):
                translated_content = translate_content(content_to_translate, target_language)
                
                if translated_content:
                    # Store translation in session state
                    st.session_state['translated_content'] = translated_content
                    
                    # Display API info
                    api_used = st.session_state.get('current_api', 'Unknown')
                    st.info(f"📢 Translated using {api_used} API")
                    
                    # Display translation
                    st.markdown(f"### {SUPPORTED_LANGUAGES[target_language]} Translation:")
                    st.markdown(translated_content)
                    
                    # Download translation
                    st.download_button(
                        "📥 Download Translation",
                        data=translated_content,
                        file_name=f"translated_{target_language}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        elif st.button("Translate"):
            st.error("⚠️ Please provide content to translate.")

if __name__ == "__main__":
    main()