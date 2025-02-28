import streamlit as st
import os
import json
import anthropic
import requests
from io import BytesIO
import base64
import re
from openai import OpenAI
import pandas as pd
import zipfile
import io

# Set page configuration
st.set_page_config(
    page_title="Translation Engine Beta",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "translator_pps":  # Your custom password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("Please enter the password to access the Translation Engine.")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Now check password and show app only if correct
if not check_password():
    st.stop()  # Stop here if password not entered correctly

# Custom CSS for styling
st.markdown("""
<style>
    /* Title styling */
    .main-title {
    font-size: 1rem;  /* Further reduced from 1.2rem to 1rem */
    font-weight: 300;
    color: #0A2647;
    background-color: transparent;  /* Removed blue background */
    padding: 1rem;               
    border-radius: 8px;         
    text-align: center;
    margin: 1rem 0 1.5rem 0;
    width: 100%;
    display: block;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 500;
        color: #144272;
        background-color: #E3F2FD;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #2C74B3;
        color: white;
        font-weight: normal;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #205295;
    }
    
    /* Output container */
    .output-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin-top: 1rem;
    }
    
    .translated {
        color: #2C74B3;
        font-weight: medium;
        margin-top: 0.5rem;
        white-space: pre-wrap;
    }
    
    /* Language tag */
    .language-tag {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.3rem 0.6rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }
    
    /* Fix text area input */
    .stTextArea textarea {
        min-height: 350px !important;
        border-radius: 0.3rem;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Make the columns more balanced */
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    /* Language dropdown styling */
    .language-dropdown {
        margin-bottom: 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-title">Translation Engine Beta <span style="font-size: 0.6em;">🚀</span></h1>', unsafe_allow_html=True)

# Utility functions
def load_env_variables():
    """
    Load environment variables from .env file if python-dotenv is installed.
    This is the only source of API keys - no user input allowed.
    
    Returns:
        dict: Dictionary of loaded environment variables
    """
    env_vars = {
        'CLAUDE_API_KEY': None,
        'OPENAI_API_KEY': None
    }
    
    try:
        from dotenv import load_dotenv
        # Silent loading without print statements
        load_dotenv()
        env_vars['CLAUDE_API_KEY'] = os.getenv('CLAUDE_API_KEY')
        env_vars['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    except ImportError:
        # Silent failure - don't print anything
        pass
    except Exception:
        # Silent failure - don't print anything
        pass
    
    return env_vars

def split_into_sentences(text):
    """
    Split text into sentences using regex pattern matching.
    
    Args:
        text (str): Text to split into sentences
        
    Returns:
        list: List of sentences
    """
    # Simple regex-based sentence splitting to avoid NLTK dependencies
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

@st.cache_data
def load_translation_prompt():
    """
    Load the translation prompt template from a file.
    Throws an error if the file is not found.
    
    Returns:
        str: Content of the translation prompt template
    """
    try:
        # Try to load from the current directory
        with open("translation_prompt.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Instead of using a fallback, raise an error
        st.error("Critical Error: translation_prompt.md file not found!")
        st.stop()  # Stop the application
        return None

def load_translation_examples_json(target_language):
    """
    Load translation examples from the custom structured JSON file for a specific target language.
    """
    try:
        import json
        import os
        
        # Check if file exists - print current working directory for debugging
        json_file = "translation_examples.json"
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        print(f"Looking for file: {os.path.join(current_dir, json_file)}")
        
        if not os.path.exists(json_file):
            print(f"Translation examples file not found: {json_file}")
            # Try to list files in the current directory to help debug
            try:
                files = os.listdir(current_dir)
                print(f"Files in current directory: {files}")
            except Exception as list_err:
                print(f"Error listing directory: {str(list_err)}")
            return []
        
        print(f"JSON file found: {json_file}")
        
        # Load JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                print("JSON loaded successfully")
                print(f"JSON structure contains keys: {list(data.keys())}")
            except Exception as json_err:
                print(f"Error parsing JSON: {str(json_err)}")
                return []
        
        # Get language code
        language_code = None
        for code, lang in data.get("languageMap", {}).items():
            if target_language.lower() in lang.lower():
                language_code = code
                break
        
        if not language_code:
            print(f"Could not find language code for {target_language}")
            return []
            
        # Extract examples from the commonPhrases
        examples = []
        
        # Add examples from commonPhrases
        for phrase_key, phrase_data in data.get("commonPhrases", {}).items():
            if "original" in phrase_data and "translations" in phrase_data:
                original = phrase_data["original"]
                if language_code in phrase_data["translations"]:
                    translation = phrase_data["translations"][language_code]
                    examples.append({
                        "source": original,
                        "translation": translation
                    })
        
        # Add examples from emailTemplates
        for template_key, template_data in data.get("emailTemplates", {}).items():
            # Extract subject
            if "subject" in template_data:
                subject = template_data["subject"]
                if "original" in subject and "translations" in subject and language_code in subject["translations"]:
                    examples.append({
                        "source": subject["original"],
                        "translation": subject["translations"][language_code]
                    })
            
            # Extract from body sections
            if "body" in template_data:
                for section_key, section_data in template_data["body"].items():
                    if "original" in section_data and "translations" in section_data and language_code in section_data["translations"]:
                        examples.append({
                            "source": section_data["original"],
                            "translation": section_data["translations"][language_code]
                        })
            
            # Extract CTA
            if "cta" in template_data:
                cta = template_data["cta"]
                if "original" in cta and "translations" in cta and language_code in cta["translations"]:
                    examples.append({
                        "source": cta["original"],
                        "translation": cta["translations"][language_code]
                    })
        
        print(f"Found {len(examples)} examples for {target_language}")
        
        # Limit to 5 diverse examples to avoid overwhelming the prompt
        return examples[:5]
        
    except Exception as e:
        print(f"Error loading translation examples from JSON: {str(e)}")
        return []

# Replace the load_translation_examples function to only use JSON
def load_translation_examples(target_language):
    """
    Load translation examples for a specific target language.
    Only uses JSON file, no Excel fallback.
    
    Args:
        target_language (str): The target language to filter examples by
        
    Returns:
        list: List of example pairs (source, translation)
    """
    return load_translation_examples_json(target_language)

def get_download_link(text, filename, link_text):
    """
    Create a download link for text content.
    
    Args:
        text (str): Text to download
        filename (str): Name of the file to download
        link_text (str): Text to display for the download link
        
    Returns:
        str: HTML for download link
    """
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="action-button">{link_text}</a>'
    return href

# Function to extract just the translation from model output
def extract_translation(text):
    """
    Extract just the translated text from the model response, stripping out any commentary or metadata.
    Preserves all paragraphs of the translation.
    
    Args:
        text (str): The model response
        
    Returns:
        str: Just the translated text with all paragraphs preserved
    """
    # Try to extract text from code blocks which often contain just the translation
    code_blocks = re.findall(r'```(?:\w+)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Try to extract text after a translation label, capturing ALL content after the label
    translation_match = re.search(r'(?:翻译|translation|translated text)[:：]\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if translation_match:
        return translation_match.group(1).strip()
    
    # If the text starts with the target language name, capture ALL content after it
    language_match = re.search(r'^(?:Chinese|Thai|Vietnamese|Spanish|Korean|Hindi)(?:\s+\((?:Simplified|Traditional)\))?[:\n](.*)', text, re.IGNORECASE | re.DOTALL)
    if language_match:
        return language_match.group(1).strip()
    
    # Check if the model has returned just the translation without any explanatory text
    # This is our goal scenario from the system prompt, so it's likely
    lines = text.split('\n')
    if len(lines) > 2 and not any(keyword in text.lower() for keyword in ["here is", "translation", "translated", "i've translated"]):
        return text
        
    # If all else fails, return the original text but warn in the log
    print("Could not extract just translation, returning full model output")
    return text

# Function to translate text with Claude API
def translate_with_claude(text, target_language, content_type, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Get the prompt template and customize it
        prompt_template = load_translation_prompt()
        if not prompt_template:
            return None
            
        prompt = prompt_template.replace("[Target Language]", target_language)
        prompt = prompt.replace("[specific audience type: casual readers/business professionals/technical experts/students/etc.]", 
                              f"readers of {content_type}")
        
        # Load examples for the target language
        examples = load_translation_examples(target_language)

        # Add examples to the prompt if available
        if examples:
            prompt += "\n\n## Translation Examples:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"\n### Example {i}:\n"
                prompt += f"Source: \n```\n{example['source']}\n```\n\n"
                prompt += f"Translation: \n```\n{example['translation']}\n```\n"

        # Add the text to translate (only once)
        prompt += f"\n\nText to translate:\n```\n{text}\n```"
        
        # Prepare the system prompt with enhanced financial understanding
        system_prompt = f"""You are an expert translator specializing in {target_language} with deep knowledge of financial markets and terminology.

Follow these translation guidelines carefully:
1. Provide ONLY the translated text without any commentary, explanations, or metadata
2. Do not include words like "translation:" or "translated text:" or any other labels
3. Just respond with the pure translated text
4. Preserve all financial terms, ticker symbols, and numerical values in their original form
5. Understand financial market terminology and idioms - translate the MEANING not just the words
6. For market concepts like "softening" (prices easing/declining gradually), "hawkish" (favoring higher interest rates), "dovish" (favoring lower rates), etc., use the appropriate financial market terminology in the target language

Examples of market phrases to understand (not translate literally):
- "Tariffs have softened" → means tariff rates have been reduced/eased
- "The market is witnessing a correction" → means prices are falling after a period of increase
- "The central bank has adopted a hawkish stance" → means they are likely to raise interest rates
- "Bears have control of the market" → means sellers are driving prices down
- "Liquidity is drying up" → means less money is available for trading

For each financial term or market idiom, translate based on the underlying financial concept, not the literal words."""
        
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        st.error(f"Claude API Error: {str(e)}")
        return None
    
    
# Function to translate text with OpenAI GPT-4o API
def translate_with_gpt4o(text, target_language, content_type, api_key):
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Get the prompt template and customize it
        prompt_template = load_translation_prompt()
        if not prompt_template:
            return None
            
        prompt = prompt_template.replace("[Target Language]", target_language)
        prompt = prompt.replace("[specific audience type: casual readers/business professionals/technical experts/students/etc.]", 
                              f"readers of {content_type}")
        
        # Load examples for the target language
        examples = load_translation_examples(target_language)

        # Add examples to the prompt if available
        if examples:
            prompt += "\n\n## Translation Examples:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"\n### Example {i}:\n"
                prompt += f"Source: \n```\n{example['source']}\n```\n\n"
                prompt += f"Translation: \n```\n{example['translation']}\n```\n"
        
        # Add the text to translate (only once)
        prompt += f"\n\nText to translate:\n```\n{text}\n```"
        
        # Prepare the system message with enhanced financial understanding
        system_message = f"""You are an expert translator specializing in {target_language} with deep knowledge of financial markets and terminology.

Follow these translation guidelines carefully:
1. Provide ONLY the translated text without any commentary, explanations, or metadata
2. Do not include words like "translation:" or "translated text:" or any other labels
3. Just respond with the pure translated text
4. Preserve all financial terms, ticker symbols, and numerical values in their original form
5. Understand financial market terminology and idioms - translate the MEANING not just the words
6. For market concepts like "softening" (prices easing/declining gradually), "hawkish" (favoring higher interest rates), "dovish" (favoring lower rates), etc., use the appropriate financial market terminology in the target language

Examples of market phrases to understand (not translate literally):
- "Tariffs have softened" → means tariff rates have been reduced/eased
- "The market is witnessing a correction" → means prices are falling after a period of increase
- "The central bank has adopted a hawkish stance" → means they are likely to raise interest rates
- "Bears have control of the market" → means sellers are driving prices down
- "Liquidity is drying up" → means less money is available for trading

For each financial term or market idiom, translate based on the underlying financial concept, not the literal words."""
        
        # Create messages array for the API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Make the API request using the OpenAI client
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4000
            )
            
            # Extract content from the response
            if completion and completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                return content
            else:
                st.error("Empty response from OpenAI API")
                return None
                
        except Exception as api_error:
            error_message = str(api_error)
            st.error(f"OpenAI API error: {error_message}")
            # Print detailed error for debugging
            print(f"Detailed OpenAI API error: {error_message}")
            return None
            
    except Exception as e:
        st.error(f"OpenAI API setup error: {str(e)}")
        return None
    
# Load environment variables - ONLY from .env file
env_vars = load_env_variables()
claude_api_key = env_vars.get('CLAUDE_API_KEY')
openai_api_key = env_vars.get('OPENAI_API_KEY')

# Set API keys directly in session state - no user input required
if claude_api_key:
    st.session_state.claude_api_key = claude_api_key
if openai_api_key:
    st.session_state.openai_api_key = openai_api_key

# Check if keys are already set
claude_key_set = hasattr(st.session_state, 'claude_api_key') and st.session_state.claude_api_key
openai_key_set = hasattr(st.session_state, 'openai_api_key') and st.session_state.openai_api_key

# Check prompt and examples status before adding to sidebar
prompt_loaded = False
prompt_length = 0
examples_loaded = False
example_count = 0
examples = []

# Check if prompt is loaded
try:
    prompt_content = load_translation_prompt()
    if prompt_content:
        prompt_loaded = True
        prompt_length = len(prompt_content.split('\n'))
except Exception as e:
    prompt_loaded = False
    print(f"Error loading prompt: {str(e)}")

# Check if examples are loaded
try:
    # Check one example language to see if any examples exist
    examples = load_translation_examples("Chinese (Simplified)")
    if examples and len(examples) > 0:
        examples_loaded = True
        example_count = len(examples)
except Exception as e:
    examples_loaded = False
    print(f"Error loading examples: {str(e)}")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # API Keys status - only display status, no input fields
    st.subheader("API Status")
    
    # Show status for the API keys
    st.write("Claude API: " + ("✅ Available" if claude_key_set else "❌ Not available"))
    st.write("OpenAI API: " + ("✅ Available" if openai_key_set else "❌ Not available"))
    
    if not claude_key_set and not openai_key_set:
        st.warning("No API keys found. Please add them to your .env file.")
    
    st.divider()
    
    # Add help information
    st.info("""
    **Content Types:**
    - Market Analysis: Financial market reports, trends, technical market data and forecasts
    - Educational Content: Tutorials, guides, and educational materials
    - Email/Post: Short communications, announcements, social media posts
    """)
    
    # Add information about translation model with learning emoji
    st.success("📝 This translation model has been trained as per good examples from previously translated material in Pepperstone, also using the best practices for financial markets content translation.")

    st.divider()
    st.subheader("Resources Status")
    
    # Show prompt status
    if prompt_loaded:
        st.write("✅ Translation prompt loaded (" + str(prompt_length) + " lines)")
    else:
        st.write("❌ Translation prompt not found!")
    
    # Show examples status
    if examples_loaded:
        st.write("✅ Translation examples available")
        with st.expander("View sample examples"):
            st.write(f"Found {example_count} examples for Chinese (Simplified)")
            # Optionally show a sample
            if len(examples) > 0:
                st.write("Sample source: " + examples[0]['source'][:50] + "...")
                st.write("Sample translation: " + examples[0]['translation'][:50] + "...")
    else:
        st.write("❌ Translation examples not found!")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<p class="sub-header"> Source </p>', unsafe_allow_html=True)
    source_text = st.text_area(
        "Please enter source text in english!",
        height=300,
        placeholder="Enter your English text here (up to 3000 words)..."
    )
    
    # Change from selectbox to multiselect for multiple languages
    target_languages = st.multiselect(
        "Select Target Language(s)",
        options=["Chinese (Simplified)", "Chinese (Traditional)", "Thai", "Vietnamese", "Spanish", "Korean", "Hindi"],
        default=["Chinese (Simplified)"]
    )
    
    # Warning if too many languages selected
    if len(target_languages) > 3:
        st.warning("Translating to multiple languages will take longer. Consider translating to fewer languages for faster results.")
    
    content_type = st.selectbox(
        "Content Type",
        options=["Market Analysis", "Educational Content", "Email/Post"],
        index=0
    )

    # Only show models that have available API keys
    available_models = []
    if claude_key_set:
        available_models.append("Claude 3.7 Sonnet")
    if openai_key_set:
        available_models.append("GPT-4o")
    
    if available_models:
        model_option = st.radio(
            "Select Translation Model",
            options=available_models,
            horizontal=True
        )
    else:
        st.error("No translation models available. Please add API keys to your .env file.")
        model_option = None
    
    translate_button = st.button("Translate")
    
    if translate_button and not target_languages:
        st.markdown('<div class="status-message warning">Please choose at least one language to translate to.</div>', unsafe_allow_html=True)
    
    if translate_button and not source_text:
        st.markdown('<div class="status-message warning">Please enter text to translate.</div>', unsafe_allow_html=True)
        
    if not available_models:
        st.markdown('<div class="status-message warning">No translation models available. Check your .env file.</div>', unsafe_allow_html=True)

# Initialize session state for translations if it doesn't exist
if 'translations' not in st.session_state:
    st.session_state.translations = {}
if 'translation_done' not in st.session_state:
    st.session_state.translation_done = False

# Modified translation process for multiple languages
if translate_button and source_text and target_languages and model_option:
    # Reset translations when starting new translation
    st.session_state.translations = {}
    st.session_state.translation_done = False
    
    # Create a progress bar
    progress_bar = st.progress(0)
    total_languages = len(target_languages)
    
    # Translate to each selected language
    for i, target_language in enumerate(target_languages):
        with st.spinner(f"Translating to {target_language} using {model_option}... ({i+1}/{total_languages})"):
            model_output = None
            
            try:
                if model_option == "Claude 3.7 Sonnet":
                    model_output = translate_with_claude(
                        source_text, 
                        target_language, 
                        content_type, 
                        st.session_state.claude_api_key
                    )
                else:  # GPT-4o
                    model_output = translate_with_gpt4o(
                        source_text, 
                        target_language, 
                        content_type, 
                        st.session_state.openai_api_key
                    )
                    
                # Extract just the translation from model output
                if model_output:
                    st.session_state.translations[target_language] = extract_translation(model_output)
                else:
                    st.error(f"No response from translation service for {target_language}.")
                    
            except Exception as e:
                st.error(f"Translation error for {target_language}: {str(e)}")
                
            # Update progress bar
            progress_bar.progress((i + 1) / total_languages)
    
    # Clear progress bar after completion
    progress_bar.empty()
    
    # Mark translation as done
    if st.session_state.translations:
        st.session_state.translation_done = True
        
        # Set the first language as default selection if not already set
        if 'selected_language' not in st.session_state and st.session_state.translations:
            st.session_state.selected_language = list(st.session_state.translations.keys())[0]
            
# Display translations (either from current run or previous run in session state)
if st.session_state.translation_done and st.session_state.translations:
    with col2:
        st.markdown('<p class="sub-header">Translation</p>', unsafe_allow_html=True)
        
        # If we have multiple translations, add a language selector
        if len(st.session_state.translations) > 1:
            # Use callback to update selected language when dropdown changes
            def update_selected_language():
                pass  # The callback just needs to exist
                
            selected_language = st.selectbox(
                "Select language to display",
                options=list(st.session_state.translations.keys()),
                key="selected_language",
                on_change=update_selected_language
            )
        else:
            # Just one language was selected
            selected_language = list(st.session_state.translations.keys())[0]
            st.session_state.selected_language = selected_language
            
        translation = st.session_state.translations[st.session_state.selected_language]
        st.markdown(f'<span class="language-tag">{st.session_state.selected_language}</span>', unsafe_allow_html=True)
        
        # Display the selected translation
        st.markdown(f'<div class="output-container translated">{translation}</div>', unsafe_allow_html=True)
        
        # Download button for the currently displayed translation
        st.download_button(
            label=f"Download {st.session_state.selected_language} Translation",
            data=translation,
            file_name=f"translation_{st.session_state.selected_language.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt",
            mime="text/plain",
            key="download_single_btn"
        )
        
        # Add option to download all translations as a ZIP if multiple were generated
        if len(st.session_state.translations) > 1:
            st.markdown("#### Download All Translations")
            
            # Create ZIP file with all translations
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for lang, trans in st.session_state.translations.items():
                    file_name = f"translation_{lang.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt"
                    zip_file.writestr(file_name, trans)
            
            # Reset buffer position to beginning
            zip_buffer.seek(0)
            
            # Offer the ZIP file for download
            st.download_button(
                label="Download All Translations (ZIP)",
                data=zip_buffer,
                file_name="all_translations.zip",
                mime="application/zip",
                key="download_zip_btn"
            )
                
# Add debug information for development purposes
if st.sidebar.checkbox("Show Debug Info", value=False):
    with st.sidebar.expander("Debug Information"):
        st.write("API Keys:")
        st.write(f"- Claude API Key: {'Set' if claude_key_set else 'Not set'}")
        st.write(f"- OpenAI API Key: {'Set' if openai_key_set else 'Not set'}")
        
        if 'openai_error' in st.session_state:
            st.write("Last OpenAI API Error:")
            st.code(st.session_state.get('openai_error', 'None'))
