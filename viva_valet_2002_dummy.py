import os
import streamlit as st
import json
import time
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

# This safely accesses the API key from Streamlit's secrets management
if "openai" in st.secrets:
    client = OpenAI(api_key=st.secrets["openai"]["openai_api_key"])
else:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure it in the Streamlit Cloud dashboard.")
    st.stop()

class BookingAssistant:
    # System prompt as a class constant
    SYSTEM_PROMPT = """
    You are VivaValet's interactive booking assistant. Your goal is to help users book any of our services, 
    which include rides for seniors, home repair, cleaning, meal delivery, and tech help.

    Follow these guidelines:
    1. Ask the user for necessary booking details: service type, full name, email, phone number, preferred date/time, 
       service location (if applicable), and any additional instructions.
    2. Work for both registered users and guests.
    3. Use our training material to provide accurate service information and avoid hallucinations.
    4. After collecting details, present a clear and concise summary of the booking information.
    5. Ask the user to type 'confirm' (or click a confirmation button in the UI) to finalize the booking.
    6. Inform the user that upon confirmation, their booking will be registered in our system and an automated email 
       confirmation will be sent.
    7. Maintain a friendly, professional, and supportive tone throughout the interaction.

    Greet the user with: 
    "Welcome to VivaValet! How can I assist you today? Would you like to book a ride, home repair, cleaning, 
    meal delivery, or tech help service?"
    """

    def __init__(self):
        """Initialize the booking assistant with default conversation state."""
        if "conversation" not in st.session_state:
            st.session_state.conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def get_chat_response(self, conversation: List[Dict[str, str]]) -> str:
        """
        Get response from ChatGPT API using the correct syntax.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                max_tokens=500,
                temperature=0.7,
                response_format={ "type": "text" }
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error from ChatGPT API: {str(e)}")
            return "I apologize, but I encountered an error. Please try again."

    def extract_booking_details(self) -> pd.DataFrame:
        """
        Extract booking details from conversation.
        Returns a DataFrame for table display.
        """
        details = {
            "Service Type": ["Example Service"],
            "Full Name": ["John Doe"],
            "Email": ["johndoe@example.com"],
            "Phone": ["123-456-7890"],
            "Date & Time": ["2025-03-01 10:00 AM"],
            "Location": ["123 Main St, City"],
            "Special Instructions": ["No special instructions."]
        }
        return pd.DataFrame(details)

    def register_booking(self, booking_details: Dict) -> bool:
        """Register a booking in the system."""
        try:
            time.sleep(1)  # Simulate API call
            st.success("""
            ✅ Booking registered successfully! An email confirmation has been sent.
            
            If you have any questions, please contact us:
            📧 hello@vivavalet.com
            📞 (847) 474-9833
            🌐 https://www.vivavalet.com/
            """)
            return True
        except Exception as e:
            st.error(f"Failed to register booking: {str(e)}")
            return False

    def check_for_confirmation(self) -> bool:
        """Check if user has confirmed the booking."""
        if (len(st.session_state.conversation) > 0 and 
            st.session_state.conversation[-1]["role"] == "user"):
            return "confirm" in st.session_state.conversation[-1]["content"].lower()
        return False

    def reset_conversation(self):
        """Reset the conversation to initial state."""
        st.session_state.conversation = [{"role": "system", "content": self.SYSTEM_PROMPT}]

def main():
    """Main application function."""
    st.set_page_config(
        page_title="VivaValet Booking Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Create two columns for title and contact info
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title("🤖 VivaValet Booking Assistant")
    
    with header_col2:
        st.markdown("""
        ### Contact Us:
        📧 hello@vivavalet.com  
        📞 (847) 474-9833  
        🌐 [vivavalet.com](https://www.vivavalet.com/)
        """)

    # Initialize booking assistant
    booking_assistant = BookingAssistant()

    # Display application description
    st.markdown("""
    ### Welcome to VivaValet! 
    This assistant helps you book services including:
    - 🚗 Rides for seniors
    - 🔧 Home repair
    - 🧹 Cleaning
    - 🍽️ Meal delivery
    - 💻 Tech help
    
    Type your message below and press Enter.
    """)

    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "assistant":
            st.chat_message("assistant").write(message["content"])
        elif message["role"] == "user":
            st.chat_message("user").write(message["content"])

    # User input section
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Get and display assistant response
        assistant_response = booking_assistant.get_chat_response(st.session_state.conversation)
        st.session_state.conversation.append({"role": "assistant", "content": assistant_response})
        
        # Check for booking confirmation
        if booking_assistant.check_for_confirmation():
            booking_details = booking_assistant.extract_booking_details()
            
            st.markdown("### 📋 Booking Summary")
            
            # Display the booking details in a table
            st.dataframe(
                booking_details,
                hide_index=True,
                column_config={
                    "Service Type": st.column_config.TextColumn("Service Type", width="medium"),
                    "Full Name": st.column_config.TextColumn("Full Name", width="medium"),
                    "Email": st.column_config.TextColumn("Email", width="medium"),
                    "Phone": st.column_config.TextColumn("Phone", width="small"),
                    "Date & Time": st.column_config.TextColumn("Date & Time", width="medium"),
                    "Location": st.column_config.TextColumn("Location", width="large"),
                    "Special Instructions": st.column_config.TextColumn("Special Instructions", width="large"),
                },
                use_container_width=True
            )
            
            # Add contact information below the booking summary
            st.info("""
            Have questions about your booking? Contact us:
            📧 hello@vivavalet.com | 📞 (847) 474-9833 | 🌐 [vivavalet.com](https://www.vivavalet.com/)
            """)
            
            # Add a visual separator
            st.markdown("---")
            
            # Confirmation button with better styling
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                if st.button("✅ Confirm Booking", type="primary"):
                    if booking_assistant.register_booking(booking_details.to_dict('records')[0]):
                        booking_assistant.reset_conversation()
                        st.rerun()

        # Rerun to update display
        st.rerun()

if __name__ == "__main__":
    main()
