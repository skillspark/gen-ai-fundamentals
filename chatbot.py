import streamlit as st
from ConversationManager import ConversationManager

def main():
    st.title("AI Chatbot using LlamaChat")
    st.write("This is a simple chatbot interface that uses a Llama generative text model to generate responses.")

    # Initialize the ConversationManager (only once per session)
    if 'conversation_manager' not in st.session_state:
        st.session_state['conversation_manager'] = ConversationManager()

    # Persona selection
    personas = list(st.session_state['conversation_manager'].system_messages.keys())
    selected_persona = st.selectbox("Choose a conversation persona:", personas)
    
    # Set the selected persona
    st.session_state['conversation_manager'].set_persona(selected_persona)

    # Custom system message option
    if selected_persona == 'custom':
        custom_system_message = st.text_area("Enter custom system message:")
        if st.button("Set Custom Message"):
            st.session_state['conversation_manager'].set_custom_system_message(custom_system_message)

    # Conversation interface
    user_input = st.text_input("Enter your message:")
    
    if st.button("Send"):
        if user_input:
            # Generate response using ConversationManager
            response = st.session_state['conversation_manager'].chat_completion(user_input)
            
            # Display the response
            if response:
                st.write("Bot:", response)
            else:
                st.error("Failed to generate a response.")

    # Additional controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset Conversation"):
            st.session_state['conversation_manager'].reset_conversation_history()
            st.success("Conversation history reset.")

    with col2:
        if st.button("View Conversation History"):
            st.json(st.session_state['conversation_manager'].conversation_history)

if __name__ == "__main__":
    main()