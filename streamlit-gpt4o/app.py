import logging
import os
from datetime import datetime
from uuid import uuid4

import streamlit as st
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from st_multimodal_chatinput import multimodal_chatinput
from dotenv import load_dotenv
load_dotenv()

__version__ = "0.0.4"

st.set_page_config(
    page_title=f"streamlit-gpt4o v{__version__}",
    page_icon="🤖",
)

logging.basicConfig(level=logging.DEBUG)


def chat_input_to_human_message(chat_input: dict) -> HumanMessage:
    text = chat_input.get("text", "")
    images = chat_input.get("images", [])
    human_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": text,
            },
        ]
        + [
            {
                "type": "image_url",
                "image_url": {
                    "url": image,
                },
            }
            for image in images
        ],
    )
    return human_message


def render_human_contents(msg: HumanMessage) -> None:
    for d in msg.content:
        if d["type"] == "text":
            st.write(d["text"])
        elif d["type"] == "image_url":
            st.image(d["image_url"]["url"], use_column_width=True)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a multimodal AI chatbot having a conversation with a human. "
            "You can accept text and images as input, but you can only respond with text. "
            "The current time is {date_time}.",
        ),
        MessagesPlaceholder(variable_name="history"),
        MessagesPlaceholder(variable_name="input"),
    ],
).partial(date_time=datetime.now().strftime("%B %d, %Y %H:%M:%S"))


llm = None
runnable = None
with_message_history = None

langsmith_api_key = None
langsmith_project_name = None
langsmith_client = None

chat_input_dict = None
chat_input_human_message = None

history = StreamlitChatMessageHistory(key="chat_messages")

if not st.session_state.get("session_id", None):
    st.session_state.session_id = str(uuid4())

top = st.container()
bottom = st.container()

with st.sidebar:
    google_api_key = st.text_input("Google Generative AI API Key", type="password")
    st.write('Gemini 1.5')
    use_flash = st.toggle(label="`Pro` ⇄ `Flash`", value=True)
    model_option = "models/gemini-1.5-flash-latest" if use_flash else "models/gemini-1.5-pro-latest"
    if google_api_key:
        llm = ChatGoogleGenerativeAI(
            model=model_option,
            streaming=True,
            verbose=True,
            google_api_key=google_api_key,
        )
        runnable = prompt | llm
        with_message_history = RunnableWithMessageHistory(
            runnable,
            lambda _: history,
            input_messages_key="input",
            history_messages_key="history",
        )

    langsmith_api_key = st.text_input("LangSmith API Key", type="password")
    langsmith_project_name = st.text_input(
        "LangSmith Project Name",
        value="streamlit-gemini",
    )
    langsmith_endpoint = st.text_input(
        "LangSmith Endpoint",
        value="https://api.smith.langchain.com",
    )
    if langsmith_api_key and langsmith_project_name:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project_name
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    else:
        for key in (
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
            "LANGCHAIN_TRACING_V2",
        ):
            os.environ.pop(key, None)

    st.markdown(
        f"## Current session ID\n`{st.session_state.get('session_id', '<none>')}`",
    )
    if st.button("Clear message history"):
        history.clear()
        st.session_state.session_id = None
        st.rerun()

# write instructions to go here to get started with google generative ai and gemini https://aistudio.google.com/


if not with_message_history:
    st.error("Please enter a Google Generative AI API key in the sidebar. \n\nTo get started with Google Generative AI and Gemini, follow these steps:\n\n1. Go to [https://aistudio.google.com/](https://aistudio.google.com/)\n2. Sign in or create a new account if you don't have one.\n3. Explore the available models and select the one that suits your needs.\n4. Obtain an API key for the Google Generative AI API.\n5. In the sidebar, enter the obtained API key in the 'Google Generative AI API Key' field.\n6. Choose whether to use the 'Pro' or 'Flash' version of the Gemini model by toggling the switch.\n7. Start using the Google Generative AI and Gemini models in your chatbot application!")

else:
    with bottom:
        chat_input_dict = multimodal_chatinput(text_color="black")
        if chat_input_dict:
            chat_input_human_message = chat_input_to_human_message(chat_input_dict)

    with top:
        for msg in history.messages:
            if msg.type.lower() in ("user", "human"):
                with st.chat_message("human"):
                    render_human_contents(msg)
            elif msg.type.lower() in ("ai", "assistant", "aimessagechunk"):
                with st.chat_message("ai"):
                    st.write(msg.content)

        if chat_input_human_message:

            with st.chat_message("human"):
                render_human_contents(chat_input_human_message)

            with st.chat_message("ai"):
                st.write_stream(
                    with_message_history.stream(
                        {"input": [chat_input_human_message]},
                        {
                            "configurable": {"session_id": st.session_state.session_id},
                        },
                    ),
                )

            chat_input_human_message = None
