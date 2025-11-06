from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_message}"),
        MessagesPlaceholder("messages")
    ]
)

llm_model = prompt_template | llm