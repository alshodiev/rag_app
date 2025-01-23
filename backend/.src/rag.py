from langchain.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_community.models import ChatOpenAI
from operator import itemgetter # for defining our chain

from decouple import config

from qdrant import vector_store

model = ChatOpenAI(
    model_name = 'gpt-4-turbo-overview',
    openai_api_key = config('OPENAI_API_KEY'),
    temperature = 0.5,
)

prompt_template = """
Answer the question based on the context, in a concise manner and using bullet points where applicable.

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vector_store.as_retriever()

def create_chain():
    chain = (
        {
                "context" : retriever.with_config(top_k = 4),
                "answer" : RunnablePassthrough(),

        }
        | RunnableParallel({
            "response" : prompt | model,
            "context" : itemgetter("context"),
        })
    )