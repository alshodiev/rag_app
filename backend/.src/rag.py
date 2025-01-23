from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter # for defining our chain
from decouple import config
from qdrant import vector_store

# Use another model -> ?
model = ChatOpenAI(
    model_name = 'gpt-4-turbo-overview',
    openai_api_key = config('OPENAI_API_KEY'),
    temperature = 0,
)

# Make this prompt better
prompt_template = """
Answer the question based on the context, in a concise manner and using bullet points where applicable.

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vector_store.as_retriever()

# Chain more actions
def create_chain():
    chain = (
        {
            "context" : retriever.with_config(top_k = 4),
            "question" : RunnablePassthrough(),

        }
        | RunnableParallel({
            "response" : prompt | model,
            "context" : itemgetter("context"),
        })
    )
    return chain

def get_answer_and_docs(question:str):
    chain = create_chain()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]
    return {
        "answer" : answer, 
        "context" : context
    }

response = get_answer_and_docs("What is the best way to evaluate a machine learning model?")
print(response)