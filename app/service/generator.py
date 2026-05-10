from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.prompt import GENERATOR_SYSTEM_PROMPT

def generate_answer(user_query: str, context: str, llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATOR_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "context": context,
        "input": user_query
    })
    
    return response