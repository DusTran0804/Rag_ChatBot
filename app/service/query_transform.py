from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import get_llm
from app.core.prompt import QUERY_TRANSFORM_PROMPT

def transform_with_multi_step(user_query: str) -> list:
    llm = get_llm(temperature=0.2)
    
    multi_query_prompt = ChatPromptTemplate.from_template(QUERY_TRANSFORM_PROMPT)

    multi_chain = multi_query_prompt | llm | StrOutputParser()
    response = multi_chain.invoke({"question": user_query})
    sub_queries = [q.strip() for q in response.split('\n') if q.strip()]

    cleaned_queries = []
    for q in sub_queries:
        cleaned_q = q.lstrip('0123456789. -')
        if cleaned_q:
            cleaned_queries.append(cleaned_q)

    for i, q in enumerate(cleaned_queries, 1):
        print(f"Sub-query {i}: {q}")
    
    return cleaned_queries


