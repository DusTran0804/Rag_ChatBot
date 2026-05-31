from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import get_llm
from app.service.query_transform import transform_with_multi_step
from app.core.prompt import ROUTER_SYSTEM_PROMPT

class RouterLogic(BaseModel):
    logic_type: Literal["COMPLEX", "SIMPLE"] = Field(description="Độ phức tạp của câu hỏi")
    target_index: Literal["ky_thuat", "doanh_nghiep", "chung"] = Field(description="Index dữ liệu phù hợp nhất")

def logical_router(user_query: str):
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(RouterLogic)

    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | structured_llm
    from app.core.config import invoke_chain_with_retry
    try:
        decision = invoke_chain_with_retry(chain, {"question": user_query})
    except Exception as e:
        print(f"Routing error: {e}")
        return {"queries": [user_query], "index": "chung"}

    final_queries = []
    if decision.logic_type == "COMPLEX":
        final_queries = transform_with_multi_step(user_query)
    else:
        final_queries = [user_query]
        
    return {
        "queries": final_queries,
        "index": decision.target_index
    }