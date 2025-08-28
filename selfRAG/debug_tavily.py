from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()


web_search_tool = TavilySearch(max_results=3)
question = "qui est SonGoku"
tavily_results = web_search_tool.invoke({"query": question})["results"]
joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
web_results = Document(page_content=joined_tavily_result)
print(type(Document))
print(Document)
print(type(web_results))
print(web_results)


# def web_search(state: GraphState) -> Dict[str, Any]:
#     print("---WEB SEARCH---")
#     question = state["question"]
#     documents = state["documents"]

#     #tavily_results = web_search_tool.invoke({"query": question})
#     tavily_results = web_search_tool.invoke({"query": question})['results']
#     joined_tavily_result = "\n".join(
#         [tavily_result["content"] for tavily_result in tavily_results]
#     )
#     web_results = Document(page_content=joined_tavily_result)
#     if documents is not None:
#         documents.append(web_results)
#     else:
#         documents = [web_results]
#     return {"documents": documents, "question": question}

# # def web_search(state: GraphState) -> Dict[str, Any]:
# #     print("---WEB SEARCH---")
# #     question = state["question"]
# #     documents = state["documents"]

# #     tavily_results = web_search_tool.invoke({"query": question})
# #     # Les résultats sont déjà des Documents, pas besoin de conversion
# #     if documents is not None:
# #         documents.extend(tavily_results)
# #     else:
# #         documents = tavily_results

# #     return {"documents": documents, "question": question}


# if __name__ == "__main__":
#     web_search(
#         state={
#             "question": "agent memory",
#             "documents": None,
#         }
#     )
