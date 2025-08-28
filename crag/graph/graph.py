from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
# from graph.nodes import *
from graph.state import GraphState

load_dotenv()


# decide which node are we going next
# def decide_to_generate(state: GraphState) -> str:
#     print("---ASSESS GRADED DOCUMENTS---")
#     if state["web_search"]:
#         print("---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVENT TO QUESTION")
#         return WEB_SEARCH
#     else:
#         print("---DECISION: ALL DOCUMENTS ARE RELEVENT TO QUESTION (GENERATE)")
#         return GENERATE


def decide_to_generate(state: GraphState) -> str:
    return "NO" if state["web_search"] else "YES"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    # GRADE_DOCUMENTS, decide_to_generate, {GENERATE: GENERATE, WEB_SEARCH: WEB_SEARCH}
    GRADE_DOCUMENTS, decide_to_generate, {"YES": GENERATE, "NO": WEB_SEARCH}
)
workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)
app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph_crag_test.png")
