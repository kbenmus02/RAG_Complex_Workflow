from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Attributes:
        question: question
        generation: LLM generation
        web_search:whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]


# Example usage
# state: GraphState = {
#    "question": "Quelle est la capitale de la France ?",
#    "generation": "La capitale est Paris.",
#    "web_search": True,
#    "documents": ["doc1", "doc2"]
# }
