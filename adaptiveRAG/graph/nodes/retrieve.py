from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("Retrieving documents based on the question...")

    question = state["question"]  # extracting the question from the current state

    documents = retriever.invoke(
        question
    )  # using the retriever to get relevant documents by semantic search

    # Update the state with the retrieved documents and just in case the question
    return {"documents": documents, "question": question}
