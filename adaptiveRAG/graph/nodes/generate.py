from typing import Any, Dict

from graph.chains.generation import generation_chain


def generate(state) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "question": question, "documents": documents}
