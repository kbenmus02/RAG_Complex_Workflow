from dotenv import load_dotenv

load_dotenv()
from pprint import pprint

from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever

# def test_foo() -> None:
#    assert 1==1


def test_retrieval_grader_with_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content if docs else "No documents found."
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_with_answer_no() -> None:
    question = "Goku story"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content if docs else "No documents found."
    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
