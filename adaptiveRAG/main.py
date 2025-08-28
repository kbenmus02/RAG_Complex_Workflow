from dotenv import load_dotenv

load_dotenv()
from graph.graph import app

if __name__ == "__main__":
    print("Hello Corrective RAG")
    # print(app.invoke(input={"question": "what is agent memory?"}))
    print(app.invoke(input={"question": "Who is Baggio R?"}))
    