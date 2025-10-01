from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import ask_support_assistant

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/resolve_ticket")
def resolve_ticket(query: Query):
    response = ask_support_assistant(query.query)
    return {"answer": response}
