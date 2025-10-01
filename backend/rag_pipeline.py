from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


def load_rag():
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory="../chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

def ask_support_assistant(query):
    qa = load_rag()
    response = qa.run(query)
    return response

if __name__ == "__main__":
    print(ask_support_assistant("How to fix a database timeout error?"))
