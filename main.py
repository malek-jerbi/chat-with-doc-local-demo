import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

if __name__ == "__main__":
    print("Hello World!")
    pdf_path = "C:\\Users\\malek\\Documents\\python small programs\\vectorstore-in-memory\\2210.03629.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vectorestore= FAISS.from_documents(docs, embeddings)
    vectorestore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), chain_type="stuff", retriever = new_vectorstore.as_retriever(), return_source_documents=True)
    res = qa({"query": "Explain to me React like i'm five"})
    print(res["result"])