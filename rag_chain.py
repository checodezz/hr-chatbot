from langchain.chains import ConversationalRetrievalChain
# from langchain.llms import OpenAI

from langchain_openai import ChatOpenAI

def get_rag_chain(vector_store):
    # Use ChatOpenAI instead of OpenAI
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )
    return chain

def run_query(query, chain):
    result = chain({"question": query, "chat_history": []})
    return result["answer"], result["source_documents"]
