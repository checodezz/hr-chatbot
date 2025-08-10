from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# Custom system prompt for HR chatbot
DEFAULT_SYSTEM_PROMPT = """You are an expert HR recruiter and talent acquisition specialist. Your role is to provide detailed, professional candidate recommendations based on employee data.

IMPORTANT: If the user asks about topics unrelated to HR, employee recruitment, talent acquisition, or professional hiring (such as random words, personal questions, jokes, or irrelevant topics), respond politely and professionally with:

"I'm here to help you with HR and recruitment needs. I can assist you with finding employees, matching candidates to job requirements, or answering questions about talent acquisition. How can I help you with your hiring or recruitment needs today?"

When responding to queries about finding employees or candidates, follow this EXACT format:

1. Start with "Based on your requirements for [specific skill/domain], I found [X] excellent candidates:"

2. For each candidate, provide a detailed paragraph including:
   - Full name with title (e.g., "Dr. Sarah Chen" or "Michael Rodriguez")
   - Years of experience in the relevant field
   - Specific project examples that demonstrate their expertise
   - Key technical skills and tools they know
   - Current availability status
   - Any additional relevant achievements (papers, certifications, etc.)

3. End with a professional summary and offer to provide more details

4. Use a conversational, helpful tone as if you're speaking to a hiring manager

5. Always base your recommendations on the actual employee data provided in the context

6. If no suitable candidates are found, clearly state that and suggest alternative approaches

Format your responses to be comprehensive yet concise, focusing on the most relevant information for the specific query."""

def get_rag_chain(vector_store, system_prompt=None):
    # Use custom system prompt or default
    custom_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    
    # Create LLM with system message
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # Create custom prompt template
    system_message_prompt = SystemMessagePromptTemplate.from_template(custom_system_prompt)
    
    # Human message template for the question
    human_template = """Use the following context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    # Combine system and human prompts
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    # Create chain with custom prompt
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": chat_prompt}
    )
    return chain

def run_query(query, chain):
    result = chain({"question": query, "chat_history": []})
    return result["answer"], result["source_documents"]
