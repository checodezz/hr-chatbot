# Custom System Prompts for HR RAG Pipeline

This document shows you how to customize the system prompt to control the LLM's output format and behavior.

## 🎯 Default System Prompt

The default system prompt is designed for professional HR recruitment responses:

```python
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
```

## 🚀 How to Use Custom System Prompts

### Method 1: Using the `/query` endpoint with system_prompt parameter

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find ML experts for healthcare",
    "system_prompt": "You are an expert HR recruiter. When recommending candidates, use this EXACT format: Start with \"Based on your requirements for [skill] expertise in [domain], I found [X] excellent candidates:\" Then for each candidate, write a detailed paragraph mentioning their name, years of experience, specific healthcare projects they worked on, their technical skills, current availability, and any relevant achievements like published papers. End with a professional summary and offer to provide more details about their healthcare projects or availability for meetings."
  }'
```

### Method 2: Using the `/query/custom-prompt` endpoint

```bash
curl -X POST "http://localhost:8000/query/custom-prompt" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find ML experts for healthcare",
    "system_prompt": "Your custom system prompt here"
  }'
```

## 📝 Example System Prompts

### 1. Detailed Healthcare Recruitment Format

```json
{
  "system_prompt": "You are an expert HR recruiter specializing in healthcare technology. When recommending candidates, use this EXACT format: Start with \"Based on your requirements for [skill] expertise in [domain], I found [X] excellent candidates:\" Then for each candidate, write a detailed paragraph mentioning their name, years of experience, specific healthcare projects they worked on, their technical skills, current availability, and any relevant achievements like published papers. End with a professional summary and offer to provide more details about their healthcare projects or availability for meetings."
}
```

### 2. Technical Skills Focus

```json
{
  "system_prompt": "You are a technical recruiter. When recommending candidates, focus on their technical skills and project experience. Format your response as: 1) Brief introduction with candidate count, 2) For each candidate: Name, Years of Experience, Key Technical Skills, Notable Projects, Current Availability. Keep responses concise and technical."
}
```

### 3. Executive Summary Format

```json
{
  "system_prompt": "You are an executive recruiter. Provide high-level summaries of candidates suitable for leadership roles. Include: Name, Years of Experience, Leadership Experience, Key Achievements, Current Role, Availability. Format as a brief executive summary suitable for C-level review."
}
```

### 4. Project-Specific Matching

```json
{
  "system_prompt": "You are a project manager looking for team members. When recommending candidates, focus on their specific project experience that matches the requirements. Include: Name, Relevant Project Experience, Technical Skills for the Project, Team Collaboration Experience, Availability for Project Timeline. Format as project-specific recommendations."
}
```

## 🎨 Customizing Response Styles

### Professional Tone

```
"You are a professional HR consultant. Use formal language and business terminology. Structure responses with clear sections and bullet points where appropriate."
```

### Conversational Tone

```
"You are a friendly HR advisor. Use conversational language and be encouraging. Make the hiring manager feel confident about their choices."
```

### Technical Focus

```
"You are a technical recruiter. Focus on technical skills, tools, and technologies. Use technical terminology and emphasize hands-on experience."
```

### Executive Focus

```
"You are an executive recruiter. Focus on leadership, strategic thinking, and business impact. Emphasize achievements and career progression."
```

## 🔧 Implementation Details

### In `rag_chain.py`:

```python
def get_rag_chain(vector_store, system_prompt=None):
    # Use custom system prompt or default
    custom_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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
```

### In `main.py`:

```python
@app.post("/query", response_model=QueryResponse)
async def query_employees(request: QueryRequest):
    try:
        # Create RAG chain with custom system prompt if provided
        if request.system_prompt:
            custom_rag_chain = get_rag_chain(vector_store, system_prompt=request.system_prompt)
            llm_answer, source_docs = run_query(request.query, custom_rag_chain)
        else:
            # Use default RAG chain
            llm_answer, source_docs = run_query(request.query, rag_chain)

        # ... rest of the function
```

## 🎯 Best Practices

1. **Be Specific**: Clearly define the desired output format
2. **Include Examples**: Reference the exact format you want
3. **Set the Tone**: Specify the communication style
4. **Focus on Context**: Emphasize what information to prioritize
5. **Test Iteratively**: Try different prompts to find the best format

## 🤝 Polite Response Handling

The system is designed to handle irrelevant or off-topic queries gracefully:

### Default Behavior for Irrelevant Queries

When users type random words, ask personal questions, or make irrelevant requests, the system responds with:

> "I'm here to help you with HR and recruitment needs. I can assist you with finding employees, matching candidates to job requirements, or answering questions about talent acquisition. How can I help you with your hiring or recruitment needs today?"

### Examples of Irrelevant Queries That Get Polite Responses:

- Random words: "banana", "xyz123", "hello world"
- Personal questions: "What's your favorite color?", "How old are you?"
- Jokes or casual conversation: "Tell me a joke", "What's the weather like?"
- Unrelated topics: "How to cook pasta", "What's the capital of France?"

### Customizing Polite Responses

You can customize the polite response by modifying the system prompt:

```json
{
  "system_prompt": "You are an HR assistant. For irrelevant queries, respond with: 'I'm focused on helping with HR and recruitment. How can I assist you with finding candidates or managing your hiring process?'"
}
```

## 🚀 Example Response

With the healthcare-focused system prompt, you get responses like:

```
"Based on your requirements for ML expertise in healthcare, I found 2 excellent candidates:

Dr. Sarah Chen would be perfect for this role. She has 6 years of ML experience and specifically worked on the 'Medical Diagnosis Platform' project where she implemented computer vision for X-ray analysis. Her skills include TensorFlow, PyTorch, and medical data processing. She's currently available and has published 3 papers on healthcare AI.

Michael Rodriguez is another strong candidate with 4 years of ML experience. He built the 'Patient Risk Prediction System' using ensemble methods and has experience with HIPAA compliance for healthcare data. He knows scikit-learn, pandas, and has worked with electronic health records.

Both have the technical depth and domain expertise you need. Would you like me to provide more details about their specific healthcare projects or check their availability for meetings?"
```
