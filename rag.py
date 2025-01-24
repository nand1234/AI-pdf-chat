from langchain_huggingface import HuggingFaceEndpoint
from embedding import get_context


def run_rag(question, file_name):
    # Step 6: Use a HuggingFace model to answer the question based on the retrieved documents
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # Example model; replace with your desired model
        task="text-generation",
        max_new_tokens=4096,
        do_sample=False,
        repetition_penalty=1.03,
    )

    # Generate an answer using the documents retrieved from the vector store
    docs = get_context(question, file_name)  # Get the relevant document text
    context = docs #
    print(context)

    # Construct the system and user prompt with clear instructions
    # Construct the system and user prompts
    system_prompt = """
    You are a helpful and question answer precise assistant. Use the provided context only to answer the user's question accurately. Now, based on the context and question provided, 
    respond with a concise and accurate answer. Stick to the context, avoid guessing. if you don't know answer say, sorry, i don't have answer.
    """

    user_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Combine the system and user prompts
    formatted_input = f"System: {system_prompt}\nUser: {user_prompt}"

    # Call the model with the formatted input
    return llm.invoke(formatted_input)


if __name__ == '__main__':
        answer = run_rag("what is Self-Reflection be short in 20 words?")
        print(f"output: {answer}")
