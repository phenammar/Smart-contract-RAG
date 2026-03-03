# This file is just for testing

#Imports
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os


CHROMA_PATH = "chroma_db"

PROMPT_TEMPLATE = """
You are a helpful assistant.

Answer only using the context below.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""

def main():
    
    # Parse the query text from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str)  # variable name -> query_text : string     
    args = parser.parse_args()

    query_text = args.query_text

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Load the existing Chroma database
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    # Perform similarity search to retrieve relevant documents
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _ in results]
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=256,
        api_key="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"  # feel free (it is free key anyway)
    )

    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("source") for doc, _ in results]

    print("Response:", response_text)
    print("Sources:", sources)


if __name__ == "__main__":
    main()