import os
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader



CHROMA_PATH = "chroma_dynamic"

PROMPT_TEMPLATE = """
Answer using only the context.

Context:
{context}

Question:
{question}
"""


# ===== Load LLM + Embeddings =====
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=256,
    api_key="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" # feel free (it is free key anyway)
)


# ===== Global DB =====
db = None


# ===== Upload + Index Document =====
def upload_and_index(file):

    global db

    loader = PyPDFLoader(file.name)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )

    return f"Document Indexed Successfully ({len(chunks)} chunks)"


# ===== Ask Question =====
def ask_question(question):

    if db is None:
        return "Please upload a document first!", ""

    results = db.similarity_search_with_relevance_scores(question, k=3)

    if len(results) == 0:
        return "No relevant results found", ""

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _ in results]
    )

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    ).format(
        context=context_text,
        question=question
    )

    response = llm.invoke(prompt).content

    sources = [doc.metadata.get("source") for doc, _ in results]

    return response, str(sources)


# ===== Gradio UI =====
with gr.Blocks(title="Dynamic RAG Assistant") as app:

    gr.Markdown("# Dynamic RAG Assistant")

    with gr.Tab("Upload Document"):
        file_input = gr.File(label="Upload PDF")
        upload_btn = gr.Button("Index Document")
        upload_status = gr.Textbox( label="Upload Status")

        upload_btn.click(
            upload_and_index,
            inputs=file_input,
            outputs=upload_status
        )

    with gr.Tab("Chat with Document"):
        question_box = gr.Textbox(label="Ask Question")
        answer_box = gr.Textbox(label="Answer")
        source_box = gr.Textbox(label="Sources")

        ask_btn = gr.Button("Ask")

        ask_btn.click(
            ask_question,
            inputs=question_box,
            outputs=[answer_box, source_box]
        )


if __name__ == "__main__":
    app.launch(share=True, debug=True)