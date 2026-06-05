from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    chunks = load_and_chunk("data/sample.pdf")
    page_count = chunks[0].metadata["total_pages"] if chunks else 0

    print(f"page count: {page_count}")
    print(f"chunk count: {len(chunks)}")
    print(f"first chunk text: {chunks[0].page_content}")
    print(f"first chunk metadata: {chunks[0].metadata}")
