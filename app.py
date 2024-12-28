import streamlit as st
from pathlib import Path
from llmModel import ReadDocuments, DocumentConfig, DocumentProcessingError

# Streamlit App Configuration
st.set_page_config(
    page_title="Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main App Function
def main():
    st.title("📄 Document Question Answering System")
    st.markdown("Upload a document and ask questions!")

    # Sidebar Configuration
    st.sidebar.header("Document Configuration")
    chunk_size = st.sidebar.slider("Chunk Size", 100, 2000, 500, step=50)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 500, 100, step=50)
    temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.7, step=0.1)
    chain_type = st.sidebar.selectbox("Chain Type", ["stuff", "map_reduce", "refine"], index=0)

    # Upload Section
    uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])
    if uploaded_file:
        # Save the uploaded file temporarily
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully.")

        # Initialize Processor
        config = DocumentConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temperature=temperature,
            chain_type=chain_type
        )

        try:
            processor = ReadDocuments(document_path=temp_path, config=config)

            # Process Document
            qa_chain = processor.create_qa_chain()
            st.success("Document processing completed successfully. You can now ask questions!")

            # Question Section
            question = st.text_input("Enter your question:")
            if question:
                with st.spinner("Retrieving answer..."):
                    response = qa_chain({"query": question})  # Use the QA chain object as a callable

                    # Extract the result and source documents
                    answer = response.get("result", "No answer found.")
                    source_docs = response.get("source_documents", [])

                    st.subheader("Answer:")
                    st.write(answer)

                    st.subheader("Source Documents:")
                    if source_docs:
                        for doc in source_docs:
                            page_number = doc.metadata.get("page_number", "Unknown")
                            snippet = doc.page_content[:1000]  # Show snippet of source
                            st.write(f"Page: {page_number}")
                            st.write(snippet + "...")
                    else:
                        st.write("No source documents found.")


        except DocumentProcessingError as e:
            st.error(f"Document processing error: {e}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

# Run the App
if __name__ == "__main__":
    main()
