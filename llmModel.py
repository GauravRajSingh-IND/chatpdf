from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Any
import logging
from logging.handlers import RotatingFileHandler

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Configure and return a logger with file and console handlers."""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = RotatingFileHandler(
        log_file, maxBytes=1024 * 1024, backupCount=5
    )
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    # Add more document types as needed


@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 500
    chunk_overlap: int = 100
    temperature: float = 0.7
    chain_type: str = "stuff"


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass


class ReadDocuments:
    """A class to handle document reading, processing, and QA chain creation."""

    def __init__(
            self,
            document_path: str | Path,
            document_type: str = "pdf",
            document_password: Optional[str] = None,
            config: Optional[DocumentConfig] = None
    ):
        """
        Initialize the document reader.

        Args:
            document_path: Path to the document
            document_type: Type of document (default: "pdf")
            document_password: Password for protected documents
            config: Configuration for document processing
        """
        self.document_path = Path(document_path)
        self.document_type = DocumentType(document_type)
        self.document_password = document_password
        self.config = config or DocumentConfig()
        self.embeddings = OpenAIEmbeddings()

        self.document_content: Optional[List[Document]] = None
        self.document_chunks: Optional[List[Document]] = None
        self.vectorDB: Optional[Chroma] = None

        self.logger = setup_logger(
            'document_processor',
            'document_processing.log'
        )

        self._validate_initialization()

    def _validate_initialization(self) -> None:
        """Validate initialization parameters."""
        if not self.document_path.exists():
            raise FileNotFoundError(
                f"Document not found at {self.document_path}"
            )

    def load_document(self) -> List[Document]:
        """
        Load the document and return its content.

        Returns:
            List of Document objects

        Raises:
            DocumentProcessingError: If document loading fails
        """
        try:
            if self.document_type == DocumentType.PDF:
                loader = PyPDFLoader(str(self.document_path))
                self.document_content = loader.load()
                self.logger.info("Document loaded successfully")
                return self.document_content

            raise ValueError(f"Unsupported document type: {self.document_type}")

        except Exception as e:
            self.logger.error(f"Error loading document: {e}")
            raise DocumentProcessingError(f"Failed to load document: {e}")

    def split_documents(self) -> List[Document]:
        """
        Split the documents into smaller chunks.

        Returns:
            List of Document chunks

        Raises:
            DocumentProcessingError: If document splitting fails
        """
        if not self.document_content:
            raise DocumentProcessingError("No document content to split")

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            self.document_chunks = text_splitter.split_documents(
                self.document_content
            )
            self.logger.info(
                f"Document split into {len(self.document_chunks)} chunks"
            )
            return self.document_chunks

        except Exception as e:
            self.logger.error(f"Error splitting document: {e}")
            raise DocumentProcessingError(f"Failed to split document: {e}")

    def create_vectordb(self) -> Chroma:
        """
        Create and return a vector database from document chunks.

        Returns:
            Chroma vector store

        Raises:
            DocumentProcessingError: If vector database creation fails
        """
        if not self.document_chunks:
            raise DocumentProcessingError("No document chunks to vectorize")

        try:
            # Use in-memory mode by setting `persist_directory=None`
            self.vectorDB = Chroma.from_documents(
                documents=self.document_chunks,
                embedding=self.embeddings,
                persist_directory="chroma_persistence"  # In-memory mode
            )
            self.logger.info("Vector database created successfully")
            return self.vectorDB

        except Exception as e:
            self.logger.error(f"Error creating vector database: {e}")
            raise DocumentProcessingError(
                f"Failed to create vector database: {e}"
            )

    def create_qa_chain(self) -> RetrievalQA:
        """
        Create and return a question-answering chain.

        Returns:
            RetrievalQA chain

        Raises:
            DocumentProcessingError: If chain creation fails
        """
        try:
            # Process document pipeline
            self.load_document()
            self.split_documents()
            self.create_vectordb()

            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=OpenAI(temperature=self.config.temperature),
                chain_type=self.config.chain_type,
                retriever=self.vectorDB.as_retriever(),
                return_source_documents=True
            )

            self.logger.info("QA chain created successfully")
            return qa

        except Exception as e:
            self.logger.error(f"Error creating QA chain: {e}")
            raise DocumentProcessingError(f"Failed to create QA chain: {e}")


def main():
    """Example usage of the ReadDocuments class."""
    try:
        config = DocumentConfig(
            chunk_size=500,
            chunk_overlap=100,
            temperature=0.7
        )

        processor = ReadDocuments(
            document_path="Docs/Fair-Work-Handbook.pdf",
            config=config
        )

        qa_chain = processor.create_qa_chain()
        print("Document processing completed successfully")
        return qa_chain

    except Exception as e:
        logging.error(f"Application error: {e}")
        raise