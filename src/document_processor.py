"""
Advanced document processing and chunking strategies
"""

from typing import List, Optional
from pathlib import Path
import logging

from langchain.docstore.document import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredWordDocumentLoader
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Advanced document processing with multiple chunking strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "recursive"
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking ('recursive', 'token', 'semantic')
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.text_splitter = self._get_text_splitter()
    
    def _get_text_splitter(self):
        """Get text splitter based on strategy"""
        if self.chunking_strategy == "recursive":
            # Best for most use cases - splits on paragraphs, then sentences
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
        elif self.chunking_strategy == "token":
            # Better for token-aware chunking
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.chunking_strategy == "semantic":
            # Semantic chunking using sentence transformers
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap,
                tokens_per_chunk=self.chunk_size
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {self.chunking_strategy}")
    
    def load_documents(self, source_path: str) -> List[Document]:
        """
        Load documents from file or directory
        
        Args:
            source_path: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        path = Path(source_path)
        documents = []
        
        try:
            if path.is_file():
                documents = self._load_single_file(str(path))
            elif path.is_dir():
                documents = self._load_directory(str(path))
            else:
                raise ValueError(f"Invalid path: {source_path}")
            
            logger.info(f"Loaded {len(documents)} documents from {source_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def _load_single_file(self, file_path: str) -> List[Document]:
        """Load a single file"""
        suffix = Path(file_path).suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path)
        elif suffix == ".txt":
            loader = TextLoader(file_path)
        elif suffix in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        return loader.load()
    
    def _load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported files from directory"""
        documents = []
        
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")
        
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process and chunk documents
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of chunked documents
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata for better retrieval
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise
    
    def process_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Process raw text into chunks
        
        Args:
            text: Raw text to process
            metadata: Optional metadata to attach
            
        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc_metadata = {"chunk_id": i, "chunk_size": len(chunk)}
            if metadata:
                doc_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        return documents

