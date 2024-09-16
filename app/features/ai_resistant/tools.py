from typing import List, Tuple, Dict, Any

import os
import json
import time

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.exceptions import OutputParserException


from app.services.logger import setup_logger
from app.services.tool_registry import ToolFile
from app.api.error_utilities import LoaderError

# from quizzify
from typing import List, Tuple, Dict, Any
#from io import BytesIO
#from fastapi import UploadFile
#from pypdf import PdfReader
from urllib.parse import urlparse
import requests
import os
import json
import time

from langchain_core.documents import Document
#from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, TextLoader
#from langchain.document_loaders import TextLoader
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import GooglePalmEmbeddings
# from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError


logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

# Base Loader class
class LangChainBaseLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def create_loader(self):
        return None
    
    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        try:
            loader = self.create_loader(self.file_path)
            loaded_documents = loader.load()
            # Clean the content of each document
            for doc in loaded_documents:
                doc.page_content = self.clean_text(doc.page_content)
                documents.append(doc)
        except Exception as e:
            logger.error(f"Error processing file {self.file_path}: {e}")
        return documents

class LangChainPDFLoader(LangChainBaseLoader):
    def create_loader(self):
        return PyPDFLoader(self.file_path)
class LangChainDocxLoader(LangChainBaseLoader):
    def create_loader(self):
        return UnstructuredWordDocumentLoader(self.file_path, mode = 'single')
class LangChainPPTLoader(LangChainBaseLoader):
    def create_loader(self):
        return UnstructuredPowerPointLoader(self.file_path, mode = 'single')
class LangChainTextLoader(LangChainBaseLoader):
    def create_loader(self):
        return TextLoader(self.file_path, encoding='utf-8')


# URLLoader
class URLLoader:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.loaders = {
            'pdf': LangChainPDFLoader,
            'docx': LangChainDocxLoader,
            'pptx': LangChainPPTLoader,
            'txt': LangChainTextLoader
        }

    def load(self, tool_file: ToolFile) -> List[Document]:
        documents = []
        url = tool_file.url
        tmp_file_path = None  # Initialize here for scope
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to download file from {url} with status code {response.status_code}")
                return []

            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)
            file_type = file_name.split(".")[-1].lower() if "." in file_name else None

            if file_type not in self.loaders:
                logger.warning(f"Unsupported file type for URL: {url}")
                return []

            # Create a temporary file to save the downloaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_type) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            if self.verbose:
                logger.info(f"Downloaded and saved file from {url} to {tmp_file_path}")

            # Load the document using the appropriate loader
            loader_class = self.loaders[file_type]
            loader = loader_class(tmp_file_path)
            documents = loader.load()

            if self.verbose:
                logger.info(f"Loader type used: {type(loader)}")
                logger.info(f"Successfully loaded file from {url}")

        except Exception as e:
            logger.error(f"Failed to process file from {url}: {e}")
            return []
        finally:
            # Ensure temporary file is deleted
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        return documents

class RAGRunnable:
    def __init__(self, func):
        self.func = func
    
    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # Result of previous function is passed as first argument to next function
            return other(self.func(*args, **kwargs))
        return RAGRunnable(chained_func)
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class RAGpipeline:
    def __init__(self, loader=None, splitter=None, vectorstore_class=None, embedding_model=None, verbose=False):
        default_config = {
            "loader": URLLoader(verbose = verbose), # Creates instance on call with verbosity
            "splitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
            "vectorstore_class": Chroma,
            "embedding_model": GooglePalmEmbeddings(model_name='models/embedding-gecko-001')
        }
        self.loader = loader or default_config["loader"]
        self.splitter = splitter or default_config["splitter"]
        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        self.verbose = verbose

    def load_file(self, tool_file: ToolFile) -> List[Document]:
        if self.verbose:
            logger.info(f"Loading {tool_file.filename}")
        
        try:
            documents = self.loader.load(tool_file)
        except Exception as e:
            logger.error(f"An error occurred while loading the file {tool_file.filename}: {e}")
            documents = []
            
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.info(f"List of documents for splitting is empty")
            return []
        if self.verbose:
            logger.info(f"Splitting {len(documents)} documents")
            logger.info(f"Splitter type used: {type(self.splitter)}")
        
            
        chunked_documents = self.splitter.split_documents(documents)
        
        if self.verbose:
            logger.info(f"Split {len(documents)} documents into {len(chunked_documents)} chunks")
        
        return chunked_documents
    
    def create_vectorstore(self, documents: List[Document]):
        if not documents:
            logger.info(f"List of documents chunks for vectorstore creation is empty")
            return None
        if self.verbose:
            logger.info(f"Creating vectorstore from {len(documents)} documents chunks")
        vectorstore = self.vectorstore_class.from_documents(documents[:25], self.embedding_model)

        if self.verbose:
            logger.info(f"Vectorstore created")
        return vectorstore
    
    def compile(self):
        # Compile the pipeline
        self.load_file = RAGRunnable(self.load_file)
        self.split_documents = RAGRunnable(self.split_documents)
        self.create_vectorstore = RAGRunnable(self.create_vectorstore)
        if self.verbose:
            logger.info(f"Completed pipeline compilation")
    
    def __call__(self, tool_file: ToolFile):
        # Returns a vectorstore ready for usage 
        
        if self.verbose: 
            logger.info(f"Executing pipeline")
            logger.info(f"Start of Pipeline received: {tool_file.filename}")
        
        pipeline = self.load_file | self.split_documents | self.create_vectorstore
        return pipeline(tool_file)


class AIResistant():
    def __init__(self, vectorstore, model = None, prompt = None):
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7),
            "prompt": read_text_file('prompts/ai_resistant.txt')
        }
        self.model = model or default_config["model"]
        self.prompt = prompt or default_config["prompt"]
        
        if vectorstore is None:
            raise ValueError("Vectorstore must be provided")
        self.vectorstore = vectorstore
        retriever = self.vectorstore.as_retriever()


