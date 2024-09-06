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
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredCSVLoader
#import pandas
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import YoutubeLoader


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError




logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()


class AIResistant():
    def __init__(self) -> None:
        pass


# Base Loader class
class LangChainBaseLoader:
    def __init__(self, files: List[str]):
        self.files = files
    
    def create_loader(self, file_path):
        return None
    
    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = self.create_loader(file_path)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        return documents

class LangChainPDFLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return PyPDFLoader(file_path)
class LangChainDocxLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return UnstructuredWordDocumentLoader(file_path, mode = 'single')
class LangChainPPTLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return UnstructuredPowerPointLoader(file_path, mode = 'single')
class LangChainCSVLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return UnstructuredCSVLoader(file_path, mode = 'elements')
class LangChainURLLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return UnstructuredURLLoader([file_path], mode = 'single')
class LangChainYouTubeLoader(LangChainBaseLoader):
    def create_loader(self, file_path):
        return YoutubeLoader.from_youtube_url(file_path)

# URLLoader

class URLLoader:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.loaders = {
            'pdf': LangChainPDFLoader,
            'docx': LangChainDocxLoader,
            'pptx': LangChainPPTLoader,
            'csv': LangChainCSVLoader,
            'url': LangChainURLLoader,
            'youtube': LangChainYouTubeLoader
        }

    def load(self, tool_files: List[ToolFile]) -> List[Document]:
        file_dict = {key: [] for key in self.loaders.keys()}
        documents = []
        any_success = False

        for tool_file in tool_files:
            try:
                url = tool_file.url
                response = requests.get(url)
                parsed_url = urlparse(url)
                path = parsed_url.path
                file_type = path.split(".")[-1].lower() if "." in path else None

                if response.status_code == 200:
                    if file_type in self.loaders:
                        file_dict[file_type].append(url)
                    elif 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
                        file_dict["youtube"].append(url)
                    else:
                        file_dict["url"].append(url)

                    if self.verbose:
                        logger.info(f"Successfully loaded file from {url}")

                    any_success = True
                else:
                    logger.error(f"Request failed to load file from {url} with status code {response.status_code}")

            except Exception as e:
                logger.error(f"Failed to load file from {url}")
                logger.error(e)
                continue

        if any_success:
            for file_type, urls in file_dict.items():
                if urls:
                    loader_class = self.loaders[file_type]
                    logger.debug(f"Loader file type used: {loader_class}")
                    loader = loader_class(urls)
                    documents.extend(loader.load())
                    if self.verbose:
                        logger.info(f"Loaded {len(documents)} documents from {file_type} files")

        if not any_success:
            raise LoaderError("Unable to load any files from URLs")

        return documents