from typing import List, Tuple, Dict, Any

import os
#import json
#import time

from langchain_core.prompts import PromptTemplate
#from langchain_core.output_parsers import JsonOutputParser
#from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_google_genai import GoogleGenerativeAI
#from langchain_core.exceptions import OutputParserException


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

#from langchain_core.documents import Document
#from langchain.schema import Document

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader, TextLoader
#from langchain.document_loaders import TextLoader
import tempfile

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain.embeddings import GooglePalmEmbeddings
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

    def load(self) -> str:
        text = ''
        try:
            loader = self.create_loader()
            loaded_documents = loader.load()
            for doc in loaded_documents:
                text += self.clean_text(doc.page_content)
        except Exception as e:
            logger.error(f"Error processing file {self.file_path}: {e}")
        return text

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

    def load(self, tool_file) -> str:
        text = ''
        url = tool_file['url']
        tmp_file_path = None  # Initialize here for scope
        try:
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Failed to download file from {url} with status code {response.status_code}")
                return ''

            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)
            file_type = file_name.split(".")[-1].lower() if "." in file_name else None

            if file_type not in self.loaders:
                logger.warning(f"Unsupported file type for URL: {url}")
                return ''

            # Create a temporary file to save the downloaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.' + file_type) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            if self.verbose:
                logger.info(f"Downloaded and saved file from {url} to {tmp_file_path}")

            # Load the document using the appropriate loader
            loader_class = self.loaders[file_type]
            loader = loader_class(tmp_file_path)
            text = loader.load()

            if self.verbose:
                logger.info(f"Loader type used: {type(loader)}")
                logger.info(f"Successfully loaded file from {url}")

        except Exception as e:
            logger.error(f"Failed to process file from {url}: {e}")
            return ''
        finally:
            # Ensure temporary file is deleted
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                if self.verbose:
                    logger.info(f"Temporary file {tmp_file_path} deleted")
        return text

class AIResistant():
    def __init__(self, tool_file, loader = None, model = None, prompt_template = None, verbose = False):
        self.tool_file = tool_file
        default_config = {
            "loader": URLLoader(verbose = verbose),
            "model": GoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7),
            "prompt_template": read_text_file('prompts/ai_resistant_prompt.txt')
        }
        self.loader = loader or default_config["loader"]
        self.model = model or default_config["model"]
        self.prompt_template = prompt_template or default_config["prompt_template"]
        self.verbose = verbose

    def generate_suggestions(self):
        assignment_text = self.loader.load(self.tool_file)
        if not assignment_text:
            return ''
        # Format the prompt with the assignment text

        # prompt = PromptTemplate(
        #     template=self.prompt_template,
        #     input_variables=["assignment_text"]
        # )
        # chain = prompt | self.model

        # try:
        #     # Generate suggestions using the AI model
        #     response = chain.invoke({"assignment_text": assignment_text})
        #     return response.content
        # except Exception as e:
        #     if self.verbose:
        #         logger.error(f"Error generating suggestions: {e}")
        #     return "An error occurred while generating suggestions."