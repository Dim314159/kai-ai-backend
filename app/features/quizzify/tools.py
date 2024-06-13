from typing import List, Tuple, Dict, Any
from io import BytesIO
from fastapi import UploadFile
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
from langchain_core.pydantic_v1 import BaseModel, Field

from services.logger import setup_logger
from services.tool_registry import ToolFile
from api.error_utilities import LoaderError

relative_path = "features/quzzify"

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

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

class LangChainPDFLoader:
    def __init__(self, files: List[str]):
        self.files = files
    
    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = PyPDFLoader(file_path)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        return documents

class LangChainDocxLoader:
    def __init__(self, files: List[str], mode: str = "single"):
        self.files = files
        self.mode = mode

    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = UnstructuredWordDocumentLoader(file_path, mode=self.mode)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        return documents

class LangChainPPTLoader:
    def __init__(self, files: List[str], mode: str = "single"):
        self.files = files
        self.mode = mode

    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = UnstructuredPowerPointLoader(file_path, mode=self.mode)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        return documents

class LangChainCSVLoader:
    def __init__(self, files: List[str], mode: str = "elements"):
        self.files = files
        self.mode = mode

    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = UnstructuredCSVLoader(file_path, mode=self.mode)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        return documents

class LangChainURLLoader:
    def __init__(self, files: List[str], mode: str = "single"):
        self.files = files
        self.mode = mode

    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = UnstructuredURLLoader([file_path], mode=self.mode)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing url {file_path}: {e}")
        return documents
    
class LangChainYouTubeLoader:
    def __init__(self, files: List[str]):
        self.files = files

    def clean_text(self, text: str) -> str:
        # Replace line breaks with spaces, but ensure sentence integrity
        lines = text.split('\n')
        cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
        return cleaned_text

    def load(self) -> List[Document]:
        documents = []
        for file_path in self.files:
            try:
                loader = YoutubeLoader.from_youtube_url(file_path)
                loaded_documents = loader.load()
                # Clean the content of each document
                for doc in loaded_documents:
                    doc.page_content = self.clean_text(doc.page_content)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing youtube link {file_path}: {e}")
        return documents



# Dmitri's URLLoader

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
                    loader = loader_class(urls)
                    documents.extend(loader.load())
                    if self.verbose:
                        logger.info(f"Loaded {len(documents)} documents from {file_type} files")

        if not any_success:
            raise LoaderError("Unable to load any files from URLs")

        return documents

class RAGpipeline:
    def __init__(self, loader=None, splitter=None, vectorstore_class=None, embedding_model=None, verbose=False):
        default_config = {
            "loader": URLLoader(verbose = verbose), # Creates instance on call with verbosity
            "splitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
            "vectorstore_class": Chroma,
            "embedding_model": VertexAIEmbeddings(model='textembedding-gecko')
        }
        self.loader = loader or default_config["loader"]
        self.splitter = splitter or default_config["splitter"]
        self.vectorstore_class = vectorstore_class or default_config["vectorstore_class"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        self.verbose = verbose

    def load_files(self, files) -> List[Document]:
        if self.verbose:
            logger.info(f"Loading {len(files)} files")
            logger.info(f"Loader type used: {type(self.loader)}")
        
        logger.debug(f"Loader is a: {type(self.loader)}")
        
        try:
            total_loaded_files = self.loader.load(files)
        except LoaderError as e:
            logger.error(f"Loader experienced error: {e}")
            raise LoaderError(e)
            
        return total_loaded_files
    
    def split_loaded_documents(self, loaded_documents: List[Document]) -> List[Document]:
        if self.verbose:
            logger.info(f"Splitting {len(loaded_documents)} documents")
            logger.info(f"Splitter type used: {type(self.splitter)}")
            
        total_chunks = []
        chunks = self.splitter.split_documents(loaded_documents)
        total_chunks.extend(chunks)
        
        if self.verbose: logger.info(f"Split {len(loaded_documents)} documents into {len(total_chunks)} chunks")
        
        return total_chunks
    
    def create_vectorstore(self, documents: List[Document]):
        if self.verbose:
            logger.info(f"Creating vectorstore from {len(documents)} documents")
        print(f'Document type: {type(documents[0])}')
        self.vectorstore = self.vectorstore_class.from_documents(documents, self.embedding_model)

        if self.verbose: logger.info(f"Vectorstore created")
        return self.vectorstore
    
    def compile(self):
        # Compile the pipeline
        self.load_files = RAGRunnable(self.load_files)
        self.split_loaded_documents = RAGRunnable(self.split_loaded_documents)
        self.create_vectorstore = RAGRunnable(self.create_vectorstore)
        if self.verbose: logger.info(f"Completed pipeline compilation")
    
    def __call__(self, documents):
        # Returns a vectorstore ready for usage 
        
        if self.verbose: 
            logger.info(f"Executing pipeline")
            logger.info(f"Start of Pipeline received: {len(documents)} documents of type {type(documents[0])}")
        
        pipeline = self.load_files | self.split_loaded_documents | self.create_vectorstore
        return pipeline(documents)


class QuizBuilder:
    def __init__(self, vectorstore, topic, prompt=None, model=None, parser=None, verbose=False):
        default_config = {
            "model": VertexAI(model="gemini-1.0-pro"),
            "parser": JsonOutputParser(pydantic_object=QuizQuestion),
            "prompt": read_text_file("prompt/quizzify-prompt.txt")
        }
        
        self.prompt = prompt or default_config["prompt"]
        self.model = model or default_config["model"]
        self.parser = parser or default_config["parser"]
        
        self.vectorstore = vectorstore
        self.topic = topic
        self.verbose = verbose
        
        if vectorstore is None: raise ValueError("Vectorstore must be provided")
        if topic is None: raise ValueError("Topic must be provided")
    
    def compile(self):
        # Return the chain
        prompt = PromptTemplate(
            template=self.prompt,
            input_variables=["topic"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        retriever = self.vectorstore.as_retriever()
        
        runner = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        
        chain = runner | prompt | self.model | self.parser
        
        if self.verbose: logger.info(f"Chain compilation complete")
        
        return chain

    def validate_response(self, response: Dict) -> bool:
        try:
            # Assuming the response is already a dictionary
            if isinstance(response, dict):
                if 'question' in response and 'choices' in response and 'answer' in response and 'explanation' in response:
                    choices = response['choices']
                    if isinstance(choices, dict):
                        for key, value in choices.items():
                            if not isinstance(key, str) or not isinstance(value, str):
                                return False
                        return True
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during response validation: {e}")
            return False

    def format_choices(self, choices: Dict[str, str]) -> List[Dict[str, str]]:
        return [{"key": k, "value": v} for k, v in choices.items()]
    
    def create_questions(self, num_questions: int = 5) -> List[Dict]:
        if self.verbose: logger.info(f"Creating {num_questions} questions")
        
        if num_questions > 10:
            return {"message": "error", "data": "Number of questions cannot exceed 10"}
        
        chain = self.compile()
        
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 5  # Allow for more attempts to generate questions

        while len(generated_questions) < num_questions and attempts < max_attempts:
            response = chain.invoke(self.topic)
            if self.verbose:
                logger.info(f"Generated response attempt {attempts + 1}: {response}")
            
            # Directly check if the response format is valid
            if self.validate_response(response):
                response["choices"] = self.format_choices(response["choices"])
                generated_questions.append(response)
                if self.verbose:
                    logger.info(f"Valid question added: {response}")
                    logger.info(f"Total generated questions: {len(generated_questions)}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid response format. Attempt {attempts + 1} of {max_attempts}")
            
            # Move to the next attempt regardless of success to ensure progress
            attempts += 1

        # Log if fewer questions are generated
        if len(generated_questions) < num_questions:
            logger.warning(f"Only generated {len(generated_questions)} out of {num_questions} requested questions")
        
        if self.verbose: logger.info(f"Deleting vectorstore")
        self.vectorstore.delete_collection()
        
        # Return the list of questions
        return generated_questions[:num_questions]

class QuestionChoice(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, D, etc.")
    value: str = Field(description="The text content of the choice")
class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoice] = Field(description="A list of choices")
    answer: str = Field(description="The correct answer")
    explanation: str = Field(description="An explanation of why the answer is correct")
