from typing import List, Tuple, Dict, Any
#from io import BytesIO
#from fastapi import UploadFile
#from pypdf import PdfReader
#from urllib.parse import urlparse
#import requests
import os
import json
import time

#from langchain_core.documents import Document
#from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_google_genai import GoogleGenerativeAI
#from langchain_google_genai import GoogleGenerativeAIEmbeddings

from services.logger import setup_logger



logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

class Syllabus:
    def __init__(self, model, attributes, prompt_template, parser, verbose = False):
        # prompt attributes
        # self.grade_level = attributes['grade_level']
        # self.course_title = attributes['course_title']
        # self.course_description = attributes['course_description']
        # self.objectives_topics = attributes['objectives_topics']
        # self.required_materials = attributes['required_materials']
        # self.num_weeks = attributes['num_weeks'] # integer
        # self.course_outline = attributes['course_outline']
        # self.grading_policy = attributes['grading_policy']
        # self.class_policy = attributes['class_policy']
        # self.customization = attributes['customization']
        self.attributes = attributes

        self.model = model
        self.prompt_template = prompt_template
        self.parser = parser
        self.verbose = verbose
        self.compile()
    
    def compile(self, ):
        # create the chain
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['grade_level', 'course_title', 'course_description', 'objectives_topics', 'required_materials', 'num_weeks', 'course_outline', 'grading_policy', 'class_policy', 'customization'],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = prompt | self.model #| self.parser
        if self.verbose: logger.info(f"Chain for Syllabus compilation complete")
    
    def validate_response(self) -> bool:
        try:
            # Use Pydantic model to validate the response
            SyllabusFormat(**self.response)
            return True
        except ValidationError as e:
            if self.verbose:
                logger.error(f"Validation error during Syllabus response validation: {e}")
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during Syllabus response validation: {e}")
            return False
    
    def create_syllabus(self):
        attempts = 0
        max_attempts = 5  # Allow for more attempts to generate syllabus
        syllabus = None

        while attempts < max_attempts:
            attempts += 1
            self.response = self.chain.invoke(self.attributes)

            if self.verbose:
                logger.info(f"Generated Syllabus response attempt {attempts}: {self.response}")
            
            # if self.validate_response():
            #     syllabus = self.response
            #     if self.verbose:
            #         logger.info(f"Valid Syllabus added")
            #     break
            # else:
            #     if self.verbose:
            #         logger.warning(f"Invalid Syllabus response format. Attempt {attempts} of {max_attempts}")
            syllabus = self.response
            break
        
        # Log if fewer questions are generated
        if syllabus is None:
            logger.warning(f"Syllabus was not created")

        return syllabus
    
class SyllabusBuilder:
    def __init__(self, attributes, model = None, prompt_template = None, parser = None, verbose=False):
        self.verbose = verbose
        self.attributes = attributes

        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.4),
            "prompt_template": read_text_file("prompts/syllabus_prompt.txt"),
            "parser": JsonOutputParser(pydantic_object=SyllabusFormat),
        }

        self.model = model or default_config["model"]
        self.prompt_template = prompt_template or default_config["prompt_template"]
        self.parser = parser or default_config["parser"]

    
    def create_syllabus(self):
        syllabus_obj = Syllabus(model = self.model, attributes = self.attributes, prompt_template = self.prompt_template, parser = self.parser, verbose = self.verbose)
        syllabus = syllabus_obj.create_syllabus()
        return syllabus


### We might going to need different pydantic classes for each section 
class SyllabusFormat(BaseModel):
    course_title: str = Field(description = 'title of the course')
    course_description: str = Field(description = 'description of the cours')
    objectives_topics: str = Field(description = 'topics covered and objectives of the course')
    required_materials: str = Field(description = 'materials required for the course')
    course_outline: str = Field(description = 'outline of the course')
    grading_policy: str = Field(description = 'graiding policy of the course')
    class_policy: str = Field(description = 'class policy of the course')