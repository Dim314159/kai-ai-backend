from typing import List, Tuple, Dict, Any

import os

#from langchain_google_vertexai import VertexAI
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

from sentence_transformers import SentenceTransformer, util

from services.logger import setup_logger

logger = setup_logger(__name__)

def read_text_file(file_path):
    # Get the directory containing the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Combine the script directory with the relative file path
    absolute_file_path = os.path.join(script_dir, file_path)
    
    with open(absolute_file_path, 'r') as file:
        return file.read()

# base class
class QuestionBase:
    def __init__(self, model, embedding_model, topic, grade_level, prompt_template, parser, verbose):
        self.model = model
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.9  # Adjust threshold as needed
        self.topic = topic
        self.grade_level = grade_level
        self.prompt_template = prompt_template
        self.parser = parser
        self.verbose = verbose
        self.set_names()
        self.compile()
    
    def set_names(self):
        self.section_name = ''
        self.main_field = 'question'
    
    def compile(self, ):
        # create the chain
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["topic", "grade_level"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = prompt | self.model | self.parser
        
        if self.verbose: logger.info(f"Chain for {self.section_name} compilation complete")
    
    def sub_validate(self):
        pass

    def validate_response(self) -> bool:
        try:
            # Use Pydantic model to validate the response
            if self.sub_validate():
                return True
            else:
                return False
        except ValidationError as e:
            if self.verbose:
                logger.error(f"Validation error during {self.section_name} response validation: {e}")
            return False
        except TypeError as e:
            if self.verbose:
                logger.error(f"TypeError during {self.section_name} response validation: {e}")
            return False

    def not_unique(self):
        question = self.response[self.main_field]
        question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)

        for q in self.question_bank:
            similarity = util.pytorch_cos_sim(question_embedding, q)
            if similarity.item() > self.similarity_threshold:
                return True

        self.question_bank.append(question_embedding)
        return False
    
    def create_questions(self, num_questions = 0) -> List:
        attempts = 0
        max_attempts = num_questions * 5  # Allow for more attempts to generate questions
        self.question_bank = []
        generated_questions = []

        while len(generated_questions) < num_questions and attempts < max_attempts:
            # Move to the next attempt regardless of success to ensure progress
            attempts += 1
            self.response = self.chain.invoke({"topic": self.topic, "grade_level": self.grade_level})

            if self.verbose:
                logger.info(f"Generated {self.section_name} response attempt {attempts}: {self.response}")
            
            # Directly check if the response format is valid
            flag = self.validate_response()
            if flag:
                # check if question is unique
                if self.section_name != 'Summary' and self.not_unique():
                    logger.warning(f"Not unique {self.section_name}. Attempt {attempts} of {max_attempts}")
                    continue
                generated_questions.append(self.response)
                if self.verbose:
                    logger.info(f"Valid {self.section_name} added: {self.response}")
                    logger.info(f"Total generated {self.section_name}(s): {len(generated_questions)}")
            else:
                if self.verbose:
                    logger.warning(f"Invalid {self.section_name} response format. Attempt {attempts} of {max_attempts}")
        
        # Log if fewer questions are generated
        if len(generated_questions) < num_questions:
            logger.warning(f"Only generated {len(generated_questions)} out of {num_questions} requested {self.section_name}(s)")

        return generated_questions

# Multiple Choice Question subclass
class MultipleChoiceQuestion(QuestionBase):
    def set_names(self):
        self.section_name = 'Miltiple-Choice question'
        self.main_field = 'question'
    
    def sub_validate(self):
        MultipleChoiceQuestionFormat(**self.response)
        return True

# Summary subclass
class Summary(QuestionBase):
    def set_names(self):
        self.section_name = 'Summary'
        self.main_field = 'description'
    
    def sub_validate(self):
        SummaryFormat(**self.response)
        return True

# Fill in the Blank subclass
class FillInTheBlankQuestion(QuestionBase):
    def set_names(self):
        self.section_name = 'Fill-in-the-blank question'
        self.main_field = 'question'
    
    def sub_validate(self):
        validated_response = FillinblankQuestionFormat(**self.response)
        if '_' in getattr(validated_response, self.main_field):
            return True
        else:
            if self.verbose:
                logger.warning(f"{self.section_name} does not have blank ___.")
            return False

# Open ended question subclass
class OpenEndedQuestion(QuestionBase):
    def set_names(self):
        self.section_name = 'Open-ended question'
        self.main_field = 'question'
    
    def sub_validate(self):
        OpenEndedQuestionFormat(**self.response)
        return True

# True False question subclass
class TrueFalseQuestion(QuestionBase):
    def set_names(self):
        self.section_name = 'True or False question'
        self.main_field = 'question'
    
    def sub_validate(self):
        TrueFalseQuestionFormat(**self.response)
        return True

# Create worksheets
class WorksheetBuilder:
    def __init__(self, topic, grade_level, model=None, embedding_model = None, prompt_summary=None, parser_summary = None, prompt_fill_in_blank=None, parser_fill_in_blank=None, prompt_multiple_choice=None, parser_multiple_choice=None, prompt_open_ended = None, parser_open_ended = None, prompt_true_false = None, parser_true_false = None, verbose=False):
        if topic is None or grade_level is None:
            raise ValueError("Topic and Grade level must be provided")
        
        default_config = {
            "model": GoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5),
            "embedding_model": SentenceTransformer('all-MiniLM-L6-v2'),
            "prompt_summary": read_text_file("prompts/worksheet_prompt_summary.txt"),
            "parser_summary": JsonOutputParser(pydantic_object=SummaryFormat),
            "prompt_fill_in_blank": read_text_file('prompts/worksheet_prompt_fill_in_blank.txt'),
            "parser_fill_in_blank": JsonOutputParser(pydantic_object = FillinblankQuestionFormat),
            "prompt_multiple_choice": read_text_file("prompts/worksheet_prompt_multiple_choice.txt"),
            "parser_multiple_choice": JsonOutputParser(pydantic_object = MultipleChoiceQuestionFormat),
            "prompt_open_ended": read_text_file("prompts/worksheet_prompt_open_ended.txt"),
            "parser_open_ended": JsonOutputParser(pydantic_object = OpenEndedQuestionFormat),
            "prompt_true_false": read_text_file("prompts/worksheet_prompt_true_false.txt"),
            "parser_true_false": JsonOutputParser(pydantic_object = TrueFalseQuestionFormat)
        }
        
        self.model = model or default_config["model"]
        self.embedding_model = embedding_model or default_config["embedding_model"]
        self.prompt_summary = prompt_summary or default_config["prompt_summary"]
        self.parser_summary = parser_summary or default_config["parser_summary"]
        self.prompt_fill_in_blank = prompt_fill_in_blank or default_config["prompt_fill_in_blank"]
        self.parser_fill_in_blank = parser_fill_in_blank or default_config["parser_fill_in_blank"]
        self.prompt_multiple_choice = prompt_multiple_choice or default_config["prompt_multiple_choice"]
        self.parser_multiple_choice = parser_multiple_choice or default_config["parser_multiple_choice"]
        self.prompt_open_ended = prompt_open_ended or default_config["prompt_open_ended"]
        self.parser_open_ended = parser_open_ended or default_config["parser_open_ended"]
        self.prompt_true_false = prompt_true_false or default_config["prompt_true_false"]
        self.parser_true_false = parser_true_false or default_config["parser_true_false"]
        
        self.topic = topic
        self.grade_level = grade_level
        self.verbose = verbose
    
    def create_worksheets(self, num_worksheets: int = 1, num_fill_in_blank: int = 1, num_multiple_choice: int = 1, num_open_ended: int = 1, num_true_false: int = 1) -> List[str]:
        ### Limit just for testing
        if num_worksheets > 10:
            return {"message": "error", "data": "Number of Worksheets cannot exceed 10"}
        
        summary_obj = Summary(self.model, self.embedding_model, self.topic, self.grade_level, self.prompt_summary, self.parser_summary, self.verbose)
        fill_in_the_blank_obj = FillInTheBlankQuestion(self.model, self.embedding_model, self.topic, self.grade_level, self.prompt_fill_in_blank, self.parser_fill_in_blank, self.verbose)
        multiple_choice_obj = MultipleChoiceQuestion(self.model, self.embedding_model, self.topic, self.grade_level, self.prompt_multiple_choice, self.parser_multiple_choice, self.verbose)
        open_ended_obj = OpenEndedQuestion(self.model, self.embedding_model, self.topic, self.grade_level, self.prompt_open_ended, self.parser_open_ended, self.verbose)
        true_false_obj = TrueFalseQuestion(self.model, self.embedding_model, self.topic, self.grade_level, self.prompt_true_false, self.parser_true_false, self.verbose)
        generated_worksheets = []
        if self.verbose: logger.info(f"Creating Summary")
        generated_summary = summary_obj.create_questions(num_questions = 1)
        if generated_summary:
            generated_summary = generated_summary[0]
        else:
            generated_summary = {'description': ''}
        for i in range(num_worksheets):
            worksheet = {}
            worksheet.update(generated_summary)
            if self.verbose: logger.info(f"Creating {num_fill_in_blank} Fill-in-the-blank questions")
            generated_fill_in_blank = fill_in_the_blank_obj.create_questions(num_questions = num_fill_in_blank)
            worksheet["fill_in_blank"] = generated_fill_in_blank
            if self.verbose: logger.info(f"Creating {num_multiple_choice} Multiple-choice questions")
            generated_multiple_choice = multiple_choice_obj.create_questions(num_questions = num_multiple_choice)
            worksheet["multiple_choice"] = generated_multiple_choice
            if self.verbose: logger.info(f"Creating {num_open_ended} Open-ended questions")
            generated_open_ended = open_ended_obj.create_questions(num_questions = num_open_ended)
            worksheet["open_ended"] = generated_open_ended
            if self.verbose: logger.info(f"Creating {num_true_false} True or False questions")
            generated_true_false = true_false_obj.create_questions(num_questions = num_true_false)
            worksheet["true_false"] = generated_true_false

            generated_worksheets.append(worksheet)
        # Return the list of worksheets
        if self.verbose: logger.info(f"Created {len(generated_worksheets)} Worksheets")
        return generated_worksheets


class QuestionChoiceFormat(BaseModel):
    key: str = Field(description="A unique identifier for the choice using letters A, B, C, D, etc.")
    value: str = Field(description="The text content of the choice")
class MultipleChoiceQuestionFormat(BaseModel):
    question: str = Field(description="The question text")
    choices: List[QuestionChoiceFormat] = Field(description="A list of choices")
    answer: str = Field(description="The correct answer")

class FillinblankQuestionFormat(BaseModel):
    question: str = Field(description="The question text")
    answer: str = Field(description="The correct answer")

class TrueFalseQuestionFormat(BaseModel):
    question: str = Field(description="The question text")
    answer: str = Field(description="The correct answer")

class OpenEndedQuestionFormat(BaseModel):
    question: str = Field(description = "The question text")
    answer: str = Field(description = "The correct answer")

class SummaryFormat(BaseModel):
    description: str = Field(description = "Description of the topic")