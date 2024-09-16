from app.services.tool_registry import ToolFile
from app.services.logger import setup_logger
from app.features.ai_resistant.tools import AIResistant, RAGpipeline
from app.api.error_utilities import LoaderError, ToolExecutorError

logger = setup_logger()

def executor(file, verbose=False):
    try:
        # Instantiate RAG pipeline with default values
        pipeline = RAGpipeline(verbose=verbose)
        pipeline.compile()
        # Process the uploaded files
        vectorstore = pipeline(file)
        output = AIResistant(vectorstore)
    
    except LoaderError as e:
        error_message = e
        logger.error(f"Error in RAGPipeline -> {error_message}")
        raise ToolExecutorError(error_message)
    
    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return output

