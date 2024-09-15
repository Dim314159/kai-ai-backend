from app.services.tool_registry import ToolFile
from app.services.logger import setup_logger
from app.features.ai_resistant.tools import AIResistant
from app.api.error_utilities import LoaderError, ToolExecutorError

logger = setup_logger()

def executor(file, verbose=False):
            
    try:
        output = AIResistant()
    
    except Exception as e:
        error_message = f"Error in executor: {e}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    return output

