import time
import logging
from functools import wraps
from config import Config

logger = logging.getLogger(__name__)

class RetryHandler:
    """Retry logic with exponential backoff for API calls"""
    
    @staticmethod
    def retry_with_backoff(max_retries=None, delay=None, backoff=None):
        """Decorator for retrying functions with exponential backoff"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                max_tries = max_retries or Config.MAX_RETRIES
                initial_delay = delay or Config.RETRY_DELAY
                backoff_factor = backoff or Config.RETRY_BACKOFF
                
                last_exception = None
                
                for attempt in range(max_tries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_tries - 1:
                            wait_time = initial_delay * (backoff_factor ** attempt)
                            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"All {max_tries} attempts failed. Last error: {str(e)}")
                
                raise last_exception
            return wrapper
        return decorator
    
    @staticmethod
    def handle_api_errors(func):
        """Decorator specifically for API error handling"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConnectionError as e:
                logger.error(f"Connection error: {str(e)}")
                raise
            except TimeoutError as e:
                logger.error(f"Timeout error: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"API error: {str(e)}")
                raise
        return wrapper
