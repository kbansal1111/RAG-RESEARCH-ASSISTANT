import tiktoken
from config import Config

class TokenManager:
    """Token budget management for context window optimization"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding(Config.TOKEN_MODEL)
        self.max_tokens = Config.MAX_CONTEXT_TOKENS
        
    def count_tokens(self, text):
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
        
    def truncate_to_token_limit(self, text, max_tokens=None):
        """Truncate text to fit within token limit"""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        current_tokens = self.count_tokens(text)
        if current_tokens <= max_tokens:
            return text
            
        # Encode and truncate
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
        
    def select_chunks_within_budget(self, chunks, query, max_tokens=None):
        """Select chunks that fit within token budget, prioritizing by relevance"""
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        query_tokens = self.count_tokens(query)
        available_tokens = max_tokens - query_tokens - 100  # Reserve 100 tokens for prompt template
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if total_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
                
        return selected_chunks, total_tokens
        
    def estimate_prompt_tokens(self, context, question):
        """Estimate total tokens in augmented prompt"""
        template_tokens = 50  # Approximate tokens for prompt template
        context_tokens = self.count_tokens(context)
        question_tokens = self.count_tokens(question)
        return template_tokens + context_tokens + question_tokens
