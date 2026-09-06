import redis
import json
import hashlib
from embedding_model import model
from config import Config

class SemanticCache:
    """Redis-based semantic caching for query deduplication"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            print("Redis cache connected successfully")
        except Exception as e:
            print(f"Redis connection failed: {e}. Caching disabled.")
            self.enabled = False
            self.redis_client = None
            
    def _generate_cache_key(self, query):
        """Generate cache key from query embedding"""
        # Create embedding
        embedding = model.encode(query)
        # Create hash from embedding
        embedding_str = str(embedding.tolist())
        return hashlib.md5(embedding_str.encode()).hexdigest()
        
    def get(self, query):
        """Retrieve cached result if available"""
        if not self.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(query)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            return None
            
    def set(self, query, result):
        """Cache query result"""
        if not self.enabled:
            return
            
        try:
            cache_key = self._generate_cache_key(query)
            cached_data = json.dumps(result)
            self.redis_client.setex(
                cache_key,
                Config.CACHE_TTL,
                cached_data
            )
        except Exception as e:
            print(f"Cache storage error: {e}")
            
    def get_stats(self):
        """Get cache statistics"""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            info = self.redis_client.info('stats')
            return {
                "enabled": True,
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "total_keys": self.redis_client.dbsize()
            }
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}
