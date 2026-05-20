#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MemCore - Lightweight AI Agent Persistent Memory Engine
轻量级AI Agent持久化记忆引擎

A zero-dependency, lightweight persistent memory system for AI coding agents.
Supports MCP protocol, semantic search, and multiple storage backends.

Author: gitstq
License: MIT
Version: 1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import urllib.parse

__version__ = "1.0.0"
__author__ = "gitstq"

# =============================================================================
# Constants & Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "storage_backend": "sqlite",  # sqlite, json, memory
    "sqlite_path": "~/.memcore/memories.db",
    "json_path": "~/.memcore/memories.json",
    "max_memories_per_session": 10000,
    "memory_retention_days": 365,
    "embedding_dim": 384,  # For lightweight semantic search
    "enable_compression": True,
    "mcp_enabled": True,
    "mcp_port": 3112,
    "similarity_threshold": 0.7,
    "max_context_tokens": 4000,
}

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Memory:
    """Represents a single memory entry."""
    id: str
    content: str
    memory_type: str = "generic"  # generic, code, error, decision, context
    session_id: str = "default"
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 to 2.0
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        data = asdict(self)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        """Create memory from dictionary."""
        return cls(**data)
    
    def get_age_days(self) -> float:
        """Get age of memory in days."""
        return (time.time() - self.timestamp) / 86400
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class MemoryQuery:
    """Query parameters for memory retrieval."""
    query: str
    session_id: Optional[str] = None
    memory_type: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    time_range_days: Optional[int] = None
    min_importance: float = 0.0


# =============================================================================
# Lightweight Embedding Engine
# =============================================================================

class LightweightEmbedder:
    """
    Zero-dependency lightweight embedding generator.
    Uses statistical feature extraction instead of neural networks.
    """
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._cache: Dict[str, List[float]] = {}
        self._cache_lock = threading.Lock()
        
    def _hash_token(self, token: str) -> int:
        """Generate deterministic hash for token."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and extract words
        text = text.lower()
        # Keep alphanumeric and some code-related chars
        text = re.sub(r'[^a-z0-9_\-\.\s]', ' ', text)
        tokens = text.split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                      'be', 'been', 'being', 'have', 'has', 'had',
                      'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to',
                      'of', 'in', 'for', 'on', 'with', 'at', 'by',
                      'from', 'as', 'into', 'through', 'during',
                      'before', 'after', 'above', 'below', 'between',
                      'and', 'but', 'or', 'yet', 'so', 'if', 'because',
                      'although', 'though', 'while', 'where', 'when',
                      'that', 'which', 'who', 'whom', 'whose', 'what',
                      'this', 'these', 'those', 'i', 'you', 'he', 'she',
                      'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        return [t for t in tokens if t not in stop_words and len(t) > 1]
    
    def _ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """Generate n-grams from tokens."""
        if len(tokens) < n:
            return tokens
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        if not text:
            return [0.0] * self.dim
            
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Tokenize and extract features
        tokens = self._tokenize(text)
        all_features = tokens + self._ngrams(tokens, 2) + self._ngrams(tokens, 3)
        
        # Initialize vector
        vector = [0.0] * self.dim
        
        # Hash-based feature projection
        for feature in all_features:
            # Weight by inverse frequency (simplified)
            weight = 1.0 + (len(feature) * 0.1)
            
            # Hash to multiple dimensions
            h = self._hash_token(feature)
            for i in range(min(8, self.dim)):
                idx = (h + i * 997) % self.dim
                # Use sine for smooth distribution
                value = ((h >> (i * 4)) & 0xF) / 15.0
                vector[idx] += weight * (value - 0.5) * 2
        
        # Normalize
        magnitude = sum(x * x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        # Cache result
        with self._cache_lock:
            if len(self._cache) > 10000:
                self._cache.clear()
            self._cache[cache_key] = vector
        
        return vector
    
    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return max(0.0, min(1.0, dot_product))


# =============================================================================
# Storage Backends
# =============================================================================

class StorageBackend:
    """Abstract base class for storage backends."""
    
    def store(self, memory: Memory) -> bool:
        raise NotImplementedError
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        raise NotImplementedError
    
    def search(self, query: MemoryQuery, embedder: LightweightEmbedder) -> List[Tuple[Memory, float]]:
        raise NotImplementedError
    
    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError
    
    def list_sessions(self) -> List[str]:
        raise NotImplementedError
    
    def cleanup_old_memories(self, max_age_days: int) -> int:
        raise NotImplementedError
    
    def close(self):
        pass


class SQLiteBackend(StorageBackend):
    """SQLite-based persistent storage."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT DEFAULT 'generic',
                session_id TEXT DEFAULT 'default',
                timestamp REAL NOT NULL,
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL NOT NULL,
                embedding TEXT,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
        """)
        conn.commit()
    
    def store(self, memory: Memory) -> bool:
        """Store a memory."""
        try:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, content, memory_type, session_id, timestamp, tags, 
                 importance, access_count, last_accessed, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.content, memory.memory_type, memory.session_id,
                memory.timestamp, json.dumps(memory.tags), memory.importance,
                memory.access_count, memory.last_accessed,
                json.dumps(memory.embedding) if memory.embedding else None,
                json.dumps(memory.metadata)
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing memory: {e}", file=sys.stderr)
            return False
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        try:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            
            if row:
                return self._row_to_memory(row)
            return None
        except Exception as e:
            print(f"Error retrieving memory: {e}", file=sys.stderr)
            return None
    
    def search(self, query: MemoryQuery, embedder: LightweightEmbedder) -> List[Tuple[Memory, float]]:
        """Search memories using semantic similarity."""
        try:
            conn = self._get_conn()
            
            # Build query
            sql = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            if query.session_id:
                sql += " AND session_id = ?"
                params.append(query.session_id)
            
            if query.memory_type:
                sql += " AND memory_type = ?"
                params.append(query.memory_type)
            
            if query.time_range_days:
                cutoff = time.time() - (query.time_range_days * 86400)
                sql += " AND timestamp > ?"
                params.append(cutoff)
            
            if query.min_importance > 0:
                sql += " AND importance >= ?"
                params.append(query.min_importance)
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(query.limit * 10)  # Get more for ranking
            
            rows = conn.execute(sql, params).fetchall()
            memories = [self._row_to_memory(row) for row in rows]
            
            # Calculate semantic similarity
            query_embedding = embedder.embed(query.query)
            results = []
            
            for memory in memories:
                if memory.embedding:
                    similarity = embedder.similarity(query_embedding, memory.embedding)
                else:
                    # Fallback to simple text matching
                    similarity = self._text_similarity(query.query, memory.content)
                
                # Boost by importance and recency
                age_factor = max(0.5, 1.0 - memory.get_age_days() / 365)
                score = similarity * memory.importance * age_factor
                
                results.append((memory, score))
            
            # Sort by score and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:query.limit]
            
        except Exception as e:
            print(f"Error searching memories: {e}", file=sys.stderr)
            return []
    
    def _text_similarity(self, query: str, content: str) -> float:
        """Simple text similarity for fallback."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}", file=sys.stderr)
            return False
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        try:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM memories"
            ).fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            print(f"Error listing sessions: {e}", file=sys.stderr)
            return []
    
    def cleanup_old_memories(self, max_age_days: int) -> int:
        """Remove memories older than specified days."""
        try:
            conn = self._get_conn()
            cutoff = time.time() - (max_age_days * 86400)
            cursor = conn.execute(
                "DELETE FROM memories WHERE timestamp < ?", (cutoff,)
            )
            conn.commit()
            return cursor.rowcount
        except Exception as e:
            print(f"Error cleaning up memories: {e}", file=sys.stderr)
            return 0
    
    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row['id'],
            content=row['content'],
            memory_type=row['memory_type'],
            session_id=row['session_id'],
            timestamp=row['timestamp'],
            tags=json.loads(row['tags']),
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=row['last_accessed'],
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            metadata=json.loads(row['metadata'])
        )
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


class JSONBackend(StorageBackend):
    """JSON file-based storage for simple use cases."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path).expanduser()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._memories: Dict[str, Memory] = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load memories from file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for mem_id, mem_data in data.items():
                        self._memories[mem_id] = Memory.from_dict(mem_data)
            except Exception as e:
                print(f"Error loading memories: {e}", file=sys.stderr)
    
    def _save(self):
        """Save memories to file."""
        try:
            with self._lock:
                data = {mid: mem.to_dict() for mid, mem in self._memories.items()}
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving memories: {e}", file=sys.stderr)
    
    def store(self, memory: Memory) -> bool:
        """Store a memory."""
        with self._lock:
            self._memories[memory.id] = memory
        self._save()
        return True
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        return self._memories.get(memory_id)
    
    def search(self, query: MemoryQuery, embedder: LightweightEmbedder) -> List[Tuple[Memory, float]]:
        """Search memories."""
        with self._lock:
            memories = list(self._memories.values())
        
        # Filter
        if query.session_id:
            memories = [m for m in memories if m.session_id == query.session_id]
        if query.memory_type:
            memories = [m for m in memories if m.memory_type == query.memory_type]
        if query.time_range_days:
            cutoff = time.time() - (query.time_range_days * 86400)
            memories = [m for m in memories if m.timestamp > cutoff]
        if query.min_importance > 0:
            memories = [m for m in memories if m.importance >= query.min_importance]
        
        # Rank by similarity
        query_embedding = embedder.embed(query.query)
        results = []
        
        for memory in memories:
            if memory.embedding:
                similarity = embedder.similarity(query_embedding, memory.embedding)
            else:
                similarity = self._text_similarity(query.query, memory.content)
            
            age_factor = max(0.5, 1.0 - memory.get_age_days() / 365)
            score = similarity * memory.importance * age_factor
            results.append((memory, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:query.limit]
    
    def _text_similarity(self, query: str, content: str) -> float:
        """Simple text similarity."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            if memory_id in self._memories:
                del self._memories[memory_id]
                self._save()
                return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        with self._lock:
            return list(set(m.session_id for m in self._memories.values()))
    
    def cleanup_old_memories(self, max_age_days: int) -> int:
        """Remove old memories."""
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            to_delete = [mid for mid, m in self._memories.items() if m.timestamp < cutoff]
            for mid in to_delete:
                del self._memories[mid]
            if to_delete:
                self._save()
        return len(to_delete)


class MemoryBackend(StorageBackend):
    """In-memory storage (non-persistent)."""
    
    def __init__(self):
        self._memories: Dict[str, Memory] = {}
        self._lock = threading.Lock()
    
    def store(self, memory: Memory) -> bool:
        with self._lock:
            self._memories[memory.id] = memory
        return True
    
    def retrieve(self, memory_id: str) -> Optional[Memory]:
        return self._memories.get(memory_id)
    
    def search(self, query: MemoryQuery, embedder: LightweightEmbedder) -> List[Tuple[Memory, float]]:
        with self._lock:
            memories = list(self._memories.values())
        
        if query.session_id:
            memories = [m for m in memories if m.session_id == query.session_id]
        if query.memory_type:
            memories = [m for m in memories if m.memory_type == query.memory_type]
        
        query_embedding = embedder.embed(query.query)
        results = []
        
        for memory in memories:
            if memory.embedding:
                similarity = embedder.similarity(query_embedding, memory.embedding)
            else:
                similarity = self._text_similarity(query.query, memory.content)
            
            age_factor = max(0.5, 1.0 - memory.get_age_days() / 365)
            score = similarity * memory.importance * age_factor
            results.append((memory, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:query.limit]
    
    def _text_similarity(self, query: str, content: str) -> float:
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        return len(query_words & content_words) / len(query_words)
    
    def delete(self, memory_id: str) -> bool:
        with self._lock:
            if memory_id in self._memories:
                del self._memories[memory_id]
                return True
        return False
    
    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(set(m.session_id for m in self._memories.values()))
    
    def cleanup_old_memories(self, max_age_days: int) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        with self._lock:
            to_delete = [mid for mid, m in self._memories.items() if m.timestamp < cutoff]
            for mid in to_delete:
                del self._memories[mid]
        return len(to_delete)


# =============================================================================
# Core Memory Engine
# =============================================================================

class MemCore:
    """
    Main memory engine for AI agents.
    Provides persistent memory storage with semantic search capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.embedder = LightweightEmbedder(self.config['embedding_dim'])
        self._backend = self._create_backend()
        self._session_counter = 0
        self._lock = threading.Lock()
    
    def _create_backend(self) -> StorageBackend:
        """Create storage backend based on configuration."""
        backend_type = self.config['storage_backend']
        
        if backend_type == 'sqlite':
            return SQLiteBackend(self.config['sqlite_path'])
        elif backend_type == 'json':
            return JSONBackend(self.config['json_path'])
        elif backend_type == 'memory':
            return MemoryBackend()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    def remember(self, content: str, memory_type: str = "generic", 
                 session_id: str = "default", tags: Optional[List[str]] = None,
                 importance: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory.
        
        Args:
            content: The memory content
            memory_type: Type of memory (generic, code, error, decision, context)
            session_id: Session identifier
            tags: List of tags for categorization
            importance: Importance score (0.0 to 2.0)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        memory_id = hashlib.md5(
            f"{session_id}:{content}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Generate embedding
        embedding = self.embedder.embed(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            session_id=session_id,
            tags=tags or [],
            importance=importance,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self._backend.store(memory)
        return memory_id
    
    def recall(self, query: str, session_id: Optional[str] = None,
               memory_type: Optional[str] = None, limit: int = 10,
               min_importance: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Search query
            session_id: Filter by session
            memory_type: Filter by memory type
            limit: Maximum number of results
            min_importance: Minimum importance threshold
            
        Returns:
            List of memory dictionaries with similarity scores
        """
        memory_query = MemoryQuery(
            query=query,
            session_id=session_id,
            memory_type=memory_type,
            limit=limit,
            min_importance=min_importance
        )
        
        results = self._backend.search(memory_query, self.embedder)
        
        # Update access stats
        for memory, score in results:
            memory.update_access()
            self._backend.store(memory)
        
        return [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type,
                "session_id": memory.session_id,
                "timestamp": memory.timestamp,
                "tags": memory.tags,
                "importance": memory.importance,
                "similarity": round(score, 4),
                "metadata": memory.metadata
            }
            for memory, score in results
        ]
    
    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        return self._backend.delete(memory_id)
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        memory = self._backend.retrieve(memory_id)
        if memory:
            return memory.to_dict()
        return None
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return self._backend.list_sessions()
    
    def get_session_memories(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all memories for a session."""
        return self.recall("*", session_id=session_id, limit=limit)
    
    def cleanup(self, max_age_days: Optional[int] = None) -> int:
        """Clean up old memories."""
        days = max_age_days or self.config['memory_retention_days']
        return self._backend.cleanup_old_memories(days)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        sessions = self._backend.list_sessions()
        return {
            "total_sessions": len(sessions),
            "sessions": sessions,
            "storage_backend": self.config['storage_backend'],
            "embedding_dim": self.config['embedding_dim'],
            "version": __version__
        }
    
    def export_memories(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export memories to a list."""
        if session_id:
            return self.get_session_memories(session_id, limit=10000)
        else:
            all_memories = []
            for sid in self.list_sessions():
                all_memories.extend(self.get_session_memories(sid, limit=10000))
            return all_memories
    
    def import_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Import memories from a list."""
        count = 0
        for mem_data in memories:
            try:
                memory = Memory.from_dict(mem_data)
                if self._backend.store(memory):
                    count += 1
            except Exception as e:
                print(f"Error importing memory: {e}", file=sys.stderr)
        return count
    
    def close(self):
        """Close the memory engine."""
        self._backend.close()


# =============================================================================
# MCP Server Implementation
# =============================================================================

class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for MCP protocol."""
    
    memcore: MemCore = None
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
    
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_json(self) -> Dict[str, Any]:
        """Read JSON from request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        return {}
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        
        if path == '/health':
            self._send_json({"status": "ok", "version": __version__})
        
        elif path == '/memories':
            session_id = params.get('session', [None])[0]
            query = params.get('q', [''])[0]
            limit = int(params.get('limit', ['10'])[0])
            
            if query:
                results = self.memcore.recall(query, session_id=session_id, limit=limit)
            else:
                results = self.memcore.get_session_memories(session_id or "default", limit)
            
            self._send_json({"memories": results})
        
        elif path == '/sessions':
            sessions = self.memcore.list_sessions()
            self._send_json({"sessions": sessions})
        
        elif path == '/stats':
            stats = self.memcore.get_stats()
            self._send_json(stats)
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        data = self._read_json()
        
        if path == '/remember':
            memory_id = self.memcore.remember(
                content=data.get('content', ''),
                memory_type=data.get('type', 'generic'),
                session_id=data.get('session_id', 'default'),
                tags=data.get('tags', []),
                importance=data.get('importance', 1.0),
                metadata=data.get('metadata', {})
            )
            self._send_json({"id": memory_id, "status": "stored"})
        
        elif path == '/recall':
            results = self.memcore.recall(
                query=data.get('query', ''),
                session_id=data.get('session_id'),
                memory_type=data.get('type'),
                limit=data.get('limit', 10),
                min_importance=data.get('min_importance', 0.0)
            )
            self._send_json({"results": results})
        
        elif path == '/forget':
            memory_id = data.get('id', '')
            success = self.memcore.forget(memory_id)
            self._send_json({"success": success})
        
        elif path == '/cleanup':
            days = data.get('days', 365)
            count = self.memcore.cleanup(days)
            self._send_json({"deleted": count})
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


class MCPServer:
    """MCP Protocol Server for MemCore."""
    
    def __init__(self, memcore: MemCore, port: int = 3112):
        self.memcore = memcore
        self.port = port
        self.server = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self, blocking: bool = False):
        """Start the MCP server."""
        MCPHandler.memcore = self.memcore
        
        class ReusableTCPServer(socketserver.ThreadingMixIn, HTTPServer):
            allow_reuse_address = True
            daemon_threads = True
        
        self.server = ReusableTCPServer(('0.0.0.0', self.port), MCPHandler)
        
        if blocking:
            print(f"🚀 MemCore MCP Server running on http://localhost:{self.port}")
            self.server.serve_forever()
        else:
            self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self._thread.start()
            print(f"🚀 MemCore MCP Server started on http://localhost:{self.port}")
    
    def stop(self):
        """Stop the MCP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 MCP Server stopped")


# =============================================================================
# CLI Interface
# =============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='memcore',
        description='MemCore - Lightweight AI Agent Persistent Memory Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  memcore server                    # Start MCP server
  memcore remember "Learned about JWT auth" --session myproject
  memcore recall "JWT auth" --session myproject
  memcore sessions                  # List all sessions
  memcore stats                     # Show statistics
  memcore cleanup --days 30         # Clean up old memories
        """
    )
    
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument('--backend', choices=['sqlite', 'json', 'memory'],
                        default='sqlite', help='Storage backend (default: sqlite)')
    parser.add_argument('--data-dir', default='~/.memcore',
                        help='Data directory (default: ~/.memcore)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start MCP server')
    server_parser.add_argument('--port', type=int, default=3112,
                               help='Server port (default: 3112)')
    
    # Remember command
    remember_parser = subparsers.add_parser('remember', help='Store a memory')
    remember_parser.add_argument('content', help='Memory content')
    remember_parser.add_argument('--session', '-s', default='default',
                                 help='Session ID (default: default)')
    remember_parser.add_argument('--type', '-t', default='generic',
                                 choices=['generic', 'code', 'error', 'decision', 'context'],
                                 help='Memory type')
    remember_parser.add_argument('--tags', help='Comma-separated tags')
    remember_parser.add_argument('--importance', '-i', type=float, default=1.0,
                                 help='Importance score (0.0-2.0)')
    
    # Recall command
    recall_parser = subparsers.add_parser('recall', help='Retrieve memories')
    recall_parser.add_argument('query', help='Search query')
    recall_parser.add_argument('--session', '-s', help='Filter by session')
    recall_parser.add_argument('--type', '-t', help='Filter by type')
    recall_parser.add_argument('--limit', '-n', type=int, default=10,
                               help='Maximum results (default: 10)')
    
    # Sessions command
    subparsers.add_parser('sessions', help='List all sessions')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old memories')
    cleanup_parser.add_argument('--days', type=int, default=365,
                                help='Delete memories older than N days (default: 365)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export memories')
    export_parser.add_argument('--session', '-s', help='Export specific session')
    export_parser.add_argument('--output', '-o', default='memories.json',
                               help='Output file (default: memories.json)')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import memories')
    import_parser.add_argument('file', help='JSON file to import')
    
    return parser


def run_cli():
    """Run CLI interface."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize MemCore
    data_dir = Path(args.data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        'storage_backend': args.backend,
        'sqlite_path': str(data_dir / 'memories.db'),
        'json_path': str(data_dir / 'memories.json'),
    }
    
    memcore = MemCore(config)
    
    try:
        if args.command == 'server':
            server = MCPServer(memcore, port=args.port)
            print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   MemCore v{__version__}                        ║
║     Lightweight AI Agent Persistent Memory Engine            ║
╠══════════════════════════════════════════════════════════════╣
║  🚀 MCP Server: http://localhost:{args.port:<4}                      ║
║  📁 Data: {config['sqlite_path'] if args.backend == 'sqlite' else config['json_path']:<42} ║
║  🔧 Backend: {args.backend:<10}                                   ║
╚══════════════════════════════════════════════════════════════╝

Endpoints:
  GET  /health              - Health check
  GET  /memories?q=query    - Search memories
  GET  /sessions            - List sessions
  POST /remember            - Store memory
  POST /recall              - Retrieve memories
  POST /forget              - Delete memory

Press Ctrl+C to stop
            """)
            try:
                server.start(blocking=True)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        
        elif args.command == 'remember':
            tags = args.tags.split(',') if args.tags else []
            memory_id = memcore.remember(
                content=args.content,
                memory_type=args.type,
                session_id=args.session,
                tags=tags,
                importance=args.importance
            )
            print(f"✅ Memory stored with ID: {memory_id}")
        
        elif args.command == 'recall':
            results = memcore.recall(
                query=args.query,
                session_id=args.session,
                memory_type=args.type,
                limit=args.limit
            )
            if results:
                print(f"\n🔍 Found {len(results)} relevant memories:\n")
                for i, mem in enumerate(results, 1):
                    print(f"  {i}. [{mem['type']}] (similarity: {mem['similarity']})")
                    print(f"     {mem['content'][:100]}{'...' if len(mem['content']) > 100 else ''}")
                    print(f"     Session: {mem['session_id']} | Tags: {', '.join(mem['tags']) or 'none'}")
                    print()
            else:
                print("❌ No memories found")
        
        elif args.command == 'sessions':
            sessions = memcore.list_sessions()
            if sessions:
                print(f"\n📁 {len(sessions)} session(s):\n")
                for session in sessions:
                    print(f"  • {session}")
            else:
                print("No sessions found")
        
        elif args.command == 'stats':
            stats = memcore.get_stats()
            print(f"""
📊 MemCore Statistics
═══════════════════════
Version:        {stats['version']}
Backend:        {stats['storage_backend']}
Embedding Dim:  {stats['embedding_dim']}
Sessions:       {stats['total_sessions']}
            """)
        
        elif args.command == 'cleanup':
            count = memcore.cleanup(args.days)
            print(f"🧹 Cleaned up {count} old memories")
        
        elif args.command == 'export':
            memories = memcore.export_memories(args.session)
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(memories, f, ensure_ascii=False, indent=2)
            print(f"📤 Exported {len(memories)} memories to {args.output}")
        
        elif args.command == 'import':
            with open(args.file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            count = memcore.import_memories(memories)
            print(f"📥 Imported {count} memories from {args.file}")
    
    finally:
        memcore.close()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    run_cli()
