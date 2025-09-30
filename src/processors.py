import asyncio
import os
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from tqdm import tqdm
from typing import Any

import ollama
from langchain_text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_document_chunks(document_data: dict) -> dict:
    """Process a single document into chunks."""
    doc_id = document_data['id']
    content = document_data['content']

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]
    )
    raw_chunks = text_splitter.split_text(content)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text.strip()) < 50: #we skip tiny chunks here
            continue
        chunk = {
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk_text.strip(),
            "word_count": len(chunk_text.split()),
            "chunk_index": i,
            "document_id": doc_id
        }
        chunks.append(chunk)
    return {
        "document_id": doc_id,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "processed_by_pid": os.getpid()
    }

class OllamaEmbeddingClient:
    def __init__(self, model: str = "nomic-embed-text", embedding_size: int = 768):
        self.model = model
        self.embedding_size = embedding_size
        self._client = ollama.AsyncClient()

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text using the specified model."""
        response = await self._client.embeddings(model=self.model, prompt=text)
        return response['embedding']

    async def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        tasks = [self.generate_embedding(text, self.model) for text in tqdm(texts)]

        ## Control concurrency to avoid overwhelming Ollama
        batch_size = 5
        embeddings = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    raise 
                else:
                    embeddings.append(result)

        return embeddings

    @property
    def client(self):
        return self._client

class DocumentProcessor:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "documents",
    ):
        self.qdrant_client = AsyncQdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.ollama_client = OllamaEmbeddingClient()

        ## Instantiate queues
        self.upload_queue = asyncio.Queue(maxsize=100)
        self.processing_queue = asyncio.Queue(maxsize=50)
        self.embedding_queue = asyncio.Queue(maxsize=20)
        self.storage_queue = asyncio.Queue(maxsize=10)

        ## Locks for shared resources
        self.stats_lock = asyncio.Lock()
        self.qdrant_lock = asyncio.Lock()

        self.stats = {
            "total_uploads": 0,
            "documents_processed": 0,
            "embeddings_generated": 0,
            "documents_stored": 0,
            "failed": 0
        }
        self.document_status = {}
        self.workers_started = False

    async def initialize_qdrant(self):
        """Initialize a Qdrant collection if it doesn't exist."""

        try:
            collections = await self.qdrant_client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if not collection_exists:
                logger.info(f"🔧 Creating Qdrant collection: {self.collection_name}")
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config = VectorParams(
                        size = self.ollama_client.embedding_size,
                        distance = Distance.COSINE
                    )
                )
                logger.info(f"✅ Qdrant collection created: {self.collection_name}")
            else:
                logger.info(f"✅ Qdrant collection already exists: {self.collection_name}")

        except Exception as e:
            logger.error(f"❌ Error initializing Qdrant collection: {e}")
            raise

    async def start_workers(self):
        """Start all background worker tasks."""

        if self.workers_started:
            return

        # Initialize Qdrant first
        await self.initialize_qdrant()

        self.worker_tasks = [
            # Document processors (CPU intensive)
            asyncio.create_task(self.document_processor_worker(1)),
            asyncio.create_task(self.document_processor_worker(2)),

            # Embedding generators (I/O with Ollama, but rate limited)
            asyncio.create_task(self.embedding_worker(1)),
            asyncio.create_task(self.embedding_worker(2)),

            # Qdrant storage workers (I/O intensive)
            asyncio.create_task(self.qdrant_storage_worker(1)),
            asyncio.create_task(self.qdrant_storage_worker(2)),

            # Stats reporter
            asyncio.create_task(self.stats_worker())
        ]

        self.workers_started = True
        logger.info("🚀 All workers started with Qdrant + Ollama!")

    async def add_document_to_queue(self, user_id: str, filename: str, content: str) -> str:
        """Add document to the processing queue"""

        doc_id = str(uuid.uuid4())
        document_item = {
            "id": doc_id,
            "user_id": user_id,
            "filename": filename,
            "content": content,
            "upload_time": datetime.isoformat(),
            'status': 'queued'
        } 
        await self.upload_queue.put(document_item)
        self.document_status[doc_id] = "queued"

        ## Use log to prevent race conditions for logging and updating class statistics
        async with self.stats_lock:
            self.stats['total_uploads'] += 1
            logger.info(f"📤 Document {filename} queued for processing")

        return doc_id

    async def document_processor_worker(self, worker_id: int):
        """Processes documents into chunks using multiprocessing."""

        logger.info(f"🔄 Document processor {worker_id} started")

        while True:
            try:
                # Get first item in the queue
                doc_item = await self.upload_queue.get()
                doc_id = doc_item['id']
                logger.info(f"🔄 Processor {worker_id}: Chunking {doc_item['filename']}")
                self.document_status[doc_id] = "processing"

                # CPU-intensive chunking in separate process
                loop = asyncio.get_running_loop()
                with ProcessPoolExecutor() as executor:
                    processed_result = await loop.run_in_executor(
                        executor, process_document_chunks, doc_item
                    )

                # Add to embedding queue
                embedding_item = {
                    **doc_item,
                    'chunks': processed_result['chunks'],
                    'processing_completed_at': datetime.isoformat(),
                }

                await self.processing_queue.put(embedding_item)

                async with self.stats_lock:
                    self.stats['document_processed'] += 1

                self.document_status[doc_id] = 'chunks_created'
                self.upload_queue.task_done()

                logger.info(f"✅ Processor {worker_id}: Created {len(processed_result['chunks'])} chunks")

            except Exception as e:
                logger.error(f"❌ Processor {worker_id} error: {e}")
                if 'doc_id' in locals():
                    self.document_status[doc_id] = 'failed'
                async with self.stats_lock:
                    self.stats['failed'] +=1
                self.upload_queue.task_done()

    async def embedding_worker(self, worker_id: int):
        """Generates embeddings using Ollama"""

        logger.info(f"🧠 Embedding worker {worker_id} started")

        while True:
            try:
                doc_item = await self.processing_queue.get()
                doc_id = doc_item['id']

                logger.info(f"🧠 Embedder {worker_id}: Generating embeddings for {doc_item['filename']}")
                self.document_status[doc_id] = "generating_embeddings"

                #Extract texts from chunks
                chunk_texts = [chunk['text'] for chunk in doc_item['chunks']]

                #Generate embeddings using Ollama (I/O bound but rate limited)
                embeddings = await self.ollama_client.generate_embeddings_batch(chunk_texts)

                #combine chunks with embeddings
                enriched_chunks = []
                for chunk, embedding in zip(doc_item['chunks'], embeddings):
                    enriched_chunks.append({
                        **chunk,
                        'embedding': embedding,
                        'embedded_at': datetime.isoformat()
                    })

                # Add to storage queue
                storage_item = {
                    **doc_item,
                    'enriched_chunks': enriched_chunks,
                    'embeddings_completed_at': datetime.isoformat(),
                }

                await self.embedding_queue.put(storage_item)

                async with self.stats_lock:
                    self.stats['embeddings_generated'] += len(embeddings)

                self.document_status[doc_id] = 'embeddings_ready'
                self.processing_queue.task_done()

                logger.info(f"✅ Embedder {worker_id}: Generated {len(embeddings)} embeddings")

            except Exception as e:
                logger.error(f"❌ Embedder {worker_id} error: {e}")
                if 'doc_id' in locals():
                    self.document_status[doc_id] = 'failed'
                async with self.stats_lock:
                    self.stats['failed'] +=1
                self.processing_queue.task_done()

    async def qdrant_storage_worker(self, worker_id: int):
        """Stores embeddings in Qdrant vector database"""

        logger.info(f"💾 Qdrant storage worker {worker_id} started")

        while True:
            try:
                doc_item = await self.embedding_queue.get()
                doc_id = doc_item['id']

                logger.info(f"💾 Storage {worker_id}: Storing embeddings for {doc_item['filename']}")
                self.document_status[doc_id] = "storing_embeddings"

                # Prepare points for Qdrant
                points = []
                for chunk in doc_item['enriched_chunks']:
                    point = PointStruct(
                        id=chunk['id'],
                        vector=chunk['embedding'],
                        payload={
                            'document_id': doc_id,
                            'user_id': doc_item['user_id'],
                            'filename': doc_item['filename'],
                            'chunk_index': chunk['chunk_index'],
                            'text': chunk['text'],
                            'word_count': chunk['word_count'],
                            'uploaded_at': doc_item['uploaded_at'],
                            'embedded_at': chunk['embedded_at']
                        }
                    )
                    points.append(point)

                # Batch upsert to Qdrant
                async with self.qdrant_lock:
                    await self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )

                async with self.stats_lock:
                    self.stats['documents_stored'] += 1

                self.document_status[doc_id] = 'stored'
                self.embedding_queue.task_done()

                logger.info(f"✅ Storage {worker_id}: {doc_item['filename']} completed!")

            except Exception as e:
                logger.error(f"❌ Storage {worker_id} error: {e}")
                if 'doc_id' in locals():
                    self.document_status[doc_id] = 'failed'
                async with self.stats_lock:
                    self.stats['failed'] +=1
                self.embedding_queue.task_done()

    async def stats_worker(self):
        """Background stats reporter"""

        while True:
            await asyncio.sleep(15)  # Report every 15 seconds
            async with self.stats_lock:
                stats = self.stats.copy()

            # Get Qdrant collection info
            try:
                collection_info = await self.qdrant_client.get_collection(self.collection_name)
                vector_count = collection_info.points_count
            except:
                vector_count = "unknown"

            logger.info(f"📊 STATS: {stats} | Qdrant vectors: {vector_count}")
            logger.info(
                f"📋 Queues - Upload: {self.upload_queue.qsize()}, "
                f"Processing: {self.processing_queue.qsize()}, "
                f"Embedding: {self.embedding_queue.qsize()}"
            )

    async def semantic_search(self, query: str, limit: int = 5) -> list[str]:
        """Search for similar documents using Qdrant"""

        try:
            query_embedding = await self.ollama_client.generate_embedding(query)

            search_result = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True
            )

            # Format results

            results = []
            for result in search_result:
                results.append({
                    'chunk_id': result.id,
                    'score': result.score,
                    'text': result.payload['text'],
                    'document_id': result.payload['document_id'],
                    'filename': result.payload['filename'],
                    'user_id': result.payload['user_id']
                })

            return results

        except Exception as e:
            logger.error(f"❌ Semantic search error: {e}")
            return []

    async def get_document_status(self, doc_id: str) -> dict:
        """Get current processing status of a document"""

        status = self.document_status.get(doc_id, 'not_found')
        async with self.stats_lock:
            current_stats = self.stats.copy()

        return {
            "document_id": doc_id,
            "status": status,
            "stats": current_stats
        }
