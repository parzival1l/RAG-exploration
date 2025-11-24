"""RAG Pipeline implementation using LangChain, Qdrant, and Replicate."""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import Replicate
from qdrant_client import QdrantClient

from config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    TOP_K_CANDIDATES,
    SCORE_THRESHOLD,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    REPLICATE_API_TOKEN,
)
from src.embeddings import get_embeddings


class RAGPipeline:
    """Complete RAG pipeline for question answering."""

    def __init__(self):
        """Initialize the RAG pipeline components."""
        # Initialize embeddings
        self.embeddings = get_embeddings()

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Initialize vector store
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=self.embeddings,
        )

        # Initialize LLM
        self.llm = Replicate(
            model=LLM_MODEL,
            replicate_api_token=REPLICATE_API_TOKEN,
            model_kwargs={
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "top_p": 0.9,
            }
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": TOP_K_CANDIDATES,
                "score_threshold": SCORE_THRESHOLD,
            }
        )

        # Create prompt template
        self.prompt = self._create_prompt()

        # Create the chain
        self.chain = self._create_chain()

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for the RAG pipeline.

        Returns:
            ChatPromptTemplate for the QA task
        """
        template = """You are an AI assistant tasked with answering questions based solely on the provided context from a knowledge base.

Context Information:
{context}

User Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. If the context contains relevant information, provide a clear and comprehensive answer
3. Cite specific sources from the context when making statements (e.g., "According to the documentation...")
4. If the context does not contain sufficient information to answer the question, clearly state: "I couldn't find relevant information in the knowledge base to answer this question."
5. Do not use any external knowledge or make assumptions beyond what's in the context
6. Distinguish clearly between facts from the context and any logical inferences you make
7. If there are multiple relevant pieces of information, synthesize them coherently

Answer:"""

        return ChatPromptTemplate.from_template(template)

    def _create_chain(self):
        """Create the LangChain LCEL chain.

        Returns:
            Runnable chain for the RAG pipeline
        """
        # Format documents function
        def format_docs(docs: List[Document]) -> str:
            """Format retrieved documents into a single context string."""
            if not docs:
                return "No relevant documents found."

            formatted = []
            for i, doc in enumerate(docs, 1):
                # Extract metadata for citation
                metadata = doc.metadata
                source_info = []

                if "source" in metadata:
                    source_info.append(f"Source: {metadata['source']}")
                if "page" in metadata:
                    source_info.append(f"Page: {metadata['page']}")
                if "title" in metadata:
                    source_info.append(f"Title: {metadata['title']}")

                source_str = f"[Document {i}" + (f" - {', '.join(source_info)}]" if source_info else "]")

                formatted.append(f"{source_str}\n{doc.page_content}\n")

            return "\n---\n".join(formatted)

        # Create the chain using LCEL
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline.

        Args:
            question: User question
            return_sources: Whether to return source documents

        Returns:
            Dictionary containing answer and optional sources
        """
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)

        # Generate answer
        answer = self.chain.invoke(question)

        result = {
            "question": question,
            "answer": answer,
            "num_sources": len(docs),
        }

        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]

        return result

    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Retrieve documents without generation.

        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)
            score_threshold: Minimum score threshold (overrides default)

        Returns:
            List of relevant documents
        """
        # Create custom retriever if parameters are specified
        if k is not None or score_threshold is not None:
            search_kwargs = {}
            if k is not None:
                search_kwargs["k"] = k
            if score_threshold is not None:
                search_kwargs["score_threshold"] = score_threshold

            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs
            )
            return retriever.get_relevant_documents(query)

        return self.retriever.get_relevant_documents(query)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection.

        Returns:
            Dictionary with collection statistics
        """
        collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)

        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": collection_info.vectors_count,
            "points_count": collection_info.points_count,
            "status": collection_info.status,
        }


def create_rag_pipeline() -> RAGPipeline:
    """Factory function to create a RAG pipeline instance.

    Returns:
        Configured RAGPipeline instance
    """
    return RAGPipeline()


if __name__ == "__main__":
    # Example usage
    pipeline = create_rag_pipeline()

    # Print collection stats
    stats = pipeline.get_collection_stats()
    print("Collection Statistics:")
    print(f"  Name: {stats['collection_name']}")
    print(f"  Points: {stats['points_count']}")
    print(f"  Status: {stats['status']}")

    # Example query
    question = "What is Qdrant?"
    result = pipeline.query(question)

    print("\n" + "=" * 50)
    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources used: {result['num_sources']}")
    print("=" * 50)
