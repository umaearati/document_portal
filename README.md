### RAG-Based Document Analysis Platform

Professionals working with large volumes of documents — contracts, reports, research papers — spend significant time manually searching for relevant information. This project addresses that by enabling users to upload any document and ask questions in natural language, receiving contextually accurate answers grounded in the document's actual content.

## Solution

A production-grade RAG system that ingests documents, builds a searchable vector index, and uses LLMs to generate grounded, context-aware responses — eliminating hallucination through retrieval-first design.

## Architecture

Built across 5 modular layers: document ingestion with PDF parsing and configurable chunking; FAISS vector storage with session-based index persistence; a RAG pipeline using structured prompt templates and Pydantic output validation; a FastAPI layer with async LLM execution and reusable RAG instances; and LangSmith-based observability for tracing and debugging.

## Tech Stack

FastAPI, FAISS, LangChain, OpenAI GPT-4, OpenAI Embeddings, LangSmith, Docker, AWS ECS Fargate, GitHub Actions CI/CD

## Optimisations

Eliminated per-request RAG reconstruction to reduce latency. Controlled context window size to prevent token overflow and manage cost. Session-based FAISS persistence for efficient retrieval across requests. Async API design for improved concurrency.

## Deployment

Containerised with Docker, deployed to AWS ECS Fargate (eu-west-2) via GitHub Actions CI/CD pipeline with automated unit testing as a deployment gate.
