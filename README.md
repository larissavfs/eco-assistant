# eco-assistant 🌱

AI-powered chatbot that helps users correctly dispose of waste using Retrieval-Augmented Generation (RAG) with LangChain.

## Problem

Incorrect waste disposal is a major environmental issue. Many people are unsure how to discard items like batteries, medicines, electronics, or medical waste.

## Solution

This project provides a chatbot where users can ask questions about waste disposal and receive accurate, context-based answers.

Examples:

* "How do I dispose of batteries?"
* "Can I throw expired medicine in the trash?"
* "Where should I discard syringes?"

## Tech Stack

* Python
* LangChain
* FAISS (vector database)
* OpenAI API (or similar LLM)

## How it works

1. Documents about waste disposal are collected
2. Text is split into smaller chunks
3. Chunks are transformed into embeddings
4. A vector database stores the embeddings
5. User queries retrieve relevant information
6. The model generates an answer based on context

## Project Structure

```
eco-disposal-assistant/
│
├── data/
├── src/
│   ├── ingest.py
│   ├── query.py
│   └── rag_pipeline.py
│
├── requirements.txt
└── .env
```

## Example

**User:** How to dispose of syringes?
**Answer:** Syringes should be placed in proper sharps containers and taken to authorized collection points such as pharmacies or healthcare units.

## Future Improvements

* Add location-based disposal guidance
* Build a web interface (Streamlit)
* Expand dataset with local environmental regulations

## Author

Larissa Soares
