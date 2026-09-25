An automated Python data pipeline that transforms unorganized media feeds, articles, and customer feedback into structured, quantitative insights using a hybrid approach of classic NLP and Large Language Models (LLMs). 
Instead of relying on rigid, word-matching tools that fail to grasp context, this application automatically extracts deep semantic meaning, tracks emerging themes, and generates interactive corporate dashboards. 
It bridges the gap between raw text streams and executive-ready marketing and PR intelligence.

Features

Hybrid NLP Preprocessing: Leverages lightweight Python libraries for efficient text cleaning, tokenization, and metadata parsing to minimize downstream token usage and API costs.
Structured AI Extraction: Utilizes the OpenAI API with strict JSON formatting to reliably capture context-aware sentiment analysis, complex topic classification, and high-accuracy keyword tracking.
Production-Ready Data Integration: Automatically maps unstructured AI responses into structured pandas DataFrames, allowing for immediate aggregation, time-series analysis, and database storage.
Dual-Layer Visualization: Generates publication-ready static charts with Seaborn and Matplotlib for formal reporting, alongside interactive, exploratory dashboards built with Plotly.
Modular Pipeline Design: Built with fully decoupled modules for API handling, data cleaning, and visualization, making the architecture easily adaptable to local open-source models like Llama 3 via Ollama.

Tech Stack

Language: Python 3.10+
Data Engineering: pandas, NumPy
AI & Semantics: OpenAI API (gpt-4o-mini), spaCy / NLTK
Visualization: Matplotlib, Seaborn, Plotly



