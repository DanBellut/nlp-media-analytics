NLP Media Analytics

An automated Python data pipeline that transforms unorganized media feeds, articles, and customer feedback into structured, quantitative insights using Large Language Models (LLMs).

Instead of relying on rigid word-matching tools that fail to grasp context, the pipeline uses an LLM to extract context-aware sentiment and topics, turning raw text into data that marketing and PR teams can act on.

Status: Early prototype. The core LLM analysis and visualization work today; the hybrid NLP preprocessing, interactive dashboards and local model support are planned. 

Features

Structured AI Extraction: Uses the OpenAI API (`gpt-4o-mini`) in strict JSON mode with `temperature=0` to reliably capture context-aware sentiment scores (-1.0 to 1.0) and topic classification.
Validated Output: Sentiment values are clamped to the valid range, and topics outside the allowed list are rejected instead of silently corrupting the data.
Prompt Injection Protection:Instructions and analyzed text are sent as separate messages, so the input cannot override the prompt.
pandas Integration:AI responses are mapped into structured DataFrames, ready for aggregation and further analysis.
Publication-Ready Charts: Seaborn and Matplotlib visualize the average sentiment per topic on a fixed scale with a neutral baseline.
Modular Design: Analysis and visualization are separate, reusable functions.

Tech Stack

Language: Python 3.10+
Data Engineering: pandas
AI & Semantics: OpenAI API (`gpt-4o-mini`)
Visualization: Matplotlib, Seaborn




