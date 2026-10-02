# Dubizzle AI Smart Filter (Backend & Agent System)

An AI-native backend service that powers the Dubizzle Smart Filter extension. It leverages advanced Agent orchestration to enable users to filter classifieds listings using natural language, supporting both complex text-based constraints and highly specific visual features.

## Architecture & System AI Harness

This repository implements a robust **System AI Harness** that bridges the gap between raw LLM capabilities and reliable, production-grade features. 

![System Architecture](Dubizzle%20AI%20Extension%20System%20Design.svg)

### Key Features
- **Agent Orchestration (LangGraph)**: The system utilizes a stateful graph to manage multi-step reasoning. It seamlessly routes user intents, extracting structural filters while retaining subjective constraints for AI evaluation.
- **Tool Use & Planning**: The LLM acts as an intelligent router and evaluator. It determines when a query requires strictly visual inspection vs text-based semantic filtering.
- **Graceful Degradation & Fallbacks**: Built with maintainability and operational stability in mind. The system features a robust, multi-tier fallback mechanism across multiple model providers. If the primary model experiences downtime or rate limits, traffic is seamlessly routed to fallback models (OpenRouter, Groq, OpenAI) without interrupting the user experience.
- **Multimodal AI Vision (Gemini)**: If an intent requires purely visual inspection, the backend extracts up to 10 images from the listing and passes them to a multimodal LLM (Gemini Vision) for evaluation. It uses Chain of Thought prompting and a fail-safe "default to keep" rule to ensure high accuracy without frustrating false negatives.
- **Robust Data Flows**: The system is designed to handle edge cases gracefully. If an image fails to download or the vision model errors out, the system defaults to keeping the listing (prioritizing false positives over false negatives) to protect the user experience.

## Why I Made These Choices

- **Cost Efficiency & Architecture**: Processing images natively using multimodal LLMs can be expensive. By relying on a lightweight Agent to parse intents and perform structural/text DOM filtering first, the expensive Vision pipeline is only triggered when absolutely necessary, achieving a balance between cost-efficiency and premium UX.
- **Maintainability**: Abstracting model providers and using environment variables ensures the codebase is flexible and not tied to a single vendor. The LangGraph architecture makes adding new tools or steps trivial.
- **End-to-End Ownership**: This project demonstrates full-stack ownership, from prompt engineering and Agent orchestration to performance optimization and deploying a scalable FastAPI backend (designed for serverless/Modal deployment).

## Build Requirements

- **Python 3.11+**
- **Modal CLI** (for cloud deployment)
- **Dependencies**: `fastapi`, `modal`, `langgraph`, `pydantic`, `Pillow`, `google-genai`, `openai`, `tenacity`.

## Deployment & Setup

This service is designed to be deployed on [Modal](https://modal.com/), leveraging their serverless infrastructure for low-latency, horizontal scaling and secure secret management.

1. Define your secrets in Modal (e.g., `custom-secret`) to store your API keys securely.
2. Install the Modal CLI:
   ```bash
   pip install modal
   modal setup
   ```
3. Serve the application for local development with hot-reloading:
   ```bash
   modal serve main.py
   ```
4. Deploy to production:
   ```bash
   modal deploy main.py
   ```
