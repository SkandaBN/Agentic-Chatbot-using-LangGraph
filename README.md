# 🤖 Agentic Chatbot using LangGraph

> **A stateful, tool-using Agentic AI chatbot built with LangGraph, Google Gemini, RAG, persistent memory, Human-in-the-Loop (HITL), Streamlit, Docker, and GitHub Actions CI/CD on AWS EC2.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-orange)](https://www.langchain.com/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/Deployment-AWS%20EC2-FF9900?logo=amazon-aws)](https://aws.amazon.com/ec2/)
[![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)](LICENSE)

---

## 📌 Overview

**Agentic Chatbot using LangGraph** is a stateful conversational AI application designed to go beyond a traditional question-answering chatbot.

Instead of simply sending every user query to an LLM, the system uses **LangGraph to orchestrate an agentic workflow** in which the LLM can decide whether it should:

* Answer the question directly
* Search the web for current information
* Retrieve information from an uploaded PDF
* Perform mathematical calculations
* Retrieve current stock prices
* Retrieve real-time weather information
* Request human approval before executing a stock-purchase simulation
* Maintain conversation state across multiple chat threads

The application combines **LLM reasoning, tool calling, Retrieval-Augmented Generation (RAG), persistent state, Human-in-the-Loop interaction, streaming responses, and production-style containerized deployment**.

---

# 🧠 Architecture

The core application is implemented as a **LangGraph StateGraph** containing a chatbot node and a ToolNode.

The graph follows a cyclic agentic pattern:

```text
                         ┌──────────────────────┐
                         │      User Query      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Streamlit UI       │
                         │   app.py             │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     LangGraph        │
                         │     StateGraph       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      chat_node       │
                         │  Gemini 2.5 Flash    │
                         └──────────┬───────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     No tool needed       Tool required
                          │                   │
                          ▼                   ▼
                   ┌────────────┐     ┌────────────────┐
                   │ Final      │     │   ToolNode     │
                   │ Response   │     └───────┬────────┘
                   └────────────┘             │
                                              ▼
                              ┌─────────────────────────────────┐
                              │          Available Tools        │
                              ├─────────────────────────────────┤
                              │ 🔎 Tavily Web Search            │
                              │ 📄 PDF RAG / FAISS              │
                              │ 🧮 Calculator                   │
                              │ 📈 Stock Price                  │
                              │ 🌦️ Current Weather              │
                              │ 🧑‍💻 Human-in-the-Loop Purchase │
                              └────────────────┬────────────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │   chat_node     │
                                      │   continues     │
                                      └────────┬────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │ Final Response  │
                                      └─────────────────┘

                 ┌─────────────────────────────────────┐
                 │       Persistent Conversation       │
                 │                                     │
                 │ SQLite + LangGraph SqliteSaver      │
                 │                                     │
                 │ thread_id → conversation state      │
                 └─────────────────────────────────────┘
```

The actual LangGraph implementation creates `chat_node` and `tools`, connects `START → chat_node`, uses `tools_condition` for conditional tool routing, and loops `tools → chat_node`. The compiled graph uses `SqliteSaver` for persistence.

---

# 🍪 Cookiecutter-Style Project Structure

The repository currently follows a lightweight structure rather than a full Python package layout:

```text
Agentic-Chatbot-using-LangGraph/
│
├── .github/
│   └── workflows/
│       └── cicd.yaml              # GitHub Actions CI/CD pipeline
│
├── .dockerignore                  # Docker build exclusions
├── .gitignore                     # Git exclusions
│
├── app.py                         # Streamlit frontend
│
├── backend.py                     # LangGraph agent + tools + RAG + persistence
│
├── requirements.txt               # Python dependencies
│
├── Dockerfile                     # Container configuration
│
├── LICENSE                        # Apache 2.0 License
│
└── README.md                      # Project documentation
```

### Backend responsibilities

```text
backend.py
│
├── Environment configuration
│
├── Gemini LLM
│   └── gemini-2.5-flash
│
├── Gemini Embeddings
│   └── gemini-embedding-001
│
├── RAG Pipeline
│   ├── PyPDFLoader
│   ├── RecursiveCharacterTextSplitter
│   ├── GoogleGenerativeAIEmbeddings
│   ├── FAISS
│   └── Retriever
│
├── Agent Tools
│   ├── Tavily Search
│   ├── Calculator
│   ├── Stock Price
│   ├── Weather
│   ├── PDF RAG
│   └── Human-in-the-Loop Stock Purchase
│
├── LangGraph State
│   └── ChatState
│
├── LangGraph Nodes
│   ├── chat_node
│   └── ToolNode
│
├── Conditional Routing
│   └── tools_condition
│
└── Persistence
    └── SQLite + SqliteSaver
```

---

# ✨ Key Features

## 1. 🧠 Agentic Decision Making

The chatbot uses **Google Gemini 2.5 Flash** with tool binding.

The LLM determines whether a query can be answered directly or requires an external capability.

For example:

```text
User
 │
 ├── "Explain machine learning"
 │        └── Direct LLM response
 │
 ├── "What is the weather in Bangalore?"
 │        └── Weather Tool
 │
 ├── "Calculate 125 * 48"
 │        └── Calculator Tool
 │
 ├── "What is Apple's current stock price?"
 │        └── Stock Tool
 │
 ├── "What does my uploaded PDF say about RAG?"
 │        └── RAG Tool
 │
 └── "Buy 10 shares of AAPL"
          └── Human Approval
```

The system prompt explicitly instructs the agent which tool to use for each category of request.

---

# 2. 🔄 LangGraph Agentic Workflow

The chatbot is implemented using:

* `StateGraph`
* `START`
* `ToolNode`
* `tools_condition`
* Conditional edges
* Cyclic graph execution
* State persistence

The workflow is:

```text
START
  │
  ▼
chat_node
  │
  ├─────────────── No Tool ───────────────► END
  │
  └─────────────── Tool Required
                    │
                    ▼
                 ToolNode
                    │
                    ▼
                chat_node
                    │
                    └──────────────► END
```

This cyclic architecture allows the agent to perform a tool call, receive its result, and then return to the LLM for final response generation.

---

# 3. 📄 Retrieval-Augmented Generation (RAG)

The chatbot supports **PDF-based question answering**.

### RAG Pipeline

```text
Upload PDF
    │
    ▼
PyPDFLoader
    │
    ▼
Document Extraction
    │
    ▼
Recursive Character Text Splitter
    │
    ├── Chunk Size: 1000
    └── Chunk Overlap: 200
    │
    ▼
Gemini Embeddings
(gemini-embedding-001)
    │
    ▼
FAISS Vector Store
    │
    ▼
Similarity Retriever
    │
    └── Top K = 4
    │
    ▼
Relevant Document Chunks
    │
    ▼
Gemini
    │
    ▼
Grounded Answer
```

The implementation uses `PyPDFLoader`, `RecursiveCharacterTextSplitter`, Google Gemini embeddings and FAISS. Retrieved chunks include their source and page metadata before being passed back to the agent.

---

# 4. 🔎 Real-Time Web Search

The chatbot integrates **Tavily Search** for queries requiring current or internet-based information.

Configuration:

```text
Maximum Results: 5
Topic: general
Search Depth: advanced
```

The agent is instructed to use the search tool for current events and recent information.

---

# 5. 🧮 Calculator Tool

A dedicated calculator tool handles mathematical expressions.

Supported operations include:

```text
+
-
*
/
sqrt()
abs()
round()
min()
max()
sum()
```

The calculator restricts Python evaluation by disabling built-ins and exposing only an explicitly allowed set of mathematical operations.

---

# 6. 📈 Stock Price Tool

The chatbot can retrieve the latest stock quote for a supplied ticker symbol through the Alpha Vantage Global Quote endpoint.

Example:

```text
User:
"What is the current price of AAPL?"

Agent:
        ↓
get_stock_price("AAPL")
        ↓
Alpha Vantage
        ↓
Stock Quote
        ↓
Gemini
        ↓
Final Response
```

---

# 7. 🌦️ Real-Time Weather Tool

The weather tool uses **OpenWeather** to retrieve current weather information.

The workflow performs:

```text
Location
   ↓
Geocoding
   ↓
Latitude / Longitude
   ↓
Current Weather API
   ↓
Temperature
Feels Like
Humidity
Pressure
Wind Speed
Visibility
   ↓
Formatted Response
```

The implementation also handles missing API keys, timeouts, HTTP errors, connection errors, and unexpected API responses.

---

# 8. 🧑‍💻 Human-in-the-Loop (HITL)

One of the important agentic capabilities of the project is **Human-in-the-Loop execution**.

The `purchase_stock` tool does not immediately execute the simulated transaction.

Instead:

```text
User requests stock purchase
          │
          ▼
      Gemini Agent
          │
          ▼
   purchase_stock()
          │
          ▼
   LangGraph interrupt
          │
          ▼
┌───────────────────────────┐
│ Human Approval Required   │
│                           │
│ Approve?                  │
│                           │
│      YES / NO             │
└─────────────┬─────────────┘
              │
        ┌─────┴─────┐
        │           │
       YES          NO
        │           │
        ▼           ▼
   Purchase       Cancel
   Simulated      Action
        │           │
        └─────┬─────┘
              ▼
        Resume Graph
              │
              ▼
        Final Response
```

LangGraph's `interrupt()` pauses execution and the Streamlit frontend later resumes the same thread using `Command(resume=decision)`.

---

# 9. 💾 Persistent Conversation Memory

The chatbot maintains conversation state using:

```text
SQLite
   +
SqliteSaver
   +
LangGraph thread_id
```

Each conversation receives a unique UUID-based `thread_id`.

```text
User
 │
 ▼
New Conversation
 │
 ▼
UUID Thread ID
 │
 ▼
LangGraph State
 │
 ▼
SQLite Checkpoint
 │
 ▼
Conversation persists
```

The application can retrieve previous threads and restore their stored message history.

---

# 10. 💬 Streamlit Chat Interface

The frontend is implemented using **Streamlit**.

The UI supports:

* Chat interface
* Multiple conversation threads
* New conversation creation
* Previous conversation loading
* Streaming assistant responses
* Tool execution status
* PDF upload
* RAG document ingestion
* Human approval interaction
* Persistent conversation state

The frontend communicates directly with the LangGraph backend imported from `backend.py`.

---

# 🛠️ Technology Stack

| Category             | Technology                        |
| -------------------- | --------------------------------- |
| Programming Language | Python 3.11                       |
| Agent Framework      | LangGraph                         |
| LLM Framework        | LangChain                         |
| LLM                  | Google Gemini 2.5 Flash           |
| Embeddings           | Gemini Embedding 001              |
| RAG                  | LangChain + FAISS                 |
| PDF Processing       | PyPDF                             |
| Text Splitting       | RecursiveCharacterTextSplitter    |
| Web Search           | Tavily                            |
| Weather              | OpenWeather                       |
| Stock Data           | Alpha Vantage                     |
| Agent State          | LangGraph StateGraph              |
| Persistence          | SQLite + SqliteSaver              |
| Human-in-the-Loop    | LangGraph interrupt / Command     |
| Frontend             | Streamlit                         |
| Containerization     | Docker                            |
| CI/CD                | GitHub Actions                    |
| Container Registry   | Docker Hub                        |
| Cloud                | AWS EC2                           |
| Deployment Runner    | Self-hosted GitHub Actions runner |
| Observability        | LangSmith                         |
| License              | Apache 2.0                        |

The dependency file confirms the core LangGraph, Gemini, Tavily, FAISS, Streamlit, SQLite checkpointing, PDF, and text-splitting packages used by the project.

---

# 📁 Application Flow

```text
                   ┌───────────────────┐
                   │      User         │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │    Streamlit      │
                   │      app.py       │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │    LangGraph      │
                   │    StateGraph     │
                   └─────────┬─────────┘
                             │
                             ▼
                   ┌───────────────────┐
                   │    Gemini LLM     │
                   │  chat_node        │
                   └─────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
          Direct Answer              Tool Required
                │                         │
                │                         ▼
                │                  ┌──────────────┐
                │                  │   ToolNode   │
                │                  └──────┬───────┘
                │                         │
                │       ┌─────────────────┼─────────────────┐
                │       │                 │                 │
                │       ▼                 ▼                 ▼
                │     RAG             Web Search        Calculator
                │       │                 │                 │
                │       ▼                 ▼                 ▼
                │     FAISS           Tavily            Math
                │
                │       ┌─────────────────┼─────────────────┐
                │       │                 │                 │
                │       ▼                 ▼                 ▼
                │    Weather          Stocks             HITL
                │       │                 │                 │
                │       └─────────────────┴─────────────────┘
                │                         │
                │                         ▼
                │                   chat_node again
                │                         │
                └─────────────────────────┘
                                          │
                                          ▼
                                  Final AI Response

                              ┌─────────────────┐
                              │ SQLite          │
                              │ Checkpointer    │
                              └─────────────────┘
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/SkandaBN/Agentic-Chatbot-using-LangGraph.git

cd Agentic-Chatbot-using-LangGraph
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key

TAVILY_API_KEY=your_tavily_api_key

OPENWEATHER_API_KEY=your_openweather_api_key

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=agentic-chatbot-project
```

> **Important:** Never commit your `.env` file or expose API keys in source code.

---

# ▶️ Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🐳 Docker Deployment

The repository contains a production-oriented Dockerfile based on:

```text
python:3.11-slim
```

It installs the required system dependencies for FAISS, installs the Python dependencies, exposes port `8501`, and starts Streamlit on `0.0.0.0`.

### Build the image

```bash
docker build -t agentic-chatbot .
```

### Run the container

```bash
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY="your_key" \
  -e TAVILY_API_KEY="your_key" \
  -e OPENWEATHER_API_KEY="your_key" \
  -e LANGSMITH_TRACING="true" \
  -e LANGSMITH_ENDPOINT="https://api.smith.langchain.com" \
  -e LANGSMITH_API_KEY="your_key" \
  -e LANGSMITH_PROJECT="agentic-chatbot-project" \
  agentic-chatbot
```

Open:

```text
http://localhost:8501
```

---

# ☁️ AWS EC2 Deployment

The project is configured for deployment on an **AWS EC2 instance** using a Docker container and a GitHub Actions self-hosted runner.

### Deployment architecture

```text
Developer
    │
    │ git push
    ▼
┌───────────────────────┐
│      GitHub           │
│      Repository       │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│   GitHub Actions      │
│                       │
│  Continuous           │
│  Integration          │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Docker Build & Push   │
│                       │
│      Docker Hub       │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ AWS EC2               │
│                       │
│ Self-hosted Runner    │
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│ Docker Container      │
│                       │
│ Streamlit :8501       │
└──────────┬────────────┘
           │
           ▼
        Users
```

The GitHub Actions workflow currently performs CI, builds and pushes the Docker image to Docker Hub, then deploys it to an AWS EC2 self-hosted runner. It also verifies that the container is running and checks the Streamlit health endpoint.

---

# 🔄 CI/CD Pipeline

The GitHub Actions pipeline follows three major stages:

### Stage 1 — Continuous Integration

```text
Checkout Repository
        ↓
Lint
        ↓
Unit Test Step
```

### Stage 2 — Build & Push

```text
Checkout
   ↓
Validate Docker Configuration
   ↓
Docker Buildx
   ↓
Docker Hub Login
   ↓
Build Docker Image
   ↓
Push Image
```

The workflow tags the image using both `main` and the Git commit SHA.

### Stage 3 — Continuous Deployment

```text
AWS EC2 Self-hosted Runner
            ↓
     Docker Hub Login
            ↓
       Pull Image
            ↓
 Stop Existing Container
            ↓
   Start New Container
            ↓
      Port 8501
            ↓
   Container Health Check
            ↓
      Deployment Complete
```

The deployment uses the container name `stapp`, maps port `8501:8501`, and verifies Streamlit's internal health endpoint before considering deployment successful.

---

# 🔐 GitHub Actions Secrets

The CI/CD pipeline expects secrets such as:

```text
DOCKER_USERNAME
DOCKER_PASSWORD
IMAGE_NAME

GOOGLE_API_KEY
TAVILY_API_KEY
OPENWEATHER_API_KEY

LANGSMITH_TRACING
LANGSMITH_ENDPOINT
LANGSMITH_API_KEY
LANGSMITH_PROJECT
```

The workflow validates required application secrets before deploying and injects them into the Docker container at runtime.

---

# 📊 Observability

The application can be integrated with **LangSmith** for tracing and monitoring agent executions.

Configuration:

```env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=agentic-chatbot-project
```

This enables better visibility into:

* LLM calls
* Tool execution
* Agent trajectories
* Latency
* Errors
* LangGraph workflow execution

---

# 🧩 Core Agent Components

| Component             | Responsibility                           |
| --------------------- | ---------------------------------------- |
| `ChatState`           | Maintains conversation messages          |
| `chat_node`           | LLM reasoning and tool selection         |
| `ToolNode`            | Executes selected tools                  |
| `tools_condition`     | Routes execution depending on tool calls |
| `SqliteSaver`         | Persists LangGraph checkpoints           |
| `rag_tool`            | Retrieves information from uploaded PDFs |
| `search_tool`         | Performs web search                      |
| `calculator`          | Performs mathematical calculations       |
| `get_stock_price`     | Retrieves stock quote                    |
| `get_current_weather` | Retrieves current weather                |
| `purchase_stock`      | Demonstrates Human-in-the-Loop execution |

---

# 🔬 Agentic Workflow Example

### Example: PDF Question

```text
User:
"What does the uploaded PDF say about transformers?"

              ↓

        Gemini Agent
              ↓
        Needs document
              ↓
          rag_tool
              ↓
       FAISS Retriever
              ↓
      Top 4 relevant chunks
              ↓
        Tool Response
              ↓
        Gemini Agent
              ↓
       Final grounded answer
```

### Example: Current Weather

```text
User
 │
 ▼
Gemini
 │
 └──► get_current_weather()
            │
            ▼
       OpenWeather
            │
            ▼
      Weather Result
            │
            ▼
         Gemini
            │
            ▼
      Final Response
```

### Example: Human Approval

```text
User
 │
 ▼
"Buy 10 AAPL shares"
 │
 ▼
Gemini
 │
 ▼
purchase_stock()
 │
 ▼
LangGraph interrupt()
 │
 ▼
Human Approval
 │
 ├── YES ──► Purchase simulated
 │
 └── NO  ──► Purchase cancelled
 │
 ▼
Resume Graph
 │
 ▼
Final Response
```

---

# 🎯 Why LangGraph?

Traditional chatbot:

```text
User → LLM → Response
```

This project:

```text
User
 ↓
State
 ↓
LLM
 ↓
Decision
 ↓
Tool
 ↓
Observation
 ↓
LLM
 ↓
Decision
 ↓
Response
```

LangGraph is particularly useful here because it provides explicit state, conditional routing, cyclic workflows, persistence, and Human-in-the-Loop execution—capabilities that fit naturally with multi-step agentic applications.

---

# 🏗️ Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

* Agentic AI
* LLM tool calling
* LangGraph state machines
* Cyclic agent workflows
* Conditional routing
* Retrieval-Augmented Generation
* Vector similarity search
* FAISS
* Embeddings
* Persistent conversational memory
* Thread-based state management
* Human-in-the-Loop
* Streaming responses
* External API integration
* Prompt-based tool selection
* Docker containerization
* GitHub Actions
* CI/CD
* Docker Hub
* AWS EC2
* Self-hosted runners
* Application health checks
* LangSmith observability

---

# 📌 Project Highlights

### AI / Agentic Layer

```text
Google Gemini
      +
LangChain
      +
LangGraph
      +
Tool Calling
      +
Conditional Routing
      +
HITL
```

### Knowledge Layer

```text
PDF
 ↓
PyPDF
 ↓
Chunking
 ↓
Gemini Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
RAG Tool
```

### Memory Layer

```text
Conversation
     ↓
LangGraph State
     ↓
Thread ID
     ↓
SQLite
     ↓
SqliteSaver
```

### Deployment Layer

```text
GitHub
 ↓
GitHub Actions
 ↓
Docker Build
 ↓
Docker Hub
 ↓
AWS EC2
 ↓
Self-hosted Runner
 ↓
Docker Container
 ↓
Streamlit
```

---

# 🧪 Example Queries

Try the following after launching the application:

### General conversation

```text
Explain the difference between AI and Generative AI.
```

### Mathematics

```text
Calculate 125 * 48 + sqrt(144).
```

### Web search

```text
What are the latest developments in artificial intelligence?
```

### Weather

```text
What is the current weather in Bangalore?
```

### Stock

```text
What is the current stock price of AAPL?
```

### PDF RAG

```text
Upload a PDF and ask:
"What are the main concepts discussed in this document?"
```

### Human-in-the-Loop

```text
Buy 10 shares of AAPL.
```

The system should pause and request human approval before completing the simulated purchase.

---

# 🔮 Future Enhancements

Potential next steps for the project include:

* [ ] Multi-agent architecture
* [ ] PostgreSQL production checkpointer
* [ ] Redis-based caching
* [ ] Authentication and authorization
* [ ] Better tool permission management
* [ ] Additional financial tools
* [ ] More document formats
* [ ] Multi-document RAG
* [ ] Hybrid search
* [ ] Reranking
* [ ] Conversation summarization
* [ ] Long-term memory
* [ ] Automated evaluation
* [ ] LangSmith-based agent evaluation
* [ ] Automated test suite
* [ ] Prometheus/Grafana monitoring
* [ ] HTTPS with a reverse proxy
* [ ] AWS load balancing
* [ ] Infrastructure as Code using Terraform
* [ ] Production-grade secrets management using AWS Secrets Manager

---

# 👨‍💻 Author

**Skanda B N**

AI / ML • Generative AI • Agentic AI • LangChain • LangGraph • Cloud & MLOps

---

# 📄 License

This project is licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Repository:**
https://github.com/SkandaBN/Agentic-Chatbot-using-LangGraph
