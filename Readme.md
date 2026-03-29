# SciVision-Agent: An Autonomous Multi-Domain Orchestrator

**Organization:** ML4SCI (Machine Learning for Science)  
**Project Category:** AI for Science and Research / Agentic AI (Task 2c)  
**Applicant:** Heet  

---

## 🏗️ Repository State & GSoC 2026 Vision

This repository is my submission for the Task 2c architecture requirements. It serves as the initial blueprint for the SciVision-Agent framework.

**Currently Working (Proof of Concept):**
Instead of pushing a bunch of empty placeholder folders, I've set up this branch as a working proof of concept. It contains the actual execution layers and routing logic we need to get started:
1. **`routing_graph.py`**: A functional LangGraph prototype. It handles state management and correctly routes tasks between the physics and astronomy nodes based on user input.
2. **`research_notebooks/`**: This folder holds the mathematically validated tensor operations. You will find the 3D PointNet++ TrackML clustering (saved at Epoch 19) and the 2D FITS log-stretch normalization script here.

**GSoC 2026 Objective:**
If I am selected for the summer program, my main goal under ML4SCI's mentorship is to connect these standalone pieces. I plan to build out the core agents so they can autonomously run the HEP scripts (like HEPTAPOD), catch and fix PyTorch errors on the fly, and use a local RAG memory to turn the output data into readable scientific reports.

---

## 1. Project Abstract
Right now, a lot of scientific machine learning involves running single-domain scripts linearly, which requires a significant amount of manual debugging and data synthesis. My proposal is the SciVision-Agent—an end-to-end, multi-agent framework built with LangGraph and CrewAI. 

Instead of just being a standard pipeline, this acts more like an automated research assistant. It takes raw data, figures out which deep-learning tool needs to process it, handles the inevitable tensor errors that pop up during execution, and eventually synthesizes the results. To show that this architecture can work across different fields, I am proposing we integrate two very different tools: a 3D point cloud clusterer for High Energy Physics, and a 2D morphological classifier for Astrophysics.

## 2. Proposed Production Architecture
Here is the directory structure I am proposing for the final GSoC deliverable. The main goal here is to strictly separate the LLM reasoning loop from the heavy PyTorch tensor operations.

```text
scivision_agent_framework/
│
├── agent_entrypoint.py          # Master script to launch the framework
├── requirements.txt             
│
├── core_agents/                 # The decision-making layer
│   ├── __init__.py
│   ├── supervisor_agent.py      # Triages prompts and routes to the correct tool
│   ├── research_analyst.py      # Synthesizes model outputs into reports
│   └── debugger_agent.py        # Catches tensor shape errors and NaN losses
│
├── orchestration/               # The LangGraph logic
│   ├── __init__.py
│   ├── state_schema.py          # Defines the memory payloads 
│   └── routing_graph.py         # The cyclical routing logic 
│
├── knowledge_base/              # RAG memory
│   ├── __init__.py
│   ├── prompts.yaml             # System instructions
│   └── vector_store/            # ChromaDB embeddings of scientific papers
│
└── execution_tools/             # Domain-specific AI models
    ├── __init__.py
    ├── physics_hep/             
    │   ├── pointnet_trackml.py  # 3D TrackML trajectory clustering
    │   └── calorimeter_gan.py   # Future HEPTAPOD integration
    │
    └── astrophysics/            
        └── yolo_fits_engine.py  # FITS log-stretch & YOLO classification