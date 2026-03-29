import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. DEFINE THE STATE (The Agent's Memory)
# ==========================================
# This dictionary structure holds the conversation history and tracking metadata
# as the data moves through the graph.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    file_path: str
    task_type: str
    current_status: str

# ==========================================
# 2. DEFINE THE NODES (The Execution Engines)
# ==========================================

def supervisor_node(state: AgentState):
    """
    The 'Brain': Analyzes the user prompt and routes to the correct scientific tool.
    In the full GSoC build, this will use an LLM. For this POC, it uses keyword routing.
    """
    last_message = state["messages"][-1].content.lower()
    
    # Routing Logic
    if "fits" in last_message or "galaxy" in last_message or "astronomy" in last_message:
        next_step = "astrophysics_tool"
    elif "csv" in last_message or "trackml" in last_message or "physics" in last_message:
        next_step = "physics_tool"
    else:
        next_step = "end"
        
    return {"task_type": next_step, "current_status": "Routed by Supervisor"}

def physics_execution_node(state: AgentState):
    """The 'Hands': High Energy Physics (3D PointNet++)."""
    print(f"\n[EXECUTION] Routing to Physics Tool for: {state.get('file_path', 'unknown_file')}")
    print(" -> Initializing 3D PointNet++ Engine...")
    print(" -> Simulating TrackML trajectory clustering...")
    
    # In production, this imports and runs your PointNet++ Colab code
    response = AIMessage(content="Physics 3D trajectory clustering complete. Manifold separated successfully.")
    return {"messages": [response], "current_status": "Physics execution completed"}

def astrophysics_execution_node(state: AgentState):
    """The 'Hands': Astrophysics (2D YOLOv10 & FITS)."""
    print(f"\n[EXECUTION] Routing to Astrophysics Tool for: {state.get('file_path', 'unknown_file')}")
    print(" -> Initializing Astropy Log-Stretch Engine...")
    print(" -> Simulating YOLOv10-S morphological classification...")
    
    # In production, this imports and runs your YOLO/Astropy Colab code
    response = AIMessage(content="Astronomy FITS log-stretch and bounding box generation complete.")
    return {"messages": [response], "current_status": "Astrophysics execution completed"}

# ==========================================
# 3. BUILD THE GRAPH (The Architecture)
# ==========================================
workflow = StateGraph(AgentState)

# Add the specific AI nodes to the graph
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("physics_tool", physics_execution_node)
workflow.add_node("astrophysics_tool", astrophysics_execution_node)

# Add the edges (How data flows from start to finish)
workflow.add_edge(START, "supervisor")

# Conditional routing based on the supervisor's decision
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["task_type"],
    {
        "physics_tool": "physics_tool",
        "astrophysics_tool": "astrophysics_tool",
        "end": END
    }
)

# After a scientific tool finishes processing, the workflow ends
workflow.add_edge("physics_tool", END)
workflow.add_edge("astrophysics_tool", END)

# Compile the graph into an executable application
app = workflow.compile()

# ==========================================
# 4. TERMINAL PROOF OF CONCEPT (Test Run)
# ==========================================
if __name__ == "__main__":
    print("Initializing SciVision-Agent Router...\n")
    
    # --- TEST 1: Astrophysics Routing ---
    print("--- TEST 1: User requests FITS processing ---")
    initial_state_astro = {
        "messages": [HumanMessage(content="Please process this new galaxy_sample.fits file.")],
        "file_path": "/data/galaxy_sample.fits",
        "task_type": "",
        "current_status": "started"
    }
    
    for output in app.stream(initial_state_astro):
        for key, value in output.items():
            print(f"Node '{key}' finished. Status: {value.get('current_status', 'N/A')}")

    print("\n" + "="*50 + "\n")

    # --- TEST 2: High Energy Physics Routing ---
    print("--- TEST 2: User requests TrackML processing ---")
    initial_state_physics = {
        "messages": [HumanMessage(content="Run clustering on the new TrackML collision CSV.")],
        "file_path": "/data/collision_event_102.csv",
        "task_type": "",
        "current_status": "started"
    }
    
    for output in app.stream(initial_state_physics):
        for key, value in output.items():
            print(f"Node '{key}' finished. Status: {value.get('current_status', 'N/A')}")