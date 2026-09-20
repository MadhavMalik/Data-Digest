"""Signal Engine — an AI-guided scientific relationship discovery engine.

Core idea:

    HYPOTHESIZE -> COMPUTE -> OBSERVE -> REFINE -> TRANSFORM -> COMPUTE
    -> VISUALIZE -> INTERPRET -> REMEMBER -> RETRIEVE -> REINTERPRET

The LLM never sees the raw dataset.  It sees compact column/dataset cards and
compact statistical feedback, and it acts as a hypothesis generator and research
planner.  A deterministic numerical engine decides what is actually true.
Elasticsearch is the persistent semantic *evidence memory*, not the statistics
engine.
"""

__version__ = "0.1.0"
