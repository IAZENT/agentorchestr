#!/usr/bin/env python3
"""
agents/agent_registry.py — Mode Selection + Capability Ranking
================================================================
Inspired by: Claude Code sub-agent routing, OpenAI agent-as-tools

Selects optimal agent mix and determines orchestration mode
based on available agents and their capabilities.
"""



class AgentRegistry:
    """Selects optimal agent mix and determines orchestration mode."""

    def __init__(self, available_agents: list[dict]):
        self.available = available_agents

    def select_best(self, agents: list[dict], n_workers: int = 3) -> list[dict]:
        """
        Pick the best N agents for the job.
        Ensures diversity when possible (different agents for different strengths).
        Falls back to N instances of the same agent when only 1 is available.
        """
        if not agents:
            return []

        n_workers = max(1, min(n_workers, 8))  # Cap at 8 (tmux pane overhead)

        # Separate CLI agents (parallel-capable) from IDE agents (not parallel)
        cli_agents = [a for a in agents if a["type"] in ("sdk", "cli")]
        ide_agents = [a for a in agents if a["type"] in ("ide", "gh")]

        if not cli_agents:
            # Only IDE agents — limited mode, pick best one
            return [self._rank_agents(ide_agents)[0]] if ide_agents else []

        # Rank CLI agents by orchestration capability
        ranked = self._rank_agents(cli_agents)

        if len(ranked) == 1:
            # Single agent: return N copies (single_agent_multi mode)
            return [ranked[0]] * n_workers

        # Multiple agents: pick top N different agents for diversity
        selected = []
        seen_names = set()
        for agent in ranked:
            if agent["name"] not in seen_names:
                selected.append(agent)
                seen_names.add(agent["name"])
                if len(selected) >= n_workers:
                    break

        # If we need more workers than unique agents, rotate
        while len(selected) < n_workers:
            for agent in ranked:
                if len(selected) >= n_workers:
                    break
                selected.append(agent)

        return selected

    def determine_mode(self, selected_agents: list[dict]) -> str:
        """
        Determine orchestration mode based on selected agents.

        Returns:
            "multi_agent" — 2+ different CLI agents (best)
            "single_agent_multi" — 1 type of CLI agent, N instances
            "hybrid" — CLI agents + IDE agents
            "ide_only" — only IDE agents (limited)
        """
        if not selected_agents:
            return "ide_only"

        cli_agents = [a for a in selected_agents if a["type"] in ("sdk", "cli")]
        ide_agents = [a for a in selected_agents if a["type"] in ("ide", "gh")]

        if not cli_agents:
            return "ide_only"

        unique_names = set(a["name"] for a in cli_agents)

        if ide_agents:
            return "hybrid"

        if len(unique_names) >= 2:
            return "multi_agent"

        return "single_agent_multi"

    def _rank_agents(self, agents: list[dict]) -> list[dict]:
        """Rank agents by capability score."""
        def score(agent):
            caps = agent.get("capabilities", {})
            orchestrate = caps.get("orchestrate", 5)
            implement = caps.get("implement", 5)
            parallel = caps.get("parallel", False)
            mcp = caps.get("mcp", False)
            # Priority: orchestrate > parallel > implement > mcp
            return (orchestrate * 3 + (10 if parallel else 0) + implement * 2 + (5 if mcp else 0))

        return sorted(agents, key=score, reverse=True)

    def get_worker_for_task(self, task: dict, selected_agents: list[dict]) -> dict:
        """
        Pick the best worker for a specific task.
        Uses agent_hint from the task if available, otherwise round-robin.
        """
        hint = task.get("agent_hint", "any")

        if hint and hint != "any":
            # Try to match the hint to an agent name
            for agent in selected_agents:
                if agent["name"] == hint:
                    return agent

        # Fallback: pick agent with highest implement score
        def implement_score(agent):
            return agent.get("capabilities", {}).get("implement", 5)

        return max(selected_agents, key=implement_score)

    def get_mcp_agents(self, selected_agents: list[dict]) -> list[dict]:
        """Get agents that support MCP for IDE integration."""
        return [a for a in selected_agents if a.get("capabilities", {}).get("mcp")]
