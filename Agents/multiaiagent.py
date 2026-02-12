#!/usr/bin/env python
# coding: utf-8

# ===================================
# Multi-Agent Architecture Module
# Exports: simple_agent, supervisor_agent
# ===================================

import os
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from operator import add
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model

# ===================================
# Environment Setup
# ===================================
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# LangSmith Tracing
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "MultiAgent-Project"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# ===================================
# LLM
# ===================================
llm = init_chat_model("groq:llama-3.1-8b-instant")


# ===================================
# Tool
# ===================================
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return str(results)


# ==========================================================
# GRAPH 1: Simple Multi-Agent (Researcher → Tools → Writer)
# ==========================================================

def make_simple_agent():
    """Builds the simple multi-agent graph with tool calling."""

    class AgentState(MessagesState):
        next_agent: str

    def researcher_agent(state: AgentState):
        """Researcher agent that searches for information"""
        messages = state["messages"]
        system_msg = SystemMessage(
            content="You are a research assistant. Use the search_web tool to find information about the user's request."
        )
        researcher_llm = llm.bind_tools([search_web])
        response = researcher_llm.invoke([system_msg] + messages)
        return {
            "messages": [response],
            "next_agent": "writer"
        }

    def writer_agent(state: AgentState):
        """Writer agent that creates summaries"""
        messages = state["messages"]

        # Extract the original user query
        user_query = next(
            (m.content for m in messages if isinstance(m, HumanMessage)),
            "the topic"
        )

        # Collect all research findings (AI messages + tool results)
        research_findings = []
        for m in messages:
            if isinstance(m, AIMessage) and m.content:
                research_findings.append(m.content)
            elif hasattr(m, "name") and m.content:  # ToolMessage
                research_findings.append(m.content)

        combined_research = "\n\n".join(research_findings)

        # Create a fresh prompt for the writer — NOT passing raw AI messages
        writer_prompt = f"""Based on the following research about "{user_query}",create a well-structured, polished summary report.
                            RESEARCH FINDINGS:
                            {combined_research}
                            Format the report with:
                            - A brief executive summary
                            - Key findings with bullet points
                            - Key takeaways
                            - Conclusion
                            You MUST write a complete report."""

        system_msg = SystemMessage(
            content="You are a professional technical writer. Create clear, well-structured reports."
        )
        response = llm.invoke([system_msg, HumanMessage(content=writer_prompt)])
        return {
            "messages": [response],
            "next_agent": "end"
        }

    def route_after_researcher(state):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "writer"

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("tools", ToolNode([search_web]))
    workflow.add_node("writer", writer_agent)

    workflow.set_entry_point("researcher")
    workflow.add_conditional_edges("researcher", route_after_researcher, {
        "tools": "tools",
        "writer": "writer"
    })
    workflow.add_edge("tools", "researcher")
    workflow.add_edge("writer", END)

    return workflow.compile()


# ==========================================================
# GRAPH 2: Supervisor Multi-Agent (Supervisor → R/A/W loop)
# ==========================================================

def make_supervisor_agent():
    """Builds the supervisor multi-agent graph."""

    class SupervisorState(MessagesState):
        next_agent: str
        research_data: str
        analysis: str
        final_report: str
        task_complete: bool
        current_task: str

    # Cache the supervisor chain — only create it once
    supervisor_chain = None

    def get_supervisor_chain():
        nonlocal supervisor_chain
        if supervisor_chain is None:
            supervisor_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a supervisor managing a team of agents:
                        1. Researcher - Gathers information and data
                        2. Analyst - Analyzes data and provides insights  
                        3. Writer - Creates reports and summaries

                        Based on the current state and conversation, decide which agent should work next.
                        If the task is complete, respond with 'DONE'.

                            Current state:
                        - Has research data: {has_research}
                        - Has analysis: {has_analysis}
                        - Has report: {has_report}

                        Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.
                """),
                ("human", "{task}")
            ])
            supervisor_chain = supervisor_prompt | llm
        return supervisor_chain

    def supervisor_node(state: SupervisorState) -> Dict:
        """Supervisor decides next agent using Groq LLM"""
        messages = state["messages"]
        
        # Always get the LATEST human message as the task
        latest_task = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                latest_task = m.content
                break
        
        if not latest_task:
            latest_task = "No task"
        
        previous_task = state.get("current_task", "")
        
        # If this is a NEW task (different from previous), reset the pipeline
        is_new_task = previous_task and previous_task != latest_task
        
        if is_new_task:
            has_research = False
            has_analysis = False
            has_report = False
        else:
            has_research = bool(state.get("research_data", ""))
            has_analysis = bool(state.get("analysis", ""))
            has_report = bool(state.get("final_report", ""))

        chain = get_supervisor_chain()
        decision = chain.invoke({
            "task": latest_task,
            "has_research": has_research,
            "has_analysis": has_analysis,
            "has_report": has_report
        })

        decision_text = decision.content.strip().lower()
        print(f"[Supervisor Decision]: {decision_text}")

        if "done" in decision_text or has_report:
            next_agent = "end"
            # Deliver the final report to the user
            report = state.get("final_report", "")
            if report:
                supervisor_msg = f"✅ Supervisor: All tasks complete! Here is your report:\n\n{report}"
            else:
                supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
        elif "researcher" in decision_text or not has_research:
            next_agent = "researcher"
            supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
        elif "analyst" in decision_text or (has_research and not has_analysis):
            next_agent = "analyst"
            supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
        elif "writer" in decision_text or (has_analysis and not has_report):
            next_agent = "writer"
            supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
        else:
            next_agent = "end"
            supervisor_msg = "✅ Supervisor: Task seems complete."

        result = {
            "messages": [AIMessage(content=supervisor_msg)],
            "next_agent": next_agent,
            "current_task": latest_task
        }
        
        # Reset old state if this is a new task
        if is_new_task:
            result["research_data"] = ""
            result["analysis"] = ""
            result["final_report"] = ""
        
        return result

    def researcher_node(state: SupervisorState) -> Dict:
        """Researcher uses tool calling to search the web"""
        task = state.get("current_task", "research topic")
        messages = state["messages"]

        # Only pass the user query + any tool messages from current loop
        # Filter out supervisor/analyst/writer noise
        relevant_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                relevant_messages.append(m)
            elif hasattr(m, "tool_calls") and m.tool_calls:
                # Previous researcher response with tool calls (for multi-step)
                relevant_messages.append(m)
            elif hasattr(m, "name") and hasattr(m, "tool_call_id"):
                # ToolMessage — result from search_web
                relevant_messages.append(m)

        system_msg = SystemMessage(
            content=f"""You are a research specialist. Use the search_web tool to find 
            comprehensive information about: {task}

            Include key facts, current trends, statistics, and case studies.
            You MUST use the search_web tool to get real data."""
        )

        researcher_llm = llm.bind_tools([search_web])
        response = researcher_llm.invoke([system_msg] + relevant_messages)

        return {
            "messages": [response],
            "next_agent": "supervisor"
        }

    def process_research_results(state: SupervisorState) -> Dict:
        """Extract ONLY tool results as research data — ignore supervisor messages"""
        messages = state["messages"]
        research_findings = []
        for m in messages:
            # Only collect ToolMessage content (search results)
            if hasattr(m, "tool_call_id") and m.content:
                research_findings.append(m.content)
        
        # Also get the researcher's final summary (last AIMessage with content, no tool_calls)
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content and not (hasattr(m, "tool_calls") and m.tool_calls):
                # Skip supervisor messages
                if not m.content.startswith(("📋", "✅", "✍️", "📊")):
                    research_findings.append(m.content)
                    break

        research_data = "\n\n".join(research_findings) if research_findings else ""
        return {"research_data": research_data}

    def route_after_researcher(state):
        """Route to tools if researcher made tool calls, otherwise done researching"""
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "continue"

    def analyst_node(state: SupervisorState) -> Dict:
        """Analyst analyzes the research data"""
        research_data = state.get("research_data", "")
        task = state.get("current_task", "")

        analysis_prompt = f"""As a data analyst, analyze this research data and provide insights:
                        Research Data:

                        {research_data}

                        Provide:
                        1. Key insights and patterns
                        2. Strategic implications
                        3. Risks and opportunities
                        4. Recommendations

                        Focus on actionable insights related to: {task}"""

        analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = analysis_response.content

        return {
            "messages": [AIMessage(content=f"📊 Analyst: Analysis complete.\n\n{analysis}")],
            "analysis": analysis,
            "next_agent": "supervisor"
        }

    def writer_node(state: SupervisorState) -> Dict:
        """Writer creates the final report from research + analysis"""
        research_data = state.get("research_data", "")
        analysis = state.get("analysis", "")
        task = state.get("current_task", "")

        writing_prompt = f"""You are a professional business analyst and technical writer. 
                        Create a comprehensive executive report.

                        <task>
                        {task}
                        </task>

                        <research_findings>
                        {research_data}
                        </research_findings>

                        <analysis>
                        {analysis}
                        </analysis>

                        <instructions>
                        Create a professional executive report with the following EXACT structure:

                        ## EXECUTIVE SUMMARY
                        Provide a concise 2-3 paragraph overview of the key findings and recommendations.

                        ## KEY FINDINGS
                        List 4-6 major findings from the research.

                        ## DETAILED ANALYSIS
                        Provide in-depth analysis addressing what the data reveals, patterns, and implications.

                        ## RECOMMENDATIONS
                        Present 3-5 actionable recommendations.

                        ## CONCLUSION
                        Summarize the main takeaways and next steps.
                        </instructions>

                        Generate the report now following the exact structure above."""

        report_response = llm.invoke([HumanMessage(content=writing_prompt)])
        report = report_response.content

        final_report = f"""
                        {'='*50}
                        EXECUTIVE REPORT
                        {'='*50}

                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        **Topic: {task}**

                        {'='*50}

                        {report}

                        {'='*50}
                        Report compiled by Multi-Agent AI System
                        {'='*50}
                        """

        return {
            "messages": [AIMessage(content="✍️ Writer: Professional report generated successfully!")],
            "final_report": final_report,
            "next_agent": "supervisor",
            "task_complete": True
        }

    def router(state: SupervisorState) -> Literal["researcher", "analyst", "writer", "__end__"]:
        """Routes based on next_agent set by supervisor"""
        next_agent = state.get("next_agent", "end")

        if next_agent == "end":
            return END

        if next_agent in ["researcher", "analyst", "writer"]:
            return next_agent

        return END

    # =====================
    # Build the graph
    # =====================
    workflow = StateGraph(SupervisorState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", ToolNode([search_web]))
    workflow.add_node("process_research", process_research_results)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor → routes to researcher/analyst/writer/END
    workflow.add_conditional_edges("supervisor", router, {
        "researcher": "researcher",
        "analyst": "analyst",
        "writer": "writer",
        END: END
    })

    # Researcher → tools (if tool calls) or process_research (if done)
    workflow.add_conditional_edges("researcher", route_after_researcher, {
        "tools": "tools",
        "continue": "process_research"
    })

    # Tools → back to researcher (for follow-up tool calls)
    workflow.add_edge("tools", "researcher")

    # Process research → back to supervisor
    workflow.add_edge("process_research", "supervisor")

    # Analyst → always goes back to supervisor
    workflow.add_edge("analyst", "supervisor")

    # Writer → always goes back to supervisor (supervisor delivers the report)
    workflow.add_edge("writer", "supervisor")

    return workflow.compile()


# ===================================
# Export compiled graphs
# ===================================
simple_agent = make_simple_agent()
supervisor_agent = make_supervisor_agent()
