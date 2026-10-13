import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ask_codebase_agent

def main():
    print("Testing LangGraph Codebase Agent Migration...")
    query = "Where is CodebaseLoader defined and what extensions does it support?"
    
    print(f"\nUser Question: {query}\n" + "=" * 60)
    result = ask_codebase_agent(query)

    print("\n--- Execution Finished ---")
    print(f"Tool Calls Made: {len(result.get('tool_calls', []))}")
    for tc in result.get('tool_calls', []):
        print(f" -> Called: {tc['function']['name']} with args: {tc['function'].get('arguments')}")
    
    print("\nFinal Answer:")
    print(result.get("final_answer"))
    
    if result.get("error"):
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()