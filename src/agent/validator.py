import ast
import re
from typing import Tuple, List

class FixValidator:
    """Validates code snippets proposed by the agent."""

    @staticmethod
    def extract_python_code_blocks(text: str) -> List[str]:
        """Extracts python code blocks from markdown formatting."""
        pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            # Fallback for generic markdown blocks
            pattern_generic = r"```\s*(.*?)\s*```"
            matches = re.findall(pattern_generic, text, re.DOTALL)
        return matches

    def validate_code_syntax(self, code_snippet: str) -> Tuple[bool, str]:
        """Checks if a python code block can be parsed by Python's AST parser."""
        try:
            ast.parse(code_snippet)
            return True, "Code passed syntax validation."
        except SyntaxError as err:
            return False, f"SyntaxError on line {err.lineno}: {err.msg}"
        except Exception as err:
            return False, f"Parsing error: {str(err)}"

    def evaluate_proposed_fix(self, answer_text: str) -> Tuple[bool, str]:
        """
        Validates the entire answer. If code blocks are present,
        verifies that every Python block contains valid syntax.
        """
        code_blocks = self.extract_python_code_blocks(answer_text)

        # If no code block was provided, check if it's purely an explanation
        if not code_blocks:
            return True, "No code blocks to validate; explanation accepted."

        for idx, block in enumerate(code_blocks, start=1):
            valid, feedback = self.validate_code_syntax(block)
            if not valid:
                return False, f"Code block #{idx} failed syntax check: {feedback}"

        return True, "All proposed code blocks passed validation."