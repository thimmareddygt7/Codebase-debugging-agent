from dataclasses import dataclass
import ast
from pathlib import Path


@dataclass
class CodeChunk:
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    class_name: str | None
    start_line: int
    end_line: int
    code: str


class CodebaseChunker:

    def _get_node_lines_and_code(
        self, source_lines: list[str], node: ast.AST
    ) -> tuple[int, int, str]:
        """Extract start line (including decorators), end line, and code segment."""
        # Include decorator start line if decorators are present
        if hasattr(node, "decorator_list") and node.decorator_list:
            start_line = node.decorator_list[0].lineno
        else:
            start_line = getattr(node, "lineno", 1)

        end_line = getattr(node, "end_lineno", start_line)
        code_segment = "\n".join(source_lines[start_line - 1 : end_line])

        return start_line, end_line, code_segment

    def chunk_file(self, file_path: str | Path) -> list[CodeChunk]:
        file_path_str = str(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            source_code = file.read()

        source_lines = source_code.splitlines()
        tree = ast.parse(source_code)

        chunks: list[CodeChunk] = []

        def _traverse(
            nodes: list[ast.AST],
            parent_qual_name: str = "",
            current_class_name: str | None = None,
        ):
            for node in nodes:
                # ==============================
                # CLASS DEFINITION
                # ==============================
                if isinstance(node, ast.ClassDef):
                    name = node.name
                    qual_name = (
                        f"{parent_qual_name}.{name}" if parent_qual_name else name
                    )
                    start_line, end_line, code = self._get_node_lines_and_code(
                        source_lines, node
                    )

                    class_chunk = CodeChunk(
                        file_path=file_path_str,
                        language="python",
                        chunk_type="class",
                        name=name,
                        qualified_name=qual_name,
                        class_name=current_class_name,
                        start_line=start_line,
                        end_line=end_line,
                        code=code,
                    )
                    chunks.append(class_chunk)

                    # Recurse into class body with updated qual_name and current_class_name
                    _traverse(
                        node.body,
                        parent_qual_name=qual_name,
                        current_class_name=name,
                    )

                # ==============================
                # FUNCTION / METHOD DEFINITION
                # ==============================
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name
                    qual_name = (
                        f"{parent_qual_name}.{name}" if parent_qual_name else name
                    )
                    chunk_type = "method" if current_class_name else "function"
                    start_line, end_line, code = self._get_node_lines_and_code(
                        source_lines, node
                    )

                    func_chunk = CodeChunk(
                        file_path=file_path_str,
                        language="python",
                        chunk_type=chunk_type,
                        name=name,
                        qualified_name=qual_name,
                        class_name=current_class_name,
                        start_line=start_line,
                        end_line=end_line,
                        code=code,
                    )
                    chunks.append(func_chunk)

                    # Recurse into function body for nested functions/classes
                    _traverse(
                        node.body,
                        parent_qual_name=qual_name,
                        current_class_name=None,
                    )

                # ==============================
                # CONTROL FLOW BLOCKS (if, try, with, for, while)
                # ==============================
                else:
                    if hasattr(node, "body") and isinstance(node.body, list):
                        _traverse(
                            node.body,
                            parent_qual_name=parent_qual_name,
                            current_class_name=current_class_name,
                        )
                    if hasattr(node, "orelse") and isinstance(node.orelse, list):
                        _traverse(
                            node.orelse,
                            parent_qual_name=parent_qual_name,
                            current_class_name=current_class_name,
                        )
                    if hasattr(node, "finalbody") and isinstance(node.finalbody, list):
                        _traverse(
                            node.finalbody,
                            parent_qual_name=parent_qual_name,
                            current_class_name=current_class_name,
                        )

        _traverse(tree.body)
        return chunks

    # ==========================================
    # CHUNK ENTIRE CODEBASE
    # ==========================================
    def chunk_codebase(self, file_paths: list) -> list[CodeChunk]:

        all_chunks = []

        for file_path in file_paths:

            try:
                chunks = self.chunk_file(file_path)
                print(f"{file_path} -> {len(chunks)} chunks")
                all_chunks.extend(chunks)

            except (SyntaxError, UnicodeDecodeError) as error:
                print(f"Skipping {file_path}: {error}")

        return all_chunks