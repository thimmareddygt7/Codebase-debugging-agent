from pathlib import Path


class CodebaseLoader:

    def __init__(self, root_path: str):

        self.root_path = Path(root_path)

        self.ignore_dirs = {
            ".git",
            "__pycache__",
            "venv",
            ".venv",
            "env",
            ".env",
            "node_modules",
            "testings",
        }

        self.supported_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
        }

    def get_source_files(self) -> list[Path]:

        source_files = []

        for path in self.root_path.rglob("*"):

            if not path.is_file():
                continue

            if any(
                part in self.ignore_dirs
                for part in path.parts
            ):
                continue

            if path.suffix in self.supported_extensions:
                source_files.append(path)

        return sorted(source_files)