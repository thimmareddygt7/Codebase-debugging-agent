from src.retrieval.retriever import CodeRetriever


class CodeSearchTool:

    NAME = "code_search"

    DESCRIPTION = (
        "Search the indexed codebase for relevant source code. "
        "Use this tool when you need to find classes, functions, "
        "methods, files, or implementation details in the codebase."
    )

    REQUIRED_RESULT_FIELDS = [
        "file_path",
        "chunk_name",
        "qualified_name",
        "start_line",
        "end_line",
        "distance",
        "code",
    ]

    def __init__(
        self,
        persist_directory: str = "vector_store"
    ):

        self.retriever = CodeRetriever(
            persist_directory=persist_directory
        )

    def validate_arguments(
        self,
        query: str,
        top_k: int = 5
    ) -> None:

        if not isinstance(query, str):

            raise TypeError(
                "query must be a string."
            )

        if not query.strip():

            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(top_k, int):

            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

    def validate_result(
        self,
        result: dict
    ) -> None:

        for field in self.REQUIRED_RESULT_FIELDS:

            if field not in result:

                raise ValueError(
                    f"Search result is missing required "
                    f"field: {field}"
                )

        if not isinstance(
            result["file_path"],
            str
        ):

            raise TypeError(
                "file_path must be a string."
            )

        if not isinstance(
            result["chunk_name"],
            str
        ):

            raise TypeError(
                "chunk_name must be a string."
            )

        if not isinstance(
            result["qualified_name"],
            str
        ):

            raise TypeError(
                "qualified_name must be a string."
            )

        if not isinstance(
            result["start_line"],
            int
        ):

            raise TypeError(
                "start_line must be an integer."
            )

        if not isinstance(
            result["end_line"],
            int
        ):

            raise TypeError(
                "end_line must be an integer."
            )

        if not isinstance(
            result["code"],
            str
        ):

            raise TypeError(
                "code must be a string."
            )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        self.validate_arguments(
            query=query,
            top_k=top_k
        )

        results = self.retriever.search(
            query=query,
            top_k=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        search_results = []

        for i in range(len(documents)):

            metadata = metadatas[i]

            result = {
                "file_path": metadata["file_path"],
                "chunk_name": metadata["name"],
                "qualified_name": metadata["qualified_name"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "distance": distances[i],
                "code": documents[i],
            }

            self.validate_result(
                result
            )

            search_results.append(
                result
            )

        return search_results

    def execute(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:

        return self.search(
            query=query,
            top_k=top_k
        )

    def get_tool_definition(self) -> dict:

        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The search question or description "
                                "of the code you want to find."
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": (
                                "Number of relevant code chunks "
                                "to retrieve."
                            ),
                            "default": 5,
                        },
                    },
                    "required": [
                        "query"
                    ],
                },
            },
        }