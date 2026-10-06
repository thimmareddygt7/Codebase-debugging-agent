class CitationBuilder:

    def build_citations(
        self,
        results
    ) -> list[str]:

        metadatas = results["metadatas"][0]

        citations = []

        for i, metadata in enumerate(metadatas):

            file_path = metadata["file_path"]
            chunk_name = metadata["name"]
            start_line = metadata["start_line"]
            end_line = metadata["end_line"]

            citation = (
                f"[SOURCE {i + 1}] "
                f"{file_path} "
                f"-> {chunk_name} "
                f"(lines {start_line}-{end_line})"
            )

            citations.append(citation)

        return citations

    def format_citations(
        self,
        citations: list[str]
    ) -> str:

        if not citations:
            return "No sources available."

        return "\n".join(citations)