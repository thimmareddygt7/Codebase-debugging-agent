class CitationVerifier:

    def verify(
        self,
        results,
        citations: list[str]
    ) -> bool:

        metadatas = results["metadatas"][0]

        if len(citations) != len(metadatas):
            return False

        for i, metadata in enumerate(metadatas):

            expected_source = f"[SOURCE {i + 1}]"

            if expected_source not in citations[i]:
                return False

            file_path = metadata["file_path"]
            chunk_name = metadata["name"]
            start_line = metadata["start_line"]
            end_line = metadata["end_line"]

            if file_path not in citations[i]:
                return False

            if chunk_name not in citations[i]:
                return False

            expected_lines = (
                f"(lines {start_line}-{end_line})"
            )

            if expected_lines not in citations[i]:
                return False

        return True