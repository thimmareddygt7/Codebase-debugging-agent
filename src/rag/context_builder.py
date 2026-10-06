class CodeContextBuilder:

    def build_context(self, results) -> str:

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        context_parts = []

        for i in range(len(documents)):

            metadata = metadatas[i]
            document = documents[i]
            distance = distances[i]

            file_path = metadata["file_path"]
            chunk_name = metadata["name"]
            start_line = metadata["start_line"]
            end_line = metadata["end_line"]

            context = (
                f"SOURCE {i + 1}\n"
                f"File: {file_path}\n"
                f"Chunk: {chunk_name}\n"
                f"Lines: {start_line}-{end_line}\n"
                f"Distance: {distance:.4f}\n"
                f"\n"
                f"{document}\n"
            )

            context_parts.append(context)

        return "\n" + "\n".join(context_parts)