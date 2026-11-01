from src.ingestion.loader import CodebaseLoader


loader = CodebaseLoader(r"C:\Users\thimm\Desktop\AIML\Codebase-debugging-agent")

files = loader.get_source_files()

for file in files:
    print(file)