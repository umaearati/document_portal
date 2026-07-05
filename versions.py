import importlib.metadata
packages = [
    "langchain",
    "python-dotenv",
    "ipykernel",
    "langchain_groq",
    "langchain_google_genai",
    "langchain-community",
    "faiss-cpu",
    "structlog",
    "PyMuPDF",
    "pylint",
    "langchain-core",
    "pytest",
    "streamlit",
    "fastapi",
    "uvicorn",
    "python-multipart",
    "docx2txt",
    "pypdf"
]
for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} (not installed)")
        
        
        

# python -c "import versions; print(versions.__file__)" - imports the versions.py file and prints its location, allowing you to verify that the correct file is being used.
# uv pip freeze > requirements.txt - exports all currently installed Python packages (and their versions) into a requirements.txt file.