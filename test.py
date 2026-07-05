# # # # Test code for document ingestion and analysis using a PDFHandler and DocumentAnalyzer

# import os
# from pathlib import Path
# from src.document_analyzer.data_ingestion import DocumentHandler       # Your PDFHandler class
# from src.document_analyzer.data_analysis import DocumentAnalyzer  # Your DocumentAnalyzer class

# # Path to the PDF you want to test
# PDF_PATH = r"/Users/nanimahi/Document_Portal/notebook/data/sample.pdf"

# # Dummy file wrapper to simulate uploaded file (Streamlit style)
# class DummyFile:
#     def __init__(self, file_path):
#         self.name = Path(file_path).name
#         self._file_path = file_path

#     def getbuffer(self):
#         return open(self._file_path, "rb").read()

# def main():
#     try:
#         # ---------- STEP 1: DATA INGESTION ----------
#         print("Starting PDF ingestion...")
#         dummy_pdf = DummyFile(PDF_PATH)

#         handler = DocumentHandler(session_id="test_ingestion_analysis")
        
#         saved_path = handler.save_pdf(dummy_pdf)
#         print(f"PDF saved at: {saved_path}")

#         text_content = handler.read_pdf(saved_path)
#         text_content = text_content[:8000] 
#         print(f"Extracted text length: {len(text_content)} chars\n")

#         # ---------- STEP 2: DATA ANALYSIS ----------
#         print("Starting metadata analysis...")
#         analyzer = DocumentAnalyzer()  # Loads LLM + parser
        
#         analysis_result = analyzer.analyze_document(text_content)

#         # ---------- STEP 3: DISPLAY RESULTS ----------
#         print("\n=== METADATA ANALYSIS RESULT ===")
#         for key, value in analysis_result.items():
#             print(f"{key}: {value}")

#     except Exception as e:
#         print(f"Test failed: {e}")

# if __name__ == "__main__":
#     main()
    
    


## Testing code for document comparison using LLMs

# import io
# from pathlib import Path
# from src.document_compare.data_ingestion import DocumentIngestion
# from src.document_compare.document_comparator import DocumentComparatorLLM

# # ---- Setup: Load local PDF files as if they were "uploaded" ---- #
# def load_fake_uploaded_file(file_path: Path):
#     return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()

# # ---- Step 1: Save and combine PDFs ---- #
# def test_compare_documents():
#     ref_path = Path("/Users/nanimahi/Document_Portal/data/document_compare/Long_Report_V1-1.pdf")
#     act_path = Path("/Users/nanimahi/Document_Portal/data/document_compare/Long_Report_V2-1.pdf")

#     # Wrap them like Streamlit UploadedFile-style
#     class FakeUpload:
#         def __init__(self, file_path: Path):
#             self.name = file_path.name
#             self._buffer = file_path.read_bytes()

#         def getbuffer(self):
#             return self._buffer

#     # Instantiate
#     comparator = DocumentIngestion()
#     ref_upload = FakeUpload(ref_path)
#     act_upload = FakeUpload(act_path)

#     # Save files and combine
#     ref_file, act_file = comparator.save_uploaded_files(ref_upload, act_upload)
#     combined_text = comparator.combine_documents()
#     comparator.clean_old_sessions(keep_latest=3)

#     print("\n Combined Text Preview (First 1000 chars):\n")
#     print(combined_text[:1000])

#     # ---- Step 2: Run LLM comparison ---- #
#     llm_comparator = DocumentComparatorLLM()
#     df = llm_comparator.compare_documents(combined_text)
    
#     print("\n Comparison DataFrame:\n")
#     print(df)

# if __name__ == "__main__":
#     test_compare_documents()
    
        



# Testing code for document chat functionality

# import sys
# from pathlib import Path
# from langchain_community.vectorstores import FAISS
# from src.single_document_chat.data_ingestion import SingleDocIngestor
# from src.single_document_chat.retrieval import ConversationalRAG
# from utils.model_loader import ModelLoader

# FAISS_INDEX_PATH = Path("faiss_index")

# def test_conversational_rag_on_pdf(pdf_path:str, question:str):
#     try:
#         model_loader = ModelLoader()
        
#         if FAISS_INDEX_PATH.exists():
#             print("Loading existing FAISS index...")
#             embeddings = model_loader.load_embeddings()
#             vectorstore = FAISS.load_local(folder_path=str(FAISS_INDEX_PATH), embeddings=embeddings,allow_dangerous_deserialization=True)
#             retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#         else:
#             # Step 2: Ingest document and create retriever
#             print("FAISS index not found. Ingesting PDF and creating index...")
#             with open(pdf_path, "rb") as f:
#                 uploaded_files = [f]
#                 ingestor = SingleDocIngestor()
#                 retriever = ingestor.ingest_files(uploaded_files)
                
#         print("Running Conversational RAG...")
#         session_id = "test_conversational_rag"
#         rag = ConversationalRAG(retriever=retriever, session_id=session_id)
#         response = rag.invoke(question)
#         print(f"\nQuestion: {question}\nAnswer: {response}")
                    
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
    
# if __name__ == "__main__":
#     # Example PDF path and question
#     pdf_path = "/Users/nanimahi/Document_Portal/data/single_document_chat/sample.pdf"
#     question = "What is the significance of the attention mechanism? can you explain it in simple terms?"

#     if not Path(pdf_path).exists():
#         print(f"PDF file does not exist at: {pdf_path}")
#         sys.exit(1)
    
#     # Run the test
#     test_conversational_rag_on_pdf(pdf_path, question)


# testing for multidoc chat
# import sys
# from pathlib import Path
# from src.multi_document_chat.data_ingestion import DocumentIngestor
# from src.multi_document_chat.retrieval import ConversationalRAG

# def test_document_ingestion_and_rag():
#     try:
#         test_files = [
#             "/Users/nanimahi/Document_Portal/data/multi_doc_chat/market_analysis_report.docx",
#             "/Users/nanimahi/Document_Portal/data/multi_doc_chat/NIPS-2017-attention-is-all-you-need-Paper.pdf"
#             "/Users/nanimahi/Document_Portal/data/multi_doc_chat/sample.txt"
#             "/Users/nanimahi/Document_Portal/data/multi_doc_chat/state_of_the_union.txt"
              
            
#         ]
        
#         uploaded_files = [] 
        
#         for file_path in test_files:
#             if Path(file_path).exists():
#                 uploaded_files.append(open(file_path, "rb"))
#             else:
#                 print(f"File does not exist: {file_path}")
                
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#         ingestor = DocumentIngestor()
        
#         retriever = ingestor.ingest_files(uploaded_files)
        
#         for f in uploaded_files:
#             f.close()
                
#         session_id = "test_multi_doc_chat"
        
#         rag = ConversationalRAG(session_id=session_id, retriever=retriever)
        
#         question = "what is President Zelenskyy said in their speech in parliament?"
        
#         answer=rag.invoke(question)
        
#         print("\n Question:", question)
        
#         print("Answer:", answer)
        
#         if not uploaded_files:
#             print("No valid files to upload.")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"Test failed: {str(e)}")
#         sys.exit(1)
        
# if __name__ == "__main__":
#     test_document_ingestion_and_rag()
    


import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from src.document_ingestion.data_ingestion import ChatIngestor
from src.document_chat.retrieval import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()


def test_document_ingestion_and_rag():
    try:
        test_files = [
            "/Users/nanimahi/Document_Portal/data/state_of_the_union(2).txt",
        ]

        uploaded_files = []

        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

        # Build index using single-module ChatIngestor
        ci = ChatIngestor(temp_base="data", faiss_base="faiss_index", use_session_dirs=True)
        
        # Using MMR (Maximal Marginal Relevance) for diverse results
        # MMR parameters:
        # - fetch_k: Number of documents to fetch before MMR re-ranking (20)
        # - lambda_mult: Diversity parameter (0=max diversity, 1=max relevance, 0.5=balanced)
        retriever = ci.built_retriver(
            uploaded_files, 
            chunk_size=200, 
            chunk_overlap=20, 
            k=5,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )
        
        # Alternative: Use similarity search instead of MMR
        # retriever = ci.built_retriver(uploaded_files, chunk_size=200, chunk_overlap=20, k=5, search_type="similarity")

        # Close file handles
        for f in uploaded_files:
            try:
                f.close()
            except Exception:
                pass

        session_id = ci.session_id
        index_dir = os.path.join("faiss_index", session_id)

        # Load RAG with MMR search
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(
            index_path=index_dir, 
            k=5, 
            index_name=os.getenv("FAISS_INDEX_NAME", "index"),
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5
        )

        # Interactive multi-turn chat loop
        chat_history = []
        print("\nType 'exit' to quit the chat.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break

            answer = rag.invoke(user_input, chat_history=chat_history)
            print("Assistant:", answer)

            # Maintain conversation history for context in subsequent turns
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

        if not uploaded_files:
            print("No valid files to upload.")
            sys.exit(1)

    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_document_ingestion_and_rag()