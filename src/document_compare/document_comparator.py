import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse, PromptType


class DocumentComparatorLLM:
    def __init__(self, loader: ModelLoader | None = None):
        load_dotenv()

        # Dependency Injection (DI)
        self.loader = loader or ModelLoader()
        self.llm = self.loader.load_llm()

        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]

        # Use fixing parser
        self.chain = self.prompt | self.llm | self.fixing_parser

        log.info("DocumentComparatorLLM initialized")

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        # Safety
        if not combined_docs or not isinstance(combined_docs, str):
            raise DocumentPortalException("combined_docs must be a non-empty string", sys)

        inputs = {
            "combined_docs": combined_docs,
            "format_instruction": self.parser.get_format_instructions(),
        }

        try:
            log.info("Invoking document comparison LLM chain")
            response = self.chain.invoke(inputs)
            log.info("Chain invoked successfully", response_preview=str(response)[:200])
            return self._format_response(response)

        except Exception as e:
            log.error("Error in compare_documents", error=str(e))
            raise DocumentPortalException("Error comparing documents", sys)

    def _format_response(self, response_parsed) -> pd.DataFrame:
        try:
            # Normalise response into something DataFrame-friendly
            if response_parsed is None:
                raise ValueError("LLM parser returned None")

            # If it's a Pydantic model (SummaryResponse), convert safely
            if hasattr(response_parsed, "model_dump"):
                payload = response_parsed.model_dump()
            else:
                payload = response_parsed

            # If payload is a dict, wrap it; if list, use directly
            if isinstance(payload, dict):
                payload = [payload]

            if not isinstance(payload, list):
                raise TypeError(f"Expected list or dict, got {type(payload)}")

            return pd.DataFrame(payload)

        except Exception as e:
            log.error("Error formatting response into DataFrame", error=str(e))
            raise DocumentPortalException("Error formatting response", sys)
