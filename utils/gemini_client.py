import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiClient:
    """
    Client for interacting with the Google Gemini Pro LLM.
    """
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("GeminiClient initialized with 'gemini-pro' model.")

    def answer_question(self, document_text: str, question: str) -> dict:
        """
        Answers a question based on the provided document text using Gemini Pro.

        Args:
            document_text: The full text content of the PDF document.
            question: The natural language question to answer.

        Returns:
            A dictionary containing 'answer' and 'reasoning'.
        """
        prompt = f"""
        You are an AI assistant specialized in understanding policy documents.
        Your task is to answer the given question based *only* on the provided policy document content.
        If the answer is not explicitly stated in the document, you must clearly state that you cannot find the answer in the document.
        Provide a concise answer and, if possible, a brief reasoning or the exact phrase from the document that supports your answer.

        ---
        Policy Document Content:
        {document_text}
        ---

        Question: "{question}"

        Please provide your answer in the following format:
        Answer: <Your concise answer here>
        Reasoning: <Brief reasoning or supporting text from the document, or "Not found in document" if applicable>
        """
        try:
            logger.info(f"Sending question to Gemini: '{question}'")
            response = self.model.generate_content(prompt)

            # Extract answer and reasoning from the response text
            answer_text = response.text
            answer_lines = answer_text.split('\n')
            
            answer = "Answer not found or could not be extracted."
            reasoning = "Reasoning not found or could not be extracted."

            for line in answer_lines:
                if line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
                elif line.startswith("Reasoning:"):
                    reasoning = line.replace("Reasoning:", "").strip()
            
            # Refine answer if LLM indicates it's not found
            if "not explicitly stated in the document" in answer.lower() or \
               "cannot find the answer" in answer.lower() or \
               "not found in the document" in answer.lower():
                answer = "Answer not found in the document."
                reasoning = "The information required to answer this question was not explicitly found in the provided document."

            logger.info(f"Gemini response for '{question}': Answer='{answer}', Reasoning='{reasoning}'")
            return {"answer": answer, "reasoning": reasoning}

        except genai.types.BlockedPromptException as e:
            logger.error(f"Prompt was blocked for question '{question}': {e}")
            return {"answer": "The prompt was blocked by the safety system.", "reasoning": str(e)}
        except Exception as e:
            logger.error(f"Error calling Gemini API for question '{question}': {e}")
            return {"answer": f"An error occurred while processing the question: {e}", "reasoning": "Error during LLM inference."}

