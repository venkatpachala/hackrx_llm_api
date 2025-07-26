# HACKRX 6.0 - LLM Document Query API

This FastAPI service lets you upload a PDF file and ask questions about its contents. It extracts text using PyMuPDF and forwards the question along with the extracted text to Gemini Pro. All requests require a bearer token matching the `API_TOKEN` environment variable.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```
