# Field Notes RAG

This Streamlit app indexes the PDF reports in the project folder, splits extracted text into overlapping chunks, retrieves the passages most relevant to a question, and uses a grounded prompt to answer from those passages.

## Run it

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app works without an API key by showing the retrieved evidence. Set `OPENAI_API_KEY` to have an OpenAI model synthesize the answer. `OPENAI_MODEL` is optional and defaults to `gpt-4.1-mini`.

## Chunk-size decision

The default is **800 tokens per chunk with 120 tokens of overlap**. These files are report-style PDFs with narrative paragraphs, headings, findings, and recommendations. An 800-token window generally keeps one argument or finding together while avoiding the unrelated material that appears in larger windows. The overlap preserves context when a section crosses a boundary.

Use 400-600 tokens for short, highly specific facts or table-heavy documents. Use 1,000-1,200 tokens when answers depend on long explanations or multi-paragraph methodology. The sidebar sliders let you test this against the corpus without changing code.

## RAG flow

1. Extract page text with `pypdf`.
2. Split each page into word windows approximating the selected token size.
3. Build a TF-IDF index with unigram and bigram features.
4. Retrieve the top matching passages.
5. Put those passages into a prompt that requires evidence-only answers and numbered citations.