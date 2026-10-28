# NotebookLM-like Processing Architecture

Status: implemented baseline  
Scope: AIS Knowledge conversational RAG on Databricks Apps

## 1. What is reverse-engineered vs inferred

This design copies **observable product behavior**, not Google's private internal implementation.

Official Gemini Notebook / NotebookLM behavior used as the target:

- Chat answers are grounded in notebook sources and provide citations back to supporting source passages.
- Conversation history is used for chat context.
- Sources can include PDFs, Google Docs, Google Slides, Google Sheets, DOCX, TXT, Markdown, CSV, PPTX, images, audio, websites and YouTube transcripts.
- Google Drive sources can be synchronized with the source document.
- Users can work across an entire source set or a selected subset.

References:

- https://support.google.com/gemininotebook/answer/16179559
- https://support.google.com/gemininotebook/answer/16215270
- https://support.google.com/gemininotebook/answer/16269187

The implementation below is our approximation optimized for a small internal AIS knowledge corpus, cost control, Thai queries and inspectable evidence.

## 2. Processing model

```text
Google Drive knowledge tree
        |
        v
Format-aware extraction
        |
        v
Normalized structured text
(page / slide / sheet / heading boundaries preserved)
        |
        v
Source metadata + section-aware chunks
        |
        +--------------------+
        |                    |
        v                    v
Thai-friendly BM25      Multilingual semantic search
        |                    |
        +------ rank fusion --+
                   |
                   v
        Source-aware diversification
                   |
                   v
          Neighbor context expansion
                   |
                   v
       Question-intent context packing
       focused / overview / global overview
                   |
                   v
            Grounded LLM synthesis
                   |
                   v
      Claim -> retrieved chunk validation
                   |
                   v
     Backend-built citations from source text
                   |
                   v
              Chat response
```

## 3. Why this differs from simple top-k RAG

A simple RAG pipeline retrieves the top few chunks and asks the LLM to answer. That fails on:

- broad questions such as "Incentive คิดยังไง"
- comparisons across Install and MA
- rules where the exception is in the next section
- summaries spanning many documents
- short follow-ups that depend on conversation context

The current pipeline therefore:

1. detects broad/global-overview questions,
2. retrieves a wider candidate set,
3. distributes evidence across relevant sources instead of allowing one file to dominate,
4. adds adjacent chunks after strong evidence,
5. for whole-corpus summaries, includes representative context from each source when the corpus is small enough,
6. gives the LLM only retrieved source evidence as factual authority,
7. derives displayed citation text from the retrieved source itself rather than trusting model-authored quotes.

## 4. File ingestion

The knowledge root is no longer Markdown-only.

### Google Workspace

| Source | Ingestion |
|---|---|
| Google Docs | export as DOCX, preserve headings/tables where possible |
| Google Slides | export as PPTX, preserve slide boundaries |
| Google Sheets | export as XLSX, preserve sheet boundaries |

### Uploaded / Drive files

| Type | Current extraction |
|---|---|
| PDF | text extraction by page |
| DOCX | headings, paragraphs, tables |
| PPTX | slide boundaries and text |
| XLSX | sheet names and cell values/formulas |
| CSV / TSV | delimited text |
| TXT / Markdown | direct text |
| HTML | visible text + headings |
| JSON / JSONL | normalized text |
| YAML / XML / SQL / source code / logs / config | direct decoded text |

Text decoding includes UTF-8, UTF-16 and Thai CP874 fallback.

Markdown remains special only because its YAML frontmatter can carry authority/version metadata. It is **not** required as the content format.

## 5. Citation locators

Normalized structure is intentionally visible in chunk headings:

- PDF -> `Page N`
- PPTX -> `Slide N`
- XLSX -> `Sheet: <name>`
- DOCX -> document heading path
- Markdown -> Markdown heading path

This allows the UI to show meaningful evidence locations without depending on file type.

## 6. Grounding policy

For the configured Drive knowledge root:

- normal user flow searches all authoritative knowledge automatically,
- retrieved evidence may be sent to the configured external LLM according to the application's Drive-wide policy,
- conversation history helps resolve references but is not factual evidence,
- factual claims must point to retrieved chunk IDs,
- the backend constructs citation text directly from retrieved source chunks.

## 7. Current limits

This is text-bearing-file parity, not full multimodal NotebookLM parity.

Not yet handled by the text extractor:

- image-only files / screenshots requiring OCR or vision,
- scanned PDFs with no extractable text,
- audio/video transcription,
- websites and YouTube URLs as first-class sources,
- legacy binary formats such as old `.doc` where reliable pure-Python extraction is unavailable.

Those should be added as separate multimodal/transcription adapters rather than pushing binary content into the text index.

## 8. Acceptance criteria

A release should not be called NotebookLM-like solely because the UI resembles it.

Minimum behavioral gates:

- direct fact lookup returns the correct supporting passage,
- broad questions synthesize multiple relevant sources,
- exception/threshold questions retain exact operators and units,
- follow-up questions resolve the previous topic without treating previous AI output as evidence,
- whole-corpus summaries do not silently collapse to a single document,
- citations always point to retrieved source content,
- unsupported questions abstain instead of filling gaps from general model knowledge,
- common text-bearing Drive files ingest without conversion to Markdown by the user.
