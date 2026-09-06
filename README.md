# AI Bookmark Organizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)](https://www.apache.org/licenses/LICENSE-2.0)
[![Embeddings: Local](https://img.shields.io/badge/Embeddings-Local-FFD21E?logo=huggingface&logoColor=black)](https://www.sbert.net/)
![Workflow: Non-destructive](https://img.shields.io/badge/Workflow-Non--destructive-2EA44F)

Turn a messy browser bookmark export into a clean, reviewable folder structure.

This Python pipeline archives older bookmarks, checks links conservatively, groups surviving pages with local embeddings, and uses an OpenAI-compatible LLM to build a compact taxonomy. It never edits your browser directly: every stage produces HTML, JSON, or CSV files you can inspect first.

## What it does

- Preserves an untouched browser export as your backup
- Separates legacy bookmarks from the collection you want to clean
- Marks links as `alive`, `dead`, `uncertain`, or `special`
- Retries uncertain links with safer URL variants and page checks
- Deduplicates and clusters usable bookmarks with local embeddings
- Uses an LLM only for taxonomy design and final classification
- Exports Netscape bookmark HTML that Chrome, Firefox, and other browsers can import

## How it works

```text
bookmarks.html
    │
    ▼
sort_bookmarks.py
    ├── console: counts by year
    └── HTML: bookmarks_by_year.html
                 │
                 ▼
        choose the archive range
                 │
                 ▼
        split_bookmarks.py
          ├── legacy archive
          └── bookmarks to analyze
                     │
                     ▼
             check_bookmarks.py
               ├── alive / dead / special
               └── uncertain
                         │
                         ▼
            check_bookmarks_uncertain.py
                         │
                         ▼
             cluster_organize_bookmarks.py
               local embeddings + LLM
                         │
                         ▼
             bookmarks_ai_organized.html
```

The organizer first discovers semantic clusters, asks the LLM to turn them into a small reusable folder hierarchy, and then classifies every bookmark into that closed taxonomy. Ambiguous results can be sent to `Needs Review`.

## Requirements

- Python 3.10+
- An OpenAI-compatible API for meaningful folder names and final classification
- Internet access while checking links and fetching page metadata

Install the dependencies:

```bash
python3 -m pip install \
  requests \
  beautifulsoup4 \
  numpy \
  scikit-learn \
  sentence-transformers \
  openai
```

## Quick start

1. Export your browser bookmarks as `bookmarks.html` and place the file in the project directory.

2. Generate a year-by-year overview:

```bash
python3 sort_bookmarks.py
```

The script prints the number of bookmarks for every year and the total to the console. It also creates `bookmarks_by_year.html`, which preserves the original folders inside top-level year folders for easy visual review.

3. Use that overview to choose a sensible archive window, then update `ARCHIVE_START` and `ARCHIVE_END` near the top of `split_bookmarks.py`. The current defaults archive bookmarks from 2012 through 2014; bookmarks outside that range continue through the pipeline.

4. Configure an OpenAI-compatible model:

```bash
export LLM_API_KEY="your-key"
export LLM_MODEL="gpt-5-mini"

# Optional for a compatible proxy or local server:
export LLM_BASE_URL="https://api.openai.com/v1"
```

5. Run the remaining pipeline in order:

```bash
python3 split_bookmarks.py
python3 check_bookmarks.py
python3 split_checked_bookmarks.py
python3 check_bookmarks_uncertain.py
python3 cluster_organize_bookmarks.py
```

6. Review `bookmark_clusters.csv`, especially `final_folder` and `reassignment_reason`, then import `bookmarks_ai_organized.html` into your browser.

> Keep your original export. The generated file is a proposed organization, not a replacement you should trust without review.

## Scripts

| Script | Purpose | Main output |
| --- | --- | --- |
| `sort_bookmarks.py` | Prints year-by-year counts and creates a visual overview grouped by year | `bookmarks_by_year.html` |
| `split_bookmarks.py` | Separates the configured archive years from bookmarks to analyze | `bookmarks_archive_2012_2014.html`, `bookmarks_to_analyze_2015_2026.html` |
| `check_bookmarks.py` | First-pass link and redirect check | `bookmark_check_results.csv`, `bookmark_check_results.json` |
| `split_checked_bookmarks.py` | Creates reviewable HTML files for each link status | `bookmarks_alive.html`, `bookmarks_dead.html`, `bookmarks_uncertain.html`, `bookmarks_special.html` |
| `check_bookmarks_uncertain.py` | Re-checks uncertain links more carefully | `smart_check_results.json`, retry-result HTML files |
| `cluster_organize_bookmarks.py` | Enriches, deduplicates, clusters, creates a taxonomy, and classifies | `bookmark_clusters.csv`, `cluster_plan.json`, `bookmarks_ai_organized.html` |

File names and tuning values currently live as constants near the top of each script. Change them there if your export or archive window differs.

## Key outputs

- `bookmark_clusters.csv` — the best place to audit every proposed move
- `cluster_plan.json` — run settings, generated taxonomy, and summary counts
- `bookmarks_ai_organized.html` — the final browser-importable bookmark file
- `bookmark_metadata_cache.json` — cached page metadata, so repeated runs avoid unnecessary downloads

Generated `.html`, `.json`, and `.csv` files are ignored by Git because bookmark exports may contain private browsing data.

## Model options

The code uses the OpenAI Python client with a configurable base URL, so it also works with compatible services.

OpenAI:

```bash
export LLM_API_KEY="..."
export LLM_MODEL="gpt-5-mini"
```

Ollama:

```bash
export LLM_API_KEY="ollama"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="qwen3:8b"
```

LiteLLM:

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="http://localhost:4000/v1"
export LLM_MODEL="your-model-name"
```

Without `LLM_API_KEY`, the organizer still creates output, but it falls back to generic cluster folders instead of an AI-generated taxonomy.

## Safety

The workflow is deliberately non-destructive:

```text
export → analyze → review → generate → manually import
```

It does not modify the browser bookmark database, delete remote content, or overwrite the original export.

## License

Apache-2.0
