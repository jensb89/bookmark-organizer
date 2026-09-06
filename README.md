# AI Bookmark Organizer

A small Python pipeline for cleaning up, validating, clustering, and reorganizing a large browser bookmark collection with a mix of deterministic tooling, local embeddings, and LLM-based classification.

The project was built to solve a very common real-world problem:

- years of accumulated browser bookmarks
- old folders that no longer reflect how the bookmarks should be organized
- many dead or redirected links
- duplicate or obsolete entries
- a large legacy archive that is not worth fully reclassifying
- a smaller modern set that *is* worth cleaning up properly

The final architecture deliberately avoids a full agent framework such as LangGraph. Instead, it uses a deterministic multi-stage pipeline where each step has one clear responsibility and produces inspectable intermediate files.

---

## Overview

The workflow is split into four major phases:

```text
Original browser bookmarks
        ↓
1. Split legacy archive vs. modern bookmarks
        ↓
2. Check links and isolate dead / uncertain / special URLs
        ↓
3. Re-check uncertain links more intelligently
        ↓
4. Cluster + create taxonomy + classify all surviving bookmarks
        ↓
Clean, organized bookmarks.html
```

The overall goal is **not** to let an autonomous agent freely modify bookmarks.

Instead, every step is reproducible, inspectable, and reversible.

---

# Why this architecture?

A full LangGraph or multi-agent workflow would be possible, but it would add complexity without improving the result much for this use case.

The bookmark cleanup problem is mostly a sequence of predictable operations:

1. parse bookmark HTML
2. extract dates and folders
3. validate URLs
4. fetch metadata
5. create embeddings
6. cluster bookmarks
7. create a compact taxonomy
8. classify bookmarks into that taxonomy
9. export a new bookmark file

These are deterministic stages with very clear inputs and outputs.

Using LangGraph here would mainly introduce:

- additional state management
- graph orchestration
- more dependencies
- more debugging complexity
- more difficult reproducibility
- little benefit from autonomous planning

The only steps that really benefit from an LLM are:

- **naming and merging semantic clusters**
- **classifying bookmarks into the final taxonomy**

Everything else is better handled by normal Python.

So the architecture is intentionally:

```text
Deterministic Python
        +
Local embeddings
        +
LLM only where semantic reasoning is useful
```

This makes the workflow easier to understand, cheaper to run, easier to debug, and much safer for a personal bookmark collection.

---

# Project Structure

The project currently contains several scripts that were created step by step while refining the workflow.

The important scripts are described below.

---

## 1. `sort_bookmarks.py` : Sort bookmarks by year

### Purpose

Reads an exported browser `bookmarks.html` file and groups bookmarks by the year in which they were added.

The browser bookmark export contains timestamps such as:

```html
ADD_DATE="1415987378"
```

The script converts these Unix timestamps into years.

### Example

Original:

```text
Bookmarks
├── AI
│   ├── LangGraph
│   └── OpenAI
├── Hardware
│   └── ESP32
└── Travel
```

Becomes:

```text
2026
├── AI
│   └── OpenAI
└── Hardware
    └── ESP32

2025
└── AI
    └── LangGraph

2023
└── Travel
```

The original folder hierarchy is preserved **inside each year**.

### Why this was useful

The year distribution immediately showed that most bookmarks were very old.

In this collection, the result looked roughly like:

```text
2012–2014  → ~4,000 legacy bookmarks
2015+      → ~500 bookmarks worth analyzing
```

That made it clear that the old bookmarks should be archived instead of wasting AI/API effort on them.

---

## 2. `split_bookmarks.py`: Archive / modern split script

This script splits the original export into two files:

```text
bookmarks_archive_2012_2014.html
bookmarks_to_analyze_2015_2026.html
```

### Purpose

The legacy archive is kept exactly as it was, including the old folder structure.

Only the more recent bookmarks continue through the cleanup pipeline.

### Why this matters

AI classification is most useful for bookmarks that are still relevant.

There is little value in carefully reorganizing thousands of links from 10+ years ago if many of them are obsolete anyway.

---

# Link Checking

## 3. `check_bookmarks.py`

### Purpose

Performs the first-pass URL health check.

It reads:

```text
bookmarks_to_analyze_2015_2026.html
```

and writes:

```text
bookmark_check_results.csv
bookmark_check_results.json
```

Each bookmark receives information such as:

```text
title
folder
url
final_url
http_status
classification
redirected
redirect_count
add_date
error
```

### Classifications

Bookmarks are classified into:

```text
alive
dead
uncertain
special
```

### `alive`

Usually means:

```text
HTTP 2xx
HTTP 3xx after redirects
```

### `dead`

Used conservatively for clear failures such as:

```text
404
410
DNS resolution failure
```

### `uncertain`

Includes cases where the website may still work but automated requests are unreliable:

```text
403 Forbidden
429 Too Many Requests
timeouts
SSL problems
server errors
Cloudflare / bot protection
```

### `special`

Browser-specific or non-HTTP URLs such as:

```text
javascript:
chrome-extension://
file://
data:
```

This is important because old bookmark collections often contain bookmarklets and browser-extension links.

---

## 4. `split_checked_bookmarks.py`

### Purpose

Takes:

```text
bookmark_check_results.json
```

and creates separate bookmark files:

```text
bookmarks_alive.html
bookmarks_dead.html
bookmarks_uncertain.html
bookmarks_special.html
```

The original folder structure is preserved.

### Why this exists

It makes manual review easy.

Instead of looking through a CSV, the result can simply be imported into a browser and inspected visually.

---

# Smarter second-pass link checking

## 5. `check_bookmarks_uncertain.py`

### Purpose

Re-checks only the bookmarks that were classified as `uncertain`.

It performs a more tolerant second pass.

### Additional strategies

The script may try:

- original URL
- HTTPS version of an old HTTP URL
- `www` / non-`www` variants
- longer timeouts
- normal browser-like GET requests
- redirects
- real HTML content detection

### Important design choice

The script **does not let AI decide whether a link is dead**.

This is intentional.

An LLM can understand what a page is about, but it should not hallucinate technical reachability.

The URL checker remains deterministic.

### Outputs

```text
smart_check_results.json
recovered_alive.html
dead_after_retry.html
still_uncertain.html
```

The JSON also stores useful metadata when possible:

```text
page_title
description
final_url
```

This metadata can later be reused by the AI organizer.

---

# AI Bookmark Organization

The organizer went through several iterations while the clustering strategy was refined.

The latest version is:

```text
cluster_organize_bookmarks_v5_1.py
```

Earlier versions are useful for understanding the evolution, but the latest script is the recommended one.

---

# Final Architecture

The final organizer uses the following pipeline:

```text
Alive bookmarks
        ↓
Fetch / reuse metadata
        ↓
Local sentence embeddings
        ↓
Semantic clustering
        ↓
LLM labels clusters
        ↓
LLM compacts clusters into a small taxonomy
        ↓
LLM classifies EVERY bookmark into that taxonomy
        ↓
Generate final bookmarks HTML
```

---

# Why embeddings first?

The organizer does **not** initially ask the LLM:

> "Which folder should bookmark #137 belong to?"

Doing this independently for every bookmark often produces inconsistent categories.

Instead, embeddings first reveal the natural structure of the collection.

For example:

```text
LangGraph
OpenAI Agents SDK
Google ADK
CrewAI
AutoGen
```

may naturally end up near one another in embedding space.

The LLM can then identify the cluster as something like:

```text
AI & Machine Learning / Agent Frameworks
```

This produces a much more consistent taxonomy.

---

# Local Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

via `sentence-transformers`.

Embeddings are computed locally.

No API request is required for this step.

The embedding text is built mainly from:

```text
bookmark title
page title
domain
description
page content excerpt
```

The old folder structure is intentionally **not included in the embedding representation**.

This avoids biasing the new organization toward the messy legacy structure.

---

# Hugging Face warning

On the first run you may see:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

This is harmless for this project.

The sentence-transformer model is downloaded from Hugging Face the first time it is used.

You **do not need a Hugging Face token** for normal use.

The warning only means that anonymous downloads have lower rate limits.

After the model has been downloaded, it is normally cached locally and future runs should not need to download it again.

If desired, a Hugging Face token can be configured to remove the warning and increase download limits, but it is optional.

---

# Semantic Clustering

The current organizer starts with approximately:

```python
TARGET_CLUSTER_COUNT = 50
```

for a collection of roughly 300–500 bookmarks.

These clusters are deliberately finer-grained than the final folder structure.

Clustering is used only to **discover topics**.

It is *not* trusted as the final classifier.

This distinction became important during development.

Earlier versions attempted to assign bookmarks directly from their nearest semantic cluster, but some unrelated bookmarks still had surprisingly high similarity scores.

Therefore the final architecture treats clusters as a discovery tool only.

---

# Two-stage taxonomy generation

The latest version creates the taxonomy in two LLM stages.

## Stage 1 — label semantic clusters

The LLM receives representative bookmarks from each cluster and creates short cluster descriptions.

Example:

```text
Cluster 12
- ESP32 display wiring
- SPI TFT tutorial
- QMI8658 sensor documentation
- EasyEDA PCB article

→ Embedded Electronics
```

---

## Stage 2 — compact the taxonomy

The cluster labels are then merged into a deliberately small final folder hierarchy.

The script currently targets roughly:

```python
MAX_FINAL_PATHS = 22
MAX_TOP_LEVEL_FOLDERS = 14
```

This prevents taxonomy explosion.

For example, instead of ending up with:

```text
Shopping
Shopping & Services
Shopping & Consumer Electronics
Consumer Products
Online Purchases
```

the model is encouraged to reuse:

```text
Shopping
```

with a few meaningful subcategories.

The same applies to media, development, finance, AI, hardware, and similar categories.

---

# Final bookmark classification

After the taxonomy exists, **every bookmark is classified again by the LLM**.

This was an important improvement over earlier versions.

The classifier primarily uses:

```text
title
page_title
domain
description
```

Secondary hints include:

```text
short page content excerpt
old folder
semantic cluster
cluster similarity
```

The old folder and cluster are intentionally weak hints.

The LLM must choose one of the already-created taxonomy paths.

It is not allowed to invent arbitrary new folders.

If nothing fits, it may return:

```text
Needs Review
```

This keeps the taxonomy stable.

---

# Why classify all bookmarks again?

Semantic clustering is excellent for:

```text
discovering themes
finding similar groups
building the taxonomy
```

But it is less reliable for:

```text
final per-bookmark assignment
```

For example, a bookmark can sometimes have a high embedding similarity to a cluster because of overlapping page text, navigation, boilerplate, or terminology even though the bookmark itself belongs elsewhere.

The final LLM pass solves this by evaluating the actual bookmark context against the **fixed taxonomy**.

This combination worked better than either embeddings alone or LLM-only classification.

---

# Main outputs

The final organizer writes:

```text
bookmark_metadata_cache.json
bookmark_clusters.csv
cluster_plan.json
bookmarks_ai_organized.html
```

---

## `bookmark_metadata_cache.json`

Stores downloaded page metadata.

This avoids fetching all pages again on every run.

---

## `bookmark_clusters.csv`

The most useful inspection file.

It contains information such as:

```text
cluster_id
cluster_name
similarity
cluster_folder
final_folder
classified_by_llm
classification_reason
title
page_title
old_folder
url
description
```

This makes it easy to inspect the result before importing anything.

---

## `cluster_plan.json`

Contains configuration and taxonomy information.

Useful for debugging and understanding how the final folder structure was created.

---

## `bookmarks_ai_organized.html`

The final browser-importable bookmark file.

This should only be imported **after reviewing the CSV**.

The original bookmark export should always be kept as a backup.

---

# Recommended workflow

A typical full run looks like this:

```text
bookmarks.html
        ↓
sort / archive split
        ↓
bookmarks_to_analyze_2015_2026.html
        ↓
check_bookmarks.py
        ↓
bookmark_check_results.json
        ↓
split_checked_bookmarks.py
        ↓
bookmarks_uncertain.html
        ↓
smart_check_bookmarks.py
        ↓
smart_check_results.json
        ↓
cluster_organize_bookmarks_v5_1.py
        ↓
bookmark_clusters.csv
bookmarks_ai_organized.html
```

---

# Installation

Recommended Python version:

```text
Python 3.10+
```

Install the dependencies:

```bash
pip install \
    requests \
    beautifulsoup4 \
    numpy \
    scikit-learn \
    sentence-transformers \
    openai
```

---

# LLM configuration

The organizer uses an OpenAI-compatible API.

Environment variables:

```bash
export LLM_API_KEY="your-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-5-mini"
```

`LLM_BASE_URL` is optional for OpenAI.

---

## OpenAI

Example:

```bash
export LLM_API_KEY="..."
export LLM_MODEL="gpt-5-mini"
```

---

## Ollama

Because the code uses an OpenAI-compatible client, local models can also be used:

```bash
export LLM_API_KEY="ollama"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="qwen3:8b"
```

---

## LiteLLM

A LiteLLM proxy can also be used:

```bash
export LLM_API_KEY="..."
export LLM_BASE_URL="http://localhost:4000/v1"
export LLM_MODEL="your-model-name"
```

This makes the organizer model-agnostic.

---

# Safety and reversibility

The scripts are intentionally non-destructive.

They do **not** directly modify the browser bookmark database.

Instead, the workflow is:

```text
export
→ analyze
→ inspect
→ create new HTML
→ manually import
```

The original bookmark export should always be kept.

This also makes experimentation easy:

```text
change cluster count
change taxonomy constraints
switch LLM
rerun
compare CSV
```

without risking the original bookmarks.

---

# Design principles

The project follows a few simple principles.

### Deterministic where possible

Use normal Python for:

- parsing
- timestamp handling
- URL checking
- redirects
- deduplication
- metadata extraction
- file generation

### AI only where useful

Use AI for:

- semantic cluster interpretation
- taxonomy design
- final category selection

### Local processing where practical

Embeddings are created locally.

### Inspectable intermediate results

Every important step writes JSON, CSV, or HTML.

### Non-destructive

Never directly modify the user's live browser bookmarks.

### Compact taxonomy over perfect ontology

The goal is a folder structure that is useful to browse, not a theoretically perfect classification system.

---

# Possible future improvements

Some possible extensions:

- detect duplicate bookmarks more aggressively
- automatically remove tracking parameters from URLs
- optionally archive page content locally
- use a browser extension for new bookmarks
- automatically classify newly added bookmarks into the existing taxonomy
- semantic bookmark search
- vector database integration
- local-only LLM mode
- Web UI for reviewing proposed moves
- optional Chrome / Firefox bookmark synchronization
- periodic dead-link checks

A future version could also use LangGraph if the project evolves into a long-running interactive system with persistent state, user approvals, repeated background actions, or multiple tools.

For the current batch-cleanup workflow, however, a deterministic pipeline remains simpler and easier to maintain.

---

# Final result

For the collection this project was built around, the pipeline reduced a messy multi-year bookmark collection into:

```text
~4,000 old bookmarks
→ preserved as legacy archive

~500 newer bookmarks
→ link checked

~320 usable bookmarks
→ semantically analyzed

50 initial semantic clusters
→ compacted by AI

~18 final folder paths
→ all bookmarks classified

only a few
→ Needs Review
```

The result is a manageable bookmark hierarchy while still preserving the original collection and making every transformation inspectable.

---

## License

MIT
Apache-2.0

