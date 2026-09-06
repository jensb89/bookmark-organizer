#!/usr/bin/env python3
import csv, html, json, os, re, time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

PRIMARY_RESULTS = 'bookmark_check_results.json'
SMART_RESULTS = 'smart_check_results.json'
METADATA_CACHE = 'bookmark_metadata_cache.json'
CSV_OUTPUT = 'bookmark_clusters.csv'
PLAN_OUTPUT = 'cluster_plan.json'
HTML_OUTPUT = 'bookmarks_ai_organized.html'

EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
FETCH_PAGE_METADATA = True
REQUEST_TIMEOUT = 12
REQUEST_DELAY = 0.15
MAX_CONTENT_CHARS = 4000
TARGET_CLUSTER_COUNT = 50
LOW_CONFIDENCE_SIMILARITY = 0.50
MAX_CLUSTER_EXAMPLES = 12
MAX_FOLDER_DEPTH = 3
PUT_LOW_CONFIDENCE_IN_REVIEW = True

LLM_API_KEY = os.getenv('LLM_API_KEY', '')
LLM_BASE_URL = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5-mini')

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9,de;q=0.8',
})

def esc(v): return html.escape(str(v), quote=True)

def load_json(path, default):
    p = Path(path)
    if not p.exists(): return default
    with p.open('r', encoding='utf-8') as f: return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_url(url):
    try:
        p = urlparse(url.strip())
        scheme = p.scheme.lower()
        host = (p.hostname or '').lower()
        if host.startswith('www.'): host = host[4:]
        netloc = host
        if p.port and not ((scheme == 'http' and p.port == 80) or (scheme == 'https' and p.port == 443)):
            netloc += f':{p.port}'
        path = p.path or '/'
        if path != '/' and path.endswith('/'): path = path[:-1]
        tracking = ('utm_', 'fbclid', 'gclid', 'mc_cid', 'mc_eid')
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(tracking)]
        q.sort()
        return urlunparse((scheme, netloc, path, '', urlencode(q), ''))
    except Exception:
        return url.strip()

def load_alive_bookmarks():
    primary = load_json(PRIMARY_RESULTS, [])
    if not primary:
        raise RuntimeError(f'{PRIMARY_RESULTS} not found or empty.')

    items = [dict(x) for x in primary if x.get('classification') == 'alive']
    smart = load_json(SMART_RESULTS, [])
    items += [dict(x) for x in smart if x.get('classification') == 'alive']

    unique = {}
    dupes = 0
    for item in items:
        url = item.get('final_url') or item.get('url', '')
        key = normalize_url(url)
        if not key: continue
        if key in unique:
            dupes += 1
            cur = unique[key]
            for field in ('page_title','description','content_excerpt','final_url'):
                if not cur.get(field) and item.get(field): cur[field] = item[field]
            continue
        item['_normalized_url'] = key
        unique[key] = item

    out = list(unique.values())
    print(f'Loaded alive bookmarks: {len(items)}')
    print(f'Removed duplicates:      {dupes}')
    print(f'Unique bookmarks:        {len(out)}')
    return out

def extract_page_data(response):
    if 'html' not in response.headers.get('content-type','').lower():
        return {'page_title':'','description':'','content_excerpt':''}
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except Exception:
        return {'page_title':'','description':'','content_excerpt':''}
    title = soup.title.get_text(' ', strip=True) if soup.title else ''
    desc = soup.find('meta', attrs={'name': re.compile('^description$', re.I)})
    if not desc: desc = soup.find('meta', attrs={'property':'og:description'})
    description = desc.get('content','').strip() if desc else ''
    for tag in soup(['script','style','noscript','svg','nav','footer','header','form']): tag.decompose()
    text = re.sub(r'\s+', ' ', soup.get_text(' ', strip=True))
    return {'page_title':title[:500], 'description':description[:1000], 'content_excerpt':text[:MAX_CONTENT_CHARS]}

def enrich_bookmarks(bookmarks):
    if not FETCH_PAGE_METADATA: return bookmarks
    cache = load_json(METADATA_CACHE, {})
    print('\nFetching missing page metadata...')
    for i, item in enumerate(bookmarks, 1):
        key = item['_normalized_url']
        if key in cache:
            item.update(cache[key]); continue
        url = item.get('final_url') or item.get('url','')
        print(f'[{i:>3}/{len(bookmarks)}] {item.get("title","")[:55]}')
        data = {'page_title':item.get('page_title',''), 'description':item.get('description',''), 'content_excerpt':item.get('content_excerpt','')}
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if 200 <= r.status_code < 400:
                fetched = extract_page_data(r)
                for k,v in fetched.items():
                    if v: data[k] = v
                item['final_url'] = r.url
            r.close()
        except requests.RequestException:
            pass
        item.update(data); cache[key] = data
        if i % 10 == 0: save_json(METADATA_CACHE, cache)
        time.sleep(REQUEST_DELAY)
    save_json(METADATA_CACHE, cache)
    return bookmarks

def semantic_text(item):
    domain = urlparse(item.get('final_url') or item.get('url','')).hostname or ''
    parts = [
        f'Bookmark title: {item.get("title","")}',
        f'Web page title: {item.get("page_title","")}',
        f'Domain: {domain}',
        f'Description: {item.get("description","")}',
        f'Page content: {item.get("content_excerpt","")}',
    ]
    return '\n'.join(x for x in parts if x.split(':',1)[-1].strip())

def create_embeddings(bookmarks):
    print(f'\nLoading embedding model: {EMBEDDING_MODEL}')
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [semantic_text(x) for x in bookmarks]
    print(f'Embedding {len(texts)} bookmarks...')
    return np.asarray(model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True), dtype=np.float32)

def cluster_embeddings(embeddings):
    cluster_count = min(TARGET_CLUSTER_COUNT, len(embeddings))
    print(f'\nClustering into {cluster_count} initial semantic clusters...')
    c = AgglomerativeClustering(n_clusters=cluster_count, metric='cosine', linkage='average')
    return c.fit_predict(embeddings)

def compute_cluster_information(embeddings, labels):
    clusters = defaultdict(list)
    for i, label in enumerate(labels): clusters[int(label)].append(i)
    info = {}
    for cid, idxs in clusters.items():
        vecs = embeddings[idxs]
        centroid = vecs.mean(axis=0)
        n = np.linalg.norm(centroid)
        if n: centroid /= n
        sims = vecs @ centroid
        ranked = np.argsort(-sims)
        info[cid] = {
            'indexes': idxs,
            'similarities': {idxs[i]: float(sims[i]) for i in range(len(idxs))},
            'representative_indexes': [idxs[int(i)] for i in ranked[:MAX_CLUSTER_EXAMPLES]],
        }
    return info

def parse_json_from_llm(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.I)
    text = re.sub(r'\s*```$', '', text)
    a, b = text.find('{'), text.rfind('}')
    if a < 0 or b < 0: raise ValueError('LLM response contained no JSON object.')
    return json.loads(text[a:b+1])

def generate_taxonomy(bookmarks, cluster_info):
    summaries = []
    for cid, data in sorted(cluster_info.items()):
        examples = []
        for i in data['representative_indexes']:
            x = bookmarks[i]
            examples.append({
                'title': x.get('title',''),
                'page_title': x.get('page_title',''),
                'domain': urlparse(x.get('final_url') or x.get('url','')).hostname or '',
                'old_folder': x.get('folder',''),
                'description': x.get('description','')[:300],
            })
        summaries.append({'cluster_id':cid, 'size':len(data['indexes']), 'examples':examples})

    if not LLM_API_KEY:
        print('\nWARNING: LLM_API_KEY not set; using raw cluster names only.')
        return {str(cid): {'name':f'Cluster {cid}', 'path':[f'Cluster {cid}'], 'description':'LLM taxonomy not run.'} for cid in cluster_info}

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    prompt = f'''You are organizing a personal browser bookmark collection.
The bookmarks were grouped semantically using embeddings. Create a clean practical taxonomy.

Rules:
- Infer topics from examples.
- Reuse broad parent categories across related clusters.
- Do not blindly preserve old folders; they are context only.
- Avoid a separate top-level folder for every cluster.
- Avoid vague names unless unavoidable.
- Maximum folder depth: {MAX_FOLDER_DEPTH}.
- Paths go broad -> specific.
- Related clusters may map to the same path, but do not merge clusters merely to reduce folder count.
- Prefer a coherent taxonomy of roughly 12-25 final folder paths for this collection.
- If a cluster is mixed, choose the path that best fits its representative core; low-confidence outliers are handled separately.
- Return ONLY valid JSON.

Format:
{{"clusters":{{"0":{{"name":"...","path":["Top","Sub"],"description":"..."}}}}}}

CLUSTERS:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}'''
    print(f'\nSending {len(summaries)} clusters to {LLM_MODEL}...')
    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {'role':'system','content':'You are an information architect. Return only valid JSON.'},
            {'role':'user','content':prompt},
        ],
    )
    mappings = parse_json_from_llm(r.choices[0].message.content or '').get('clusters', {})
    for cid in cluster_info:
        if str(cid) not in mappings:
            mappings[str(cid)] = {'name':f'Cluster {cid}', 'path':['Needs Review',f'Cluster {cid}'], 'description':'No taxonomy returned.'}
    return mappings


def unique_taxonomy_paths(taxonomy):
    """Return de-duplicated taxonomy paths available for second-stage assignment."""
    seen = set()
    paths = []

    for entry in taxonomy.values():
        path = entry.get("path") or [entry.get("name", "Needs Review")]
        path = [str(x).strip() for x in path if str(x).strip()]

        if not path:
            continue

        key = tuple(path)

        if key not in seen:
            seen.add(key)
            paths.append(path)

    return paths



def classify_all_bookmarks(bookmarks, labels, cluster_info, taxonomy):
    """
    Final LLM classification stage.

    Clustering is used only to discover a practical taxonomy.
    Every bookmark is then classified independently into one of the
    existing taxonomy paths, or Needs Review.
    """
    assignments = {}

    allowed_paths = unique_taxonomy_paths(taxonomy)
    allowed_path_strings = [" / ".join(path) for path in allowed_paths]

    if not LLM_API_KEY:
        print(
            "\nWARNING: LLM_API_KEY is not set. "
            "Falling back to the cluster taxonomy paths."
        )

        for i, item in enumerate(bookmarks):
            cid = int(labels[i])
            path = taxonomy[str(cid)].get("path") or [
                taxonomy[str(cid)].get("name", f"Cluster {cid}")
            ]

            assignments[i] = {
                "path": path,
                "reassigned": False,
                "reason": "LLM final classification unavailable; kept cluster path",
            }

        return assignments

    client = OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    batch_size = 20

    print(
        f"\nFinal LLM classification for all {len(bookmarks)} bookmarks..."
    )

    for start in range(0, len(bookmarks), batch_size):
        batch_indexes = list(
            range(start, min(start + batch_size, len(bookmarks)))
        )

        batch_items = []

        for i in batch_indexes:
            item = bookmarks[i]
            cid = int(labels[i])

            batch_items.append({
                "id": i,
                "title": item.get("title", ""),
                "page_title": item.get("page_title", ""),
                "domain": urlparse(
                    item.get("final_url") or item.get("url", "")
                ).hostname or "",
                "description": item.get("description", "")[:500],
                # Keep content excerpt deliberately short; full page text can
                # contain navigation/boilerplate that harms classification.
                "content_excerpt": item.get("content_excerpt", "")[:500],
                "old_folder": item.get("folder", ""),
                "cluster_name": taxonomy[str(cid)].get(
                    "name",
                    f"Cluster {cid}"
                ),
                "cluster_path": " / ".join(
                    taxonomy[str(cid)].get("path")
                    or [taxonomy[str(cid)].get("name", f"Cluster {cid}")]
                ),
                "cluster_similarity": round(
                    cluster_info[cid]["similarities"][i],
                    4,
                ),
            })

        prompt = f"""
You are performing the FINAL classification of a personal browser bookmark collection.

A semantic clustering step was used only to discover a useful taxonomy.
Do NOT blindly trust the bookmark's current cluster or old folder.
Classify every bookmark by what the page itself is actually about.

Choose exactly one existing folder path from ALLOWED_PATHS for every bookmark.

Rules:
- Use title, page title, domain and description as the strongest evidence.
- content_excerpt is secondary evidence only.
- old_folder and cluster information are weak hints only.
- Do NOT invent a new folder path.
- Prefer the most specific appropriate existing path.
- If no existing path reasonably fits, return exactly ["Needs Review"].
- Return one result for every bookmark id.
- Return ONLY valid JSON.

Required format:
{{
  "assignments": {{
    "0": {{
      "path": ["Top", "Subcategory"],
      "reason": "short explanation"
    }}
  }}
}}

ALLOWED_PATHS:
{json.dumps(allowed_path_strings, ensure_ascii=False, indent=2)}

BOOKMARKS:
{json.dumps(batch_items, ensure_ascii=False, indent=2)}
""".strip()

        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful information architect and bookmark classifier. "
                        "Use only the supplied taxonomy and return valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        parsed = parse_json_from_llm(
            r.choices[0].message.content or ""
        )

        returned = parsed.get("assignments", {})
        allowed_tuples = {tuple(path) for path in allowed_paths}

        for i in batch_indexes:
            entry = returned.get(str(i), {})
            path = entry.get("path", ["Needs Review"])

            if isinstance(path, str):
                path = [
                    p.strip()
                    for p in path.split("/")
                    if p.strip()
                ]

            path = [
                str(p).strip()
                for p in path
                if str(p).strip()
            ]

            if tuple(path) not in allowed_tuples:
                path = ["Needs Review"]

            assignments[i] = {
                "path": path,
                "reassigned": True,
                "reason": entry.get(
                    "reason",
                    "Final LLM classification",
                ),
            }

        print(
            f"  Classified "
            f"{min(start + batch_size, len(bookmarks))}"
            f"/{len(bookmarks)}"
        )

    return assignments


def write_csv(bookmarks, labels, cluster_info, taxonomy, assignments):
    fields = [
        "cluster_id",
        "cluster_name",
        "similarity",
        "cluster_low_confidence",
        "cluster_folder",
        "final_folder",
        "classified_by_llm",
        "classification_reason",
        "title",
        "page_title",
        "old_folder",
        "url",
        "description",
    ]

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for i, item in enumerate(bookmarks):
            cid = int(labels[i])
            sim = cluster_info[cid]["similarities"][i]
            t = taxonomy[str(cid)]
            cluster_path = t.get("path") or [
                t.get("name", f"Cluster {cid}")
            ]
            final_assignment = assignments[i]

            w.writerow({
                "cluster_id": cid,
                "cluster_name": t.get("name", ""),
                "similarity": round(sim, 4),
                "cluster_low_confidence": sim < LOW_CONFIDENCE_SIMILARITY,
                "cluster_folder": " / ".join(cluster_path),
                "final_folder": " / ".join(final_assignment["path"]),
                "classified_by_llm": final_assignment["reassigned"],
                "classification_reason": final_assignment["reason"],
                "title": item.get("title", ""),
                "page_title": item.get("page_title", ""),
                "old_folder": item.get("folder", ""),
                "url": item.get("final_url") or item.get("url", ""),
                "description": item.get("description", ""),
            })

def build_tree(items):
    root = {'_bookmarks':[]}
    for x in items:
        node = root
        for folder in x['_ai_path']:
            node = node.setdefault(folder, {'_bookmarks':[]})
        node['_bookmarks'].append(x)
    return root

def bookmark_html(x, indent):
    attrs = [f'HREF="{esc(x.get("final_url") or x.get("url",""))}"']
    if x.get('add_date'): attrs.append(f'ADD_DATE="{esc(x["add_date"])}"')
    return '    '*indent + f'<DT><A {" ".join(attrs)}>{esc(x.get("title","Untitled"))}</A>'

def write_tree(tree, out, indent):
    for x in tree['_bookmarks']: out.append(bookmark_html(x, indent))
    for folder, sub in tree.items():
        if folder == '_bookmarks': continue
        out.append('    '*indent + f'<DT><H3>{esc(folder)}</H3>')
        out.append('    '*indent + '<DL><p>')
        write_tree(sub, out, indent+1)
        out.append('    '*indent + '</DL><p>')

def write_html(bookmarks, assignments):
    prepared = []

    for i, item in enumerate(bookmarks):
        y = dict(item)
        y["_ai_path"] = assignments[i]["path"]
        prepared.append(y)

    out = [
        "<!DOCTYPE NETSCAPE-Bookmark-file-1>",
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        "<TITLE>AI Organized Bookmarks</TITLE>",
        "<H1>AI Organized Bookmarks</H1>",
        "<DL><p>",
    ]

    write_tree(build_tree(prepared), out, 1)
    out.append("</DL><p>")

    Path(HTML_OUTPUT).write_text(
        "\n".join(out),
        encoding="utf-8",
    )

def main():
    bookmarks = enrich_bookmarks(load_alive_bookmarks())

    embeddings = create_embeddings(bookmarks)
    labels = cluster_embeddings(embeddings)
    info = compute_cluster_information(embeddings, labels)

    print(f"\nCreated {len(info)} initial semantic clusters.")

    for cid, data in sorted(
        info.items(),
        key=lambda p: -len(p[1]["indexes"])
    ):
        print(
            f"  Cluster {cid:>3}: "
            f"{len(data['indexes']):>3} bookmarks"
        )

    taxonomy = generate_taxonomy(bookmarks, info)

    assignments = classify_all_bookmarks(
        bookmarks,
        labels,
        info,
        taxonomy,
    )

    needs_review_count = sum(
        1
        for value in assignments.values()
        if value["path"] == ["Needs Review"]
    )

    final_paths = {
        tuple(value["path"])
        for value in assignments.values()
        if value["path"] != ["Needs Review"]
    }

    save_json(
        PLAN_OUTPUT,
        {
            "settings": {
                "embedding_model": EMBEDDING_MODEL,
                "target_cluster_count": TARGET_CLUSTER_COUNT,
                "llm_model": LLM_MODEL if LLM_API_KEY else None,
                "old_folder_in_embedding": False,
                "final_llm_classifies_all_bookmarks": True,
                "taxonomy_is_closed": True,
            },
            "bookmark_count": len(bookmarks),
            "initial_cluster_count": len(info),
            "final_folder_count": len(final_paths),
            "needs_review_count": needs_review_count,
            "clusters": taxonomy,
        },
    )

    write_csv(
        bookmarks,
        labels,
        info,
        taxonomy,
        assignments,
    )

    write_html(
        bookmarks,
        assignments,
    )

    print("\n================================")
    print("AI bookmark organizer V4 complete")
    print("================================")
    print(f"Bookmarks:          {len(bookmarks)}")
    print(f"Initial clusters:   {len(info)}")
    print(f"Final folder paths: {len(final_paths)}")
    print(f"Needs Review:       {needs_review_count}")
    print("\nCreated:")
    print(f"  {CSV_OUTPUT}")
    print(f"  {PLAN_OUTPUT}")
    print(f"  {HTML_OUTPUT}")
    print(f"  {METADATA_CACHE}")
    print(
        "\nInspect final_folder and reassignment_reason "
        "in bookmark_clusters.csv before importing the HTML."
    )


if __name__ == '__main__':
    main()
