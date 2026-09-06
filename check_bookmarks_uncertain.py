import json
import re
import time
import html
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


INPUT_FILE = "bookmarks_uncertain.html"

JSON_OUTPUT = "smart_check_results.json"

ALIVE_OUTPUT = "recovered_alive.html"
DEAD_OUTPUT = "dead_after_retry.html"
UNCERTAIN_OUTPUT = "still_uncertain.html"

TIMEOUT = 20
DELAY = 0.3


# ---------------------------------------------------------
# Session
# ---------------------------------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Cache-Control": "no-cache",
})

retry = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

session.mount(
    "http://",
    HTTPAdapter(max_retries=retry)
)

session.mount(
    "https://",
    HTTPAdapter(max_retries=retry)
)


# ---------------------------------------------------------
# Parse Netscape bookmark file
# ---------------------------------------------------------

def parse_bookmarks(filename):

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    bookmarks = []

    folder_stack = []
    pending_folder = None

    h3_pattern = re.compile(
        r"<H3\b[^>]*>(.*?)</H3>",
        re.IGNORECASE
    )

    bookmark_pattern = re.compile(
        r"<A\b[^>]*>.*?</A>",
        re.IGNORECASE
    )

    for line in lines:

        folder_match = h3_pattern.search(line)

        if folder_match:
            fragment = BeautifulSoup(
                folder_match.group(1),
                "html.parser"
            )

            pending_folder = fragment.get_text(strip=True)
            continue

        if re.search(r"<DL><p>", line, re.IGNORECASE):
            if pending_folder is not None:
                folder_stack.append(pending_folder)
                pending_folder = None
            continue

        bookmark_match = bookmark_pattern.search(line)

        if bookmark_match:
            fragment = BeautifulSoup(
                bookmark_match.group(0),
                "html.parser"
            )

            link = fragment.find("a")

            if link:
                bookmarks.append({
                    "title": link.get_text(strip=True),
                    "url": link.get("href", ""),
                    "folder": " / ".join(folder_stack),
                    "add_date": link.get("add_date", "")
                })

            continue

        if re.search(r"</DL><p>", line, re.IGNORECASE):
            if folder_stack:
                folder_stack.pop()

    return bookmarks


# ---------------------------------------------------------
# URL variants
# ---------------------------------------------------------

def generate_url_variants(url):
    """
    Try sensible URL variants:
    original,
    HTTPS if HTTP,
    www/non-www alternative.
    """

    variants = []

    def add(candidate):
        if candidate and candidate not in variants:
            variants.append(candidate)

    add(url)

    try:
        parsed = urlparse(url)

        if parsed.scheme == "http":
            add(
                urlunparse(
                    parsed._replace(scheme="https")
                )
            )

        host = parsed.hostname

        if host:
            if host.startswith("www."):
                new_host = host[4:]
            else:
                new_host = "www." + host

            netloc = new_host

            if parsed.port:
                netloc += f":{parsed.port}"

            add(
                urlunparse(
                    parsed._replace(netloc=netloc)
                )
            )

            if parsed.scheme == "http":
                add(
                    urlunparse(
                        parsed._replace(
                            scheme="https",
                            netloc=netloc
                        )
                    )
                )

    except Exception:
        pass

    return variants


# ---------------------------------------------------------
# Basic content quality check
# ---------------------------------------------------------

def looks_like_real_page(response):

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if "text/html" not in content_type:
        return False

    try:
        text = response.text[:50000]
    except Exception:
        return False

    if len(text.strip()) < 100:
        return False

    lower = text.lower()

    # obvious browser/security/interstitial pages
    suspicious_markers = [
        "checking your browser",
        "just a moment...",
        "enable javascript and cookies",
        "access denied",
        "request blocked",
        "temporarily unavailable",
    ]

    if any(marker in lower for marker in suspicious_markers):
        return False

    return True


# ---------------------------------------------------------
# Classify response
# ---------------------------------------------------------

def classify_response(response):

    code = response.status_code

    # Working page
    if 200 <= code < 300:
        if looks_like_real_page(response):
            return "alive"

        return "uncertain"

    # Redirect ending normally is handled automatically
    if 300 <= code < 400:
        return "alive"

    # Very likely genuinely dead
    if code in (404, 410):
        return "dead"

    # Site probably exists but blocks us
    if code in (
        401,
        403,
        405,
        407,
        409,
        423,
        425,
        429
    ):
        return "uncertain"

    # Legal restriction / geo blocking
    if code == 451:
        return "uncertain"

    if 400 <= code < 500:
        return "uncertain"

    if code >= 500:
        return "uncertain"

    return "uncertain"


# ---------------------------------------------------------
# Check one URL
# ---------------------------------------------------------

def smart_check(bookmark):

    original_url = bookmark["url"]

    result = {
        **bookmark,
        "classification": "uncertain",
        "tested_urls": [],
        "final_url": "",
        "http_status": "",
        "page_title": "",
        "description": "",
        "error": ""
    }

    if not original_url.startswith(
        ("http://", "https://")
    ):
        result["classification"] = "uncertain"
        result["error"] = "Unsupported URL scheme"
        return result

    last_error = ""

    for candidate in generate_url_variants(
        original_url
    ):

        attempt = {
            "url": candidate,
            "status": None,
            "final_url": "",
            "error": ""
        }

        try:
            response = session.get(
                candidate,
                timeout=TIMEOUT,
                allow_redirects=True
            )

            attempt["status"] = response.status_code
            attempt["final_url"] = response.url

            result["tested_urls"].append(attempt)

            classification = classify_response(
                response
            )

            if classification == "alive":

                result["classification"] = "alive"
                result["final_url"] = response.url
                result["http_status"] = response.status_code

                try:
                    soup = BeautifulSoup(
                        response.text,
                        "html.parser"
                    )

                    if soup.title:
                        result["page_title"] = (
                            soup.title.get_text(
                                strip=True
                            )
                        )

                    desc = soup.find(
                        "meta",
                        attrs={"name": "description"}
                    )

                    if not desc:
                        desc = soup.find(
                            "meta",
                            attrs={
                                "property":
                                "og:description"
                            }
                        )

                    if desc and desc.get("content"):
                        result["description"] = (
                            desc["content"].strip()
                        )

                except Exception:
                    pass

                response.close()
                return result

            if classification == "dead":

                # Important:
                # do NOT stop immediately.
                # Another sensible URL variant may work.
                last_error = (
                    f"HTTP {response.status_code}"
                )

            else:
                last_error = (
                    f"HTTP {response.status_code}"
                )

            response.close()

        except requests.exceptions.SSLError as e:
            attempt["error"] = "SSL error"
            result["tested_urls"].append(attempt)
            last_error = f"SSL error: {e}"

        except requests.exceptions.ConnectTimeout:
            attempt["error"] = "Connection timeout"
            result["tested_urls"].append(attempt)
            last_error = "Connection timeout"

        except requests.exceptions.ReadTimeout:
            attempt["error"] = "Read timeout"
            result["tested_urls"].append(attempt)
            last_error = "Read timeout"

        except requests.exceptions.TooManyRedirects:
            attempt["error"] = "Too many redirects"
            result["tested_urls"].append(attempt)
            last_error = "Too many redirects"

        except requests.exceptions.ConnectionError as e:
            error_text = str(e)

            attempt["error"] = error_text
            result["tested_urls"].append(attempt)

            last_error = error_text

        except requests.RequestException as e:
            attempt["error"] = str(e)
            result["tested_urls"].append(attempt)
            last_error = str(e)

    # -----------------------------------------------------
    # Final classification after all variants failed
    # -----------------------------------------------------

    statuses = [
        attempt["status"]
        for attempt in result["tested_urls"]
        if attempt["status"] is not None
    ]

    if statuses:
        # Only classify as dead if every actual HTTP
        # response was clearly 404/410.
        if all(
            status in (404, 410)
            for status in statuses
        ):
            result["classification"] = "dead"

        else:
            result["classification"] = "uncertain"

    else:
        # DNS failures across all variants are reasonably
        # strong evidence the site is gone.
        errors = " ".join(
            attempt["error"]
            for attempt in result["tested_urls"]
        ).lower()

        dns_markers = [
            "failed to resolve",
            "nameresolutionerror",
            "nodename nor servname",
            "name or service not known"
        ]

        if any(
            marker in errors
            for marker in dns_markers
        ):
            result["classification"] = "dead"
        else:
            result["classification"] = "uncertain"

    result["error"] = last_error

    return result


# ---------------------------------------------------------
# Bookmark HTML writer
# ---------------------------------------------------------

def escape(value):
    return html.escape(
        str(value),
        quote=True
    )


def build_tree(bookmarks):

    tree = {
        "_bookmarks": []
    }

    for bookmark in bookmarks:

        folder = bookmark.get(
            "folder",
            ""
        )

        path = (
            tuple(
                part.strip()
                for part in folder.split(" / ")
                if part.strip()
            )
            if folder
            else ()
        )

        node = tree

        for part in path:

            if part not in node:
                node[part] = {
                    "_bookmarks": []
                }

            node = node[part]

        node["_bookmarks"].append(
            bookmark
        )

    return tree


def bookmark_to_html(bookmark, indent):

    url = (
        bookmark.get("final_url")
        or bookmark.get("url", "")
    )

    title = bookmark.get(
        "title",
        "Untitled"
    )

    add_date = bookmark.get(
        "add_date",
        ""
    )

    attrs = [
        f'HREF="{escape(url)}"'
    ]

    if add_date:
        attrs.append(
            f'ADD_DATE="{escape(add_date)}"'
        )

    return (
        "    " * indent
        + f"<DT><A {' '.join(attrs)}>"
        + escape(title)
        + "</A>"
    )


def write_tree(tree, output, indent):

    for bookmark in tree["_bookmarks"]:
        output.append(
            bookmark_to_html(
                bookmark,
                indent
            )
        )

    for folder_name, subtree in tree.items():

        if folder_name == "_bookmarks":
            continue

        output.append(
            "    " * indent
            + f"<DT><H3>{escape(folder_name)}</H3>"
        )

        output.append(
            "    " * indent
            + "<DL><p>"
        )

        write_tree(
            subtree,
            output,
            indent + 1
        )

        output.append(
            "    " * indent
            + "</DL><p>"
        )


def write_bookmark_file(
    filename,
    title,
    bookmarks
):

    output = [
        '<!DOCTYPE NETSCAPE-Bookmark-file-1>',
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        f"<TITLE>{escape(title)}</TITLE>",
        f"<H1>{escape(title)}</H1>",
        "<DL><p>"
    ]

    tree = build_tree(
        bookmarks
    )

    write_tree(
        tree,
        output,
        1
    )

    output.append(
        "</DL><p>"
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(output)
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

bookmarks = parse_bookmarks(
    INPUT_FILE
)

print(
    f"\nFound {len(bookmarks)} uncertain bookmarks.\n"
)


results = []

for i, bookmark in enumerate(
    bookmarks,
    start=1
):

    print(
        f"[{i:>3}/{len(bookmarks)}] "
        f"{bookmark['title'][:60]}"
    )

    result = smart_check(
        bookmark
    )

    results.append(
        result
    )

    print(
        f"     -> "
        f"{result['classification'].upper()}"
    )

    if result.get("final_url"):
        print(
            f"        {result['final_url'][:100]}"
        )

    time.sleep(DELAY)


# ---------------------------------------------------------
# Save JSON
# ---------------------------------------------------------

with open(
    JSON_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# ---------------------------------------------------------
# Split result groups
# ---------------------------------------------------------

alive = [
    r for r in results
    if r["classification"] == "alive"
]

dead = [
    r for r in results
    if r["classification"] == "dead"
]

uncertain = [
    r for r in results
    if r["classification"] == "uncertain"
]


write_bookmark_file(
    ALIVE_OUTPUT,
    "Recovered Alive Bookmarks",
    alive
)

write_bookmark_file(
    DEAD_OUTPUT,
    "Dead After Retry",
    dead
)

write_bookmark_file(
    UNCERTAIN_OUTPUT,
    "Still Uncertain",
    uncertain
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n================================")
print("Smart bookmark check complete")
print("================================")

print(
    f"Recovered alive: {len(alive)}"
)

print(
    f"Dead:            {len(dead)}"
)

print(
    f"Still uncertain: {len(uncertain)}"
)

print(
    f"\nTotal:           {len(results)}"
)

print("\nCreated:")
print(f"  {JSON_OUTPUT}")
print(f"  {ALIVE_OUTPUT}")
print(f"  {DEAD_OUTPUT}")
print(f"  {UNCERTAIN_OUTPUT}")