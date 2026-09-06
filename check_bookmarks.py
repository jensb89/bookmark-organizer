import csv
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


INPUT_FILE = "bookmarks_to_analyze_2015_2026.html"

CSV_OUTPUT = "bookmark_check_results.csv"
JSON_OUTPUT = "bookmark_check_results.json"

REQUEST_TIMEOUT = 10
DELAY_BETWEEN_REQUESTS = 0.15


# ---------------------------------------------------------
# HTTP session
# ---------------------------------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0 Safari/537.36"
    )
})

retry = Retry(
    total=2,
    connect=2,
    read=2,
    backoff_factor=0.4,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"]
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
# Parse bookmark file
# ---------------------------------------------------------

def parse_bookmarks(filename):
    """
    Parse Chrome/Netscape bookmark HTML while preserving folder paths.
    """

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

        # Folder declaration
        folder_match = h3_pattern.search(line)

        if folder_match:
            fragment = BeautifulSoup(
                folder_match.group(1),
                "html.parser"
            )

            pending_folder = fragment.get_text(strip=True)
            continue

        # Enter folder
        if re.search(r"<DL><p>", line, re.IGNORECASE):

            if pending_folder is not None:
                folder_stack.append(pending_folder)
                pending_folder = None

            continue

        # Bookmark
        bookmark_match = bookmark_pattern.search(line)

        if bookmark_match:
            fragment = BeautifulSoup(
                bookmark_match.group(0),
                "html.parser"
            )

            link = fragment.find("a")

            if link is None:
                continue

            url = link.get("href", "")

            bookmarks.append({
                "title": link.get_text(strip=True),
                "url": url,
                "folder": " / ".join(folder_stack),
                "add_date": link.get("add_date", "")
            })

            continue

        # Leave folder
        if re.search(r"</DL><p>", line, re.IGNORECASE):

            if folder_stack:
                folder_stack.pop()

    return bookmarks


# ---------------------------------------------------------
# URL classification
# ---------------------------------------------------------

def classify_http_status(status_code):
    """
    Interpret common HTTP status codes.
    """

    if status_code is None:
        return "uncertain"

    if 200 <= status_code < 400:
        return "alive"

    if status_code in {
        401,
        403,
        405,
        407,
        408,
        409,
        423,
        425,
        429
    }:
        return "uncertain"

    if status_code in {
        404,
        410,
        451
    }:
        return "dead"

    if 400 <= status_code < 500:
        return "uncertain"

    if status_code >= 500:
        return "uncertain"

    return "uncertain"


# ---------------------------------------------------------
# Check a single bookmark
# ---------------------------------------------------------

def check_bookmark(bookmark):

    url = bookmark["url"]

    result = {
        **bookmark,
        "final_url": "",
        "http_status": "",
        "classification": "",
        "redirected": False,
        "redirect_count": 0,
        "error": ""
    }

    # -----------------------------------------------------
    # Special bookmark types
    # -----------------------------------------------------

    if not url:
        result["classification"] = "uncertain"
        result["error"] = "Empty URL"
        return result

    if url.startswith("javascript:"):
        result["classification"] = "special"
        result["error"] = "JavaScript bookmarklet"
        return result

    if url.startswith("chrome-extension://"):
        result["classification"] = "special"
        result["error"] = "Chrome extension URL"
        return result

    if url.startswith(("file://", "about:", "data:")):
        result["classification"] = "special"
        result["error"] = "Local or special browser URL"
        return result

    if not url.startswith(("http://", "https://")):
        result["classification"] = "special"
        result["error"] = "Unsupported URL scheme"
        return result

    # -----------------------------------------------------
    # HTTP request
    # -----------------------------------------------------

    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True
        )

        result["http_status"] = response.status_code
        result["final_url"] = response.url

        result["redirect_count"] = len(response.history)
        result["redirected"] = len(response.history) > 0

        result["classification"] = classify_http_status(
            response.status_code
        )

        response.close()

    except requests.exceptions.SSLError as e:
        result["classification"] = "uncertain"
        result["error"] = f"SSL error: {e}"

    except requests.exceptions.ConnectTimeout:
        result["classification"] = "uncertain"
        result["error"] = "Connection timeout"

    except requests.exceptions.ReadTimeout:
        result["classification"] = "uncertain"
        result["error"] = "Read timeout"

    except requests.exceptions.TooManyRedirects:
        result["classification"] = "uncertain"
        result["error"] = "Too many redirects"

    except requests.exceptions.ConnectionError as e:

        error_text = str(e)

        # DNS failures / refused connection are highly
        # indicative of dead links, but still not 100%.
        if (
            "NameResolutionError" in error_text
            or "Failed to resolve" in error_text
            or "nodename nor servname provided" in error_text
        ):
            result["classification"] = "dead"
            result["error"] = "DNS resolution failed"

        else:
            result["classification"] = "uncertain"
            result["error"] = f"Connection error: {error_text}"

    except requests.RequestException as e:
        result["classification"] = "uncertain"
        result["error"] = str(e)

    return result


# ---------------------------------------------------------
# Run checker
# ---------------------------------------------------------

bookmarks = parse_bookmarks(INPUT_FILE)

print(f"\nFound {len(bookmarks)} bookmarks.\n")


results = []

counts = {
    "alive": 0,
    "dead": 0,
    "uncertain": 0,
    "special": 0
}


for index, bookmark in enumerate(bookmarks, start=1):

    print(
        f"[{index:>3}/{len(bookmarks)}] "
        f"{bookmark['title'][:60]}"
    )

    result = check_bookmark(bookmark)

    results.append(result)

    classification = result["classification"]

    counts[classification] = (
        counts.get(classification, 0) + 1
    )

    status = result["http_status"]

    if result["error"]:
        print(
            f"     -> {classification.upper()} "
            f"{result['error'][:100]}"
        )

    else:
        print(
            f"     -> {classification.upper()} "
            f"HTTP {status}"
        )

        if result["redirected"]:
            print(
                f"        redirect -> "
                f"{result['final_url'][:100]}"
            )

    time.sleep(DELAY_BETWEEN_REQUESTS)


# ---------------------------------------------------------
# CSV output
# ---------------------------------------------------------

fields = [
    "title",
    "folder",
    "url",
    "final_url",
    "http_status",
    "classification",
    "redirected",
    "redirect_count",
    "add_date",
    "error"
]


with open(
    CSV_OUTPUT,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    writer.writerows(results)


# ---------------------------------------------------------
# JSON output
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
# Summary
# ---------------------------------------------------------

print("\n================================")
print("Bookmark check complete")
print("================================")

print(f"Alive:      {counts['alive']}")
print(f"Dead:       {counts['dead']}")
print(f"Uncertain:  {counts['uncertain']}")
print(f"Special:    {counts['special']}")

print(f"\nTotal:      {len(results)}")

print("\nCreated:")
print(f"  {CSV_OUTPUT}")
print(f"  {JSON_OUTPUT}")