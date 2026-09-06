import re
import html
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup

INPUT_FILE = "bookmarks.html"
OUTPUT_FILE = "bookmarks_by_year.html"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_year(add_date):
    if not add_date:
        return "Unknown"

    try:
        return str(datetime.fromtimestamp(int(add_date)).year)
    except (ValueError, TypeError, OverflowError):
        return "Unknown"


def clean_text(value):
    """Decode HTML entities and remove any accidental HTML tags."""
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


# ---------------------------------------------------------
# Parse Chrome/Netscape bookmark file
# ---------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()


bookmarks = defaultdict(list)

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

    # -----------------------------------------------------
    # Folder declaration
    # -----------------------------------------------------

    folder_match = h3_pattern.search(line)

    if folder_match:
        pending_folder = clean_text(folder_match.group(1))
        continue


    # -----------------------------------------------------
    # Opening <DL> means we enter the folder that was
    # declared immediately before it.
    # -----------------------------------------------------

    if re.search(r"<DL><p>", line, re.IGNORECASE):

        if pending_folder is not None:
            folder_stack.append(pending_folder)
            pending_folder = None

        continue


    # -----------------------------------------------------
    # Bookmark
    # -----------------------------------------------------

    bookmark_match = bookmark_pattern.search(line)

    if bookmark_match:

        bookmark_html = bookmark_match.group(0)

        # Parsing one individual <A> is safe and convenient.
        fragment = BeautifulSoup(bookmark_html, "html.parser")
        link = fragment.find("a")

        if link is None:
            continue

        year = get_year(link.get("add_date"))

        bookmarks[year].append({
            "path": tuple(folder_stack),
            "title": link.get_text(),
            "attrs": dict(link.attrs)
        })

        continue


    # -----------------------------------------------------
    # Closing folder
    # -----------------------------------------------------

    if re.search(r"</DL><p>", line, re.IGNORECASE):

        if folder_stack:
            folder_stack.pop()


# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print("\nFound bookmarks:\n")

years_numeric = sorted(
    [year for year in bookmarks if year != "Unknown"],
    key=int,
    reverse=True
)

years = years_numeric.copy()

if "Unknown" in bookmarks:
    years.append("Unknown")


total = 0

for year in years:
    count = len(bookmarks[year])
    total += count
    print(f"{year}: {count}")

print(f"\nTOTAL: {total}")


if total == 0:
    raise RuntimeError("No bookmarks found.")


# ---------------------------------------------------------
# Build folder tree
# ---------------------------------------------------------

def build_tree(items):

    tree = {
        "_bookmarks": []
    }

    for bookmark in items:

        node = tree

        for folder in bookmark["path"]:

            if folder not in node:
                node[folder] = {
                    "_bookmarks": []
                }

            node = node[folder]

        node["_bookmarks"].append(bookmark)

    return tree


# ---------------------------------------------------------
# HTML output helpers
# ---------------------------------------------------------

def escape(value):
    return html.escape(str(value), quote=True)


def bookmark_html(bookmark, indent):

    attrs = []

    for key, value in bookmark["attrs"].items():

        if isinstance(value, list):
            value = " ".join(value)

        attrs.append(
            f'{key.upper()}="{escape(value)}"'
        )

    attributes = " ".join(attrs)

    return (
        "    " * indent
        + f'<DT><A {attributes}>'
        + escape(bookmark["title"])
        + "</A>"
    )


output = [
    '<!DOCTYPE NETSCAPE-Bookmark-file-1>',
    '<!-- Automatically generated bookmark file -->',
    '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
    '<TITLE>Bookmarks by Year</TITLE>',
    '<H1>Bookmarks by Year</H1>',
    '<DL><p>'
]


def write_tree(tree, indent):

    # bookmarks directly inside this folder
    for bookmark in tree["_bookmarks"]:
        output.append(
            bookmark_html(bookmark, indent)
        )

    # subfolders
    for name, subtree in tree.items():

        if name == "_bookmarks":
            continue

        output.append(
            "    " * indent
            + f"<DT><H3>{escape(name)}</H3>"
        )

        output.append(
            "    " * indent
            + "<DL><p>"
        )

        write_tree(
            subtree,
            indent + 1
        )

        output.append(
            "    " * indent
            + "</DL><p>"
        )


# ---------------------------------------------------------
# Create one top-level folder per year
# ---------------------------------------------------------

for year in years:

    output.append(
        f'    <DT><H3>{escape(year)}</H3>'
    )

    output.append(
        '    <DL><p>'
    )

    tree = build_tree(
        bookmarks[year]
    )

    write_tree(
        tree,
        2
    )

    output.append(
        '    </DL><p>'
    )


output.append('</DL><p>')


# ---------------------------------------------------------
# Write output file
# ---------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(output)
    )


print(
    f"\nCreated: {OUTPUT_FILE}"
)