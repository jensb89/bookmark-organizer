import re
import html
from datetime import datetime
from collections import defaultdict
from bs4 import BeautifulSoup

INPUT_FILE = "bookmarks.html"

ARCHIVE_FILE = "bookmarks_archive_2012_2014.html"
ANALYZE_FILE = "bookmarks_to_analyze_2015_2026.html"

ARCHIVE_START = 2012
ARCHIVE_END = 2014


def get_year(add_date):
    if not add_date:
        return None

    try:
        return datetime.fromtimestamp(int(add_date)).year
    except (ValueError, TypeError, OverflowError):
        return None


def clean_text(value):
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


# ---------------------------------------------------------
# Parse original bookmark export
# ---------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()


all_bookmarks = []

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
        pending_folder = clean_text(folder_match.group(1))
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
        bookmark_html = bookmark_match.group(0)

        fragment = BeautifulSoup(
            bookmark_html,
            "html.parser"
        )

        link = fragment.find("a")

        if link is None:
            continue

        year = get_year(
            link.get("add_date")
        )

        all_bookmarks.append({
            "year": year,
            "path": tuple(folder_stack),
            "title": link.get_text(),
            "attrs": dict(link.attrs)
        })

        continue

    # Leave folder
    if re.search(r"</DL><p>", line, re.IGNORECASE):

        if folder_stack:
            folder_stack.pop()


# ---------------------------------------------------------
# Split archive vs active
# ---------------------------------------------------------

archive_bookmarks = []
analyze_bookmarks = []
unknown_bookmarks = []


for bookmark in all_bookmarks:

    year = bookmark["year"]

    if year is None:
        unknown_bookmarks.append(bookmark)

    elif ARCHIVE_START <= year <= ARCHIVE_END:
        archive_bookmarks.append(bookmark)

    else:
        analyze_bookmarks.append(bookmark)


# ---------------------------------------------------------
# HTML output helpers
# ---------------------------------------------------------

def escape(value):
    return html.escape(str(value), quote=True)


def bookmark_to_html(bookmark, indent):

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


def build_tree(bookmarks):

    tree = {
        "_bookmarks": []
    }

    for bookmark in bookmarks:

        node = tree

        for folder in bookmark["path"]:

            if folder not in node:
                node[folder] = {
                    "_bookmarks": []
                }

            node = node[folder]

        node["_bookmarks"].append(
            bookmark
        )

    return tree


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


def write_bookmark_file(filename, title, bookmarks):

    output = [
        '<!DOCTYPE NETSCAPE-Bookmark-file-1>',
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        f'<TITLE>{escape(title)}</TITLE>',
        f'<H1>{escape(title)}</H1>',
        '<DL><p>'
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
        '</DL><p>'
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
# Write both files
# ---------------------------------------------------------

write_bookmark_file(
    ARCHIVE_FILE,
    "Bookmark Archive 2012-2014",
    archive_bookmarks
)

write_bookmark_file(
    ANALYZE_FILE,
    "Bookmarks to Analyze",
    analyze_bookmarks + unknown_bookmarks
)


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

year_counts = defaultdict(int)

for bookmark in all_bookmarks:
    year_counts[bookmark["year"]] += 1


print("\nBookmarks by year:\n")

for year in sorted(
    [y for y in year_counts if y is not None],
    reverse=True
):
    print(
        f"{year}: {year_counts[year]}"
    )

if None in year_counts:
    print(
        f"Unknown: {year_counts[None]}"
    )


print("\n--------------------------------")
print(f"TOTAL:       {len(all_bookmarks)}")
print(f"ARCHIVE:     {len(archive_bookmarks)}")
print(f"TO ANALYZE:  {len(analyze_bookmarks)}")
print(f"UNKNOWN:     {len(unknown_bookmarks)}")
print("--------------------------------")

print(f"\nCreated:")
print(f"  {ARCHIVE_FILE}")
print(f"  {ANALYZE_FILE}")