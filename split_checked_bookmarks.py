import json
import html
from collections import defaultdict

INPUT_FILE = "bookmark_check_results.json"

OUTPUT_FILES = {
    "alive": "bookmarks_alive.html",
    "dead": "bookmarks_dead.html",
    "uncertain": "bookmarks_uncertain.html",
    "special": "bookmarks_special.html",
}


# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


print(f"Loaded {len(results)} bookmark results.")


# ---------------------------------------------------------
# Group by classification
# ---------------------------------------------------------

groups = defaultdict(list)

for bookmark in results:
    classification = bookmark.get(
        "classification",
        "uncertain"
    )

    groups[classification].append(bookmark)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def escape(value):
    return html.escape(
        str(value),
        quote=True
    )


def build_tree(bookmarks):
    """
    Recreate the original folder hierarchy.

    The JSON contains folders as:
        "Lesezeichenleiste / Coding / AI"
    """

    root = {
        "_bookmarks": []
    }

    for bookmark in bookmarks:

        folder = bookmark.get(
            "folder",
            ""
        ).strip()

        if folder:
            path = tuple(
                part.strip()
                for part in folder.split(" / ")
                if part.strip()
            )
        else:
            path = ()

        node = root

        for folder_name in path:

            if folder_name not in node:
                node[folder_name] = {
                    "_bookmarks": []
                }

            node = node[folder_name]

        node["_bookmarks"].append(
            bookmark
        )

    return root


def bookmark_to_html(bookmark, indent):

    title = bookmark.get(
        "title",
        "Untitled"
    )

    url = bookmark.get(
        "url",
        ""
    )

    add_date = bookmark.get(
        "add_date",
        ""
    )

    attributes = [
        f'HREF="{escape(url)}"'
    ]

    if add_date:
        attributes.append(
            f'ADD_DATE="{escape(add_date)}"'
        )

    attrs_text = " ".join(attributes)

    return (
        "    " * indent
        + f"<DT><A {attrs_text}>"
        + escape(title)
        + "</A>"
    )


def write_tree(tree, output, indent):

    # bookmarks directly in current folder
    for bookmark in tree.get(
        "_bookmarks",
        []
    ):
        output.append(
            bookmark_to_html(
                bookmark,
                indent
            )
        )

    # folders
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
        f'<TITLE>{escape(title)}</TITLE>',
        f'<H1>{escape(title)}</H1>',
        '<DL><p>'
    ]

    tree = build_tree(bookmarks)

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
# Create files
# ---------------------------------------------------------

titles = {
    "alive": "Alive Bookmarks",
    "dead": "Dead Bookmarks",
    "uncertain": "Uncertain Bookmarks",
    "special": "Special Bookmarks",
}


for classification, filename in OUTPUT_FILES.items():

    bookmarks = groups.get(
        classification,
        []
    )

    write_bookmark_file(
        filename,
        titles[classification],
        bookmarks
    )

    print(
        f"{classification:10}: "
        f"{len(bookmarks):3} -> {filename}"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\nCreated bookmark files:")

for classification, filename in OUTPUT_FILES.items():
    print(
        f"  {filename}"
    )