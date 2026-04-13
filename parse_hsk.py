import io
import os
import requests
import pdfplumber
from collections import defaultdict

PDFS = [
    # New HSK 3.0
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band1.pdf", "hsk3_band1.txt"),
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band2.pdf", "hsk3_band2.txt"),
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band3.pdf", "hsk3_band3.txt"),
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band4.pdf", "hsk3_band4.txt"),
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band5.pdf", "hsk3_band5.txt"),
    ("https://www.chinaeducenter.com/cecdl/vocabulary_band6.pdf", "hsk3_band6.txt"),
    # band789 is a scanned image PDF with no text layer and no English meanings — skipped
    # ("https://www.chinaeducenter.com/cecdl/vocabulary_band789.pdf", "hsk3_band789.txt"),
    # HSK 2.0
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level1.pdf", "hsk2_level1.txt"),
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level2.pdf", "hsk2_level2.txt"),
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level3.pdf", "hsk2_level3.txt"),
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level4.pdf", "hsk2_level4.txt"),
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level5.pdf", "hsk2_level5.txt"),
    ("https://www.chinaeducenter.com/cecdl/HSK_Vocabulary_Level6.pdf", "hsk2_level6.txt"),
]

OUTPUT_DIR = "output"


def download_pdf(url: str) -> bytes:
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def detect_columns(words: list[dict]) -> tuple[float, float, float, float]:
    """
    Detect the x-boundaries for [index, chinese, pinyin, meaning] columns.
    Returns (idx_max, chinese_max, pinyin_max) — thresholds between columns.
    Words with x0 <= idx_max are index, <= chinese_max are chinese, etc.
    """
    # Collect all x0 values (rounded)
    xs = [w["x0"] for w in words]
    if not xs:
        return (80, 200, 310, 9999)

    # Find clusters: sort unique x0s and look for big gaps
    unique_xs = sorted(set(round(x) for x in xs))

    # We expect 4 clusters: index (~50), chinese (~100-160), pinyin (~169-345), meaning (rest)
    # Find gaps between consecutive x values
    gaps = []
    for i in range(1, len(unique_xs)):
        gap = unique_xs[i] - unique_xs[i - 1]
        if gap > 15:
            gaps.append((gap, unique_xs[i - 1], unique_xs[i]))

    gaps.sort(reverse=True)

    # Take the 3 largest gaps as column separators
    separators = sorted([(lo + hi) / 2 for _, lo, hi in gaps[:3]])

    if len(separators) >= 3:
        return (separators[0], separators[1], separators[2])

    # Fallback: use fixed fractions of page width
    page_width = max(xs)
    return (page_width * 0.15, page_width * 0.42, page_width * 0.65)


def group_words_into_rows(words: list[dict], y_tolerance: float = 3.0) -> list[list[dict]]:
    """Group words into rows based on similar 'top' values."""
    if not words:
        return []

    rows = []
    current_row = [words[0]]
    current_top = words[0]["top"]

    for word in words[1:]:
        if abs(word["top"] - current_top) <= y_tolerance:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w["x0"]))
            current_row = [word]
            current_top = word["top"]

    if current_row:
        rows.append(sorted(current_row, key=lambda w: w["x0"]))

    return rows


def assign_column(x0: float, col1_max: float, col2_max: float, col3_max: float) -> int:
    """Return 0=index, 1=chinese, 2=pinyin, 3=meaning."""
    if x0 <= col1_max:
        return 0
    if x0 <= col2_max:
        return 1
    if x0 <= col3_max:
        return 2
    return 3


def parse_pdf(data: bytes) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (title, [(chinese, pinyin, meaning), ...])."""
    title = ""
    rows: list[tuple[str, str, str]] = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # Detect columns from first data page (page 0)
        first_page_words = pdf.pages[0].extract_words()
        # Skip title row (highest y = smallest top value)
        data_words = [w for w in first_page_words if w["top"] > first_page_words[0]["top"] + 5]
        col1_max, col2_max, col3_max = detect_columns(data_words)

        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()
            if not words:
                continue

            word_rows = group_words_into_rows(words)

            for row_idx, row_words in enumerate(word_rows):
                # Title: first row on first page
                if page_num == 0 and row_idx == 0:
                    title = " ".join(w["text"] for w in row_words)
                    continue

                # Assign each word to a column bucket
                cols: dict[int, list[str]] = defaultdict(list)
                for w in row_words:
                    col = assign_column(w["x0"], col1_max, col2_max, col3_max)
                    cols[col].append(w["text"])

                # Must have chinese (col 1) and at least one other content column
                if 1 not in cols:
                    continue

                # Check col 0 is a pure integer index — skip non-data rows
                index_text = " ".join(cols.get(0, []))
                if index_text and not index_text.strip().isdigit():
                    continue

                chinese = " ".join(cols.get(1, [])).strip()
                pinyin = " ".join(cols.get(2, [])).strip()
                meaning = " ".join(cols.get(3, [])).strip()

                if not chinese or not meaning:
                    continue

                rows.append((chinese, pinyin, meaning))

    return title, rows


def write_flashcards(rows: list[tuple[str, str, str]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for chinese, pinyin, meaning in rows:
            f.write(f"{chinese}\t{pinyin}\t{meaning}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for url, filename in PDFS:
        print(f"\nProcessing {filename} ...")
        try:
            pdf_data = download_pdf(url)
            title, rows = parse_pdf(pdf_data)
            out_path = os.path.join(OUTPUT_DIR, filename)
            write_flashcards(rows, out_path)
            print(f"  Title : {title!r}")
            print(f"  Cards : {len(rows)} → {out_path}")
        except Exception as e:
            import traceback
            print(f"  ERROR : {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
