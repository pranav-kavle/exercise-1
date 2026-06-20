#!/usr/bin/env python3
"""
extract_structured.py

Extract structured data from unstructured documents (HTML, Markdown, plain text)
using the Anthropic Claude API and a user-supplied JSON schema.

Usage:
    python extract_structured.py --input <file> --schema <schema.json> [--output <out.json>]

Examples:
    python extract_structured.py --input invoice.html --schema invoice_schema.json
    python extract_structured.py --input resume.md   --schema resume_schema.json --output result.json
    python extract_structured.py --input notes.txt   --schema notes_schema.json

Requirements:
    pip install anthropic markdownify
"""

# Test

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_type(path: Path) -> str:
    """Return 'html', 'markdown', or 'text' based on file extension."""
    ext = path.suffix.lower()
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".md", ".markdown"}:
        return "markdown"
    return "text"


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, retaining as much content as possible."""
    try:
        from markdownify import markdownify as md
        return md(
            html,
            heading_style="ATX",        # # H1, ## H2, …
            bullets="-",                 # consistent list bullets
            strip=["script", "style"],   # drop non-content tags
            convert_as_inline=[],
        )
    except ImportError:
        # Graceful fallback: strip tags with stdlib only
        import re
        print(
            "[warning] 'markdownify' not installed — falling back to basic tag stripping.\n"
            "          Install it for much better HTML→Markdown conversion:\n"
            "          pip install markdownify",
            file=sys.stderr,
        )
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


def load_document(path: Path) -> tuple[str, str]:
    """
    Read a document file and return (content_as_markdown_or_text, detected_type).
    HTML files are converted to Markdown before being returned.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    doc_type = detect_type(path)

    if doc_type == "html":
        print(f"[info] Detected HTML — converting to Markdown …")
        content = html_to_markdown(raw)
        return content, "html"

    return raw, doc_type


def load_schema(path: Path) -> dict:
    """Load and validate that the schema file is valid JSON."""
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        sys.exit(f"[error] Schema file is not valid JSON: {exc}")
    if not isinstance(schema, dict):
        sys.exit("[error] Schema must be a JSON object (dict).")
    return schema


# ---------------------------------------------------------------------------
# Claude extraction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise data-extraction assistant.
The user will provide a document (plain text or Markdown) and a JSON schema.
Your task is to extract every piece of information that matches the schema
from the document and return a single, valid JSON object that conforms to
the schema exactly.

Rules:
- Return ONLY the JSON object — no explanation, no markdown fences.
- Use null for fields that are absent from the document.
- Do not invent or infer data that is not present in the document.
- Preserve original formatting of values (e.g. dates, phone numbers) unless
  the schema specifies a particular format.
- If the schema contains an array, extract all matching items.
"""


def build_user_message(document: str, schema: dict) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"## JSON Schema\n\n```json\n{schema_str}\n```\n\n"
        f"## Document\n\n{document}"
    )


def extract_with_claude(document: str, schema: dict, model: str = "claude-sonnet-4-6") -> dict:
    """Call the Anthropic API and return the extracted dict."""
    try:
        import anthropic
    except ImportError:
        sys.exit(
            "[error] 'anthropic' package not found.\n"
            "        Install it with:  pip install anthropic"
        )

    client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env

    print(f"[info] Sending document to {model} for extraction …")

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_message(document, schema)}
        ],
    )

    raw_text = response.content[0].text.strip()

    # Strip accidental markdown fences the model might still add
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        # drop first and last fence lines
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw_text = "\n".join(inner).strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[warning] Model returned non-JSON content. Raw output:\n{raw_text}", file=sys.stderr)
        sys.exit(f"[error] Could not parse model response as JSON: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured data from a document using Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i", required=True, metavar="FILE",
        help="Path to the input document (.html, .htm, .md, .markdown, .txt, or any text file)",
    )
    parser.add_argument(
        "--schema", "-s", required=True, metavar="SCHEMA_JSON",
        help="Path to the JSON schema file describing the desired output structure",
    )
    parser.add_argument(
        "--output", "-o", metavar="OUT_JSON", default=None,
        help="Optional path to write the extracted JSON (defaults to stdout)",
    )
    parser.add_argument(
        "--model", "-m", default="claude-sonnet-4-6",
        help="Anthropic model to use (default: claude-sonnet-4-6)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path  = Path(args.input)
    schema_path = Path(args.schema)

    # Validate paths
    if not input_path.exists():
        sys.exit(f"[error] Input file not found: {input_path}")
    if not schema_path.exists():
        sys.exit(f"[error] Schema file not found: {schema_path}")

    # Load inputs
    document, doc_type = load_document(input_path)
    schema             = load_schema(schema_path)

    print(f"[info] Input type : {doc_type} ({input_path.name})")    
    print(f"[info] Schema     : {schema_path.name}")
    print(f"[info] Document length: {len(document):,} chars")

    # Extract
    result = extract_with_claude(document, schema, model=args.model)

    # Output
    result_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(result_json + "\n", encoding="utf-8")
        print(f"[info] Extracted JSON written to: {out_path}")
    else:
        print("\n--- Extracted JSON ---")
        print(result_json)


if __name__ == "__main__":
    main()