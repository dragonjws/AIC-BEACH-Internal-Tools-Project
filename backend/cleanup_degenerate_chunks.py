"""
Surgically removes degenerate chunks (ones whose actual content is just a
bare 'URL: http...' line, or otherwise too short to be a real answer) from
an EXISTING Chroma collection -- without re-fetching from S3 or re-embedding
everything else.

This does NOT fix chunk boundaries retroactively (that would require
re-chunking the affected source documents), but it does stop those junk
chunks from ever being returned as an "answer" again, since they're removed
from the collection entirely.

Run this once after applying the load_and_chunk_documents fix, instead of
a full rm -rf ./my_local_db rebuild.

Usage:
    python cleanup_degenerate_chunks.py
    python cleanup_degenerate_chunks.py --dry-run   # preview what would be deleted
"""

import argparse
import chromadb

MIN_CONTENT_LENGTH = 40  # matches MIN_CHUNK_CONTENT_LENGTH in main.py


def get_actual_content(document_text: str) -> str:
    """Chunks are stored as 'Context: <title> \\nContent: <original text>'."""
    marker = "\nContent: "
    idx = document_text.find(marker)
    if idx != -1:
        return document_text[idx + len(marker):].strip()
    return document_text.strip()


def is_degenerate(content_text: str) -> bool:
    actual = get_actual_content(content_text)
    if len(actual) < MIN_CONTENT_LENGTH:
        return True
    if actual.startswith("URL: http"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="./my_local_db")
    parser.add_argument("--collection", default="test_docs")
    parser.add_argument("--dry-run", action="store_true", help="preview without deleting")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_or_create_collection(name=args.collection)

    print("Fetching all chunks from collection (this may take a moment for large collections)...")
    data = collection.get(include=["documents", "metadatas"])

    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    print(f"Total chunks in collection: {len(ids)}")

    bad_ids = []
    for chunk_id, doc_text, meta in zip(ids, documents, metadatas):
        if is_degenerate(doc_text):
            bad_ids.append(chunk_id)

    print(f"Found {len(bad_ids)} degenerate chunks ({len(bad_ids)/max(len(ids),1)*100:.2f}% of total)")

    if bad_ids:
        print("\nSample of chunks that will be removed:")
        for chunk_id in bad_ids[:10]:
            idx = ids.index(chunk_id)
            title = metadatas[idx].get("title", "unknown")
            preview = get_actual_content(documents[idx])[:80]
            print(f"  - {chunk_id} | {title} | content: {preview!r}")

    if args.dry_run:
        print(f"\nDry run: would delete {len(bad_ids)} chunks. Re-run without --dry-run to actually delete.")
        return

    if not bad_ids:
        print("Nothing to clean up.")
        return

    confirm = input(f"\nDelete these {len(bad_ids)} chunks from the collection? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Aborted, nothing deleted.")
        return

    # Chroma also has a batch size cap on deletes; chunk it defensively.
    max_batch = client.get_max_batch_size()
    for start in range(0, len(bad_ids), max_batch):
        end = start + max_batch
        collection.delete(ids=bad_ids[start:end])
        print(f"  deleted {start}-{min(end, len(bad_ids))} of {len(bad_ids)}")

    print(f"\nDone. Removed {len(bad_ids)} degenerate chunks.")
    print("Note: this does not re-chunk the affected source documents, so if a document's")
    print("useful content was split awkwardly around the removed URL-only fragment, that")
    print("document now simply has one fewer (junk) chunk -- its real content chunks are untouched.")


if __name__ == "__main__":
    main()