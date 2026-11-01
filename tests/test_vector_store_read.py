import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.retrieval.vector_store import CodeVectorStore

# --------------------------------------------------
# STEP 1: Connect to existing vector store
# --------------------------------------------------

vector_store = CodeVectorStore()

# --------------------------------------------------
# STEP 2: Check record count
# --------------------------------------------------

count = vector_store.count()

print("=" * 70)
print("VECTOR STORE VERIFICATION")
print("=" * 70)

print()
print("Total records:", count)

# --------------------------------------------------
# STEP 3: Verify that records exist
# --------------------------------------------------

if count == 0:
    print()
    print("=" * 70)
    print("STATUS: FAIL")
    print("=" * 70)

    raise AssertionError(
        "Vector store is empty. No records available."
    )

# --------------------------------------------------
# STEP 4: Get one existing record ID
# --------------------------------------------------

records = vector_store.collection.get(
    limit=1,
    include=[
        "documents",
        "metadatas",
    ],
)

ids = records["ids"]

if not ids:
    print()
    print("=" * 70)
    print("STATUS: FAIL")
    print("=" * 70)

    raise AssertionError(
        "Vector store contains records, "
        "but no record IDs were returned."
    )

chunk_id = ids[0]

# --------------------------------------------------
# STEP 5: Retrieve the record using its real ID
# --------------------------------------------------

result = vector_store.get_chunk(chunk_id)

# --------------------------------------------------
# STEP 6: Display retrieved data
# --------------------------------------------------

print()
print("=" * 70)
print("RETRIEVED RECORD")
print("=" * 70)

print()

print("ID:")
print(result["ids"])

print()

print("Metadata:")
print(result["metadatas"])

print()

print("Document:")
print(result["documents"][0])

# --------------------------------------------------
# STEP 7: Validate retrieved record
# --------------------------------------------------

if not result["ids"]:
    raise AssertionError(
        "Record ID was not retrieved."
    )

if not result["documents"]:
    raise AssertionError(
        "Record document was not retrieved."
    )

if not result["metadatas"]:
    raise AssertionError(
        "Record metadata was not retrieved."
    )

print()
print("=" * 70)
print("STATUS: PASS")
print("=" * 70)