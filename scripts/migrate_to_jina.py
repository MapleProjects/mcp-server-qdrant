"""Migrate the memory collection to a long-context multilingual embedding model.

Reads every point from the source collection, drops exact duplicates, re-embeds
the survivors with the target model and writes them to a new collection keeping
the original point ids. Resumable: progress is checkpointed to a JSON file.

Usage:
    python scripts/migrate_to_jina.py \
        --source agents-memory \
        --target agents-memory-v2 \
        --model jinaai/jina-embeddings-v2-base-es
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client import QdrantClient, models  # noqa: E402

from mcp_server_qdrant.qdrant import (  # noqa: E402
    CONTENT_HASH_PATH,
    CREATED_AT_PATH,
    DOCUMENT_PATH,
    UPDATED_AT_PATH,
    content_fingerprint,
)
from mcp_server_qdrant.settings import METADATA_PATH  # noqa: E402

BATCH = 16


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_created_at(metadata: dict | None) -> str:
    """Use the recorded date when the memory carries one, else now."""
    if isinstance(metadata, dict):
        raw = metadata.get("date")
        if isinstance(raw, str):
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(raw.strip()[:19], fmt)
                    return parsed.replace(tzinfo=timezone.utc).isoformat(
                        timespec="seconds"
                    )
                except ValueError:
                    continue
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_source(client: QdrantClient, source: str) -> list[dict]:
    points: list[dict] = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=source,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            points.append(
                {
                    "id": str(point.id),
                    "content": payload.get(DOCUMENT_PATH, "") or "",
                    "metadata": payload.get(METADATA_PATH),
                }
            )
        if offset is None:
            break
    return points


def dedupe(points: list[dict]) -> tuple[list[dict], int]:
    """Keep one point per (content, metadata) fingerprint.

    The survivor is the one with the richest metadata, then the earliest id, so
    the result is deterministic across runs.
    """
    best: dict[str, dict] = {}
    for point in points:
        fingerprint = content_fingerprint(point["content"], point["metadata"])
        point["fingerprint"] = fingerprint
        current = best.get(fingerprint)
        if current is None:
            best[fingerprint] = point
            continue
        current_score = len(current["metadata"] or {})
        new_score = len(point["metadata"] or {})
        if new_score > current_score or (
            new_score == current_score and point["id"] < current["id"]
        ):
            best[fingerprint] = point
    kept = list(best.values())
    return kept, len(points) - len(kept)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--source", default="agents-memory")
    parser.add_argument("--target", default="agents-memory-v2")
    parser.add_argument("--model", default="jinaai/jina-embeddings-v2-base-es")
    parser.add_argument("--checkpoint", default="/tmp/migrate_jina_checkpoint.json")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    client = QdrantClient(url=args.url, check_compatibility=False)

    log(f"leyendo {args.source}")
    points = read_source(client, args.source)
    log(f"leidos {len(points)} puntos")

    kept, removed = dedupe(points)
    log(f"duplicados exactos eliminados: {removed} | a migrar: {len(kept)}")

    from fastembed import TextEmbedding

    log(f"cargando modelo {args.model}")
    t0 = time.time()
    model = TextEmbedding(args.model)
    dim = model._get_model_description(args.model).dim
    vector_name = f"fast-{args.model.split('/')[-1].lower()}"
    log(f"modelo listo en {time.time() - t0:.1f}s | dim={dim} | vector={vector_name}")

    if args.recreate and client.collection_exists(args.target):
        log(f"borrando coleccion {args.target}")
        client.delete_collection(args.target)

    if not client.collection_exists(args.target):
        client.create_collection(
            collection_name=args.target,
            vectors_config={
                vector_name: models.VectorParams(
                    size=dim, distance=models.Distance.COSINE
                )
            },
        )
        log(f"coleccion {args.target} creada")
        for field, schema in (
            (CONTENT_HASH_PATH, models.PayloadSchemaType.KEYWORD),
            (CREATED_AT_PATH, models.PayloadSchemaType.DATETIME),
            (f"{METADATA_PATH}.category", models.PayloadSchemaType.KEYWORD),
            (f"{METADATA_PATH}.entity_name", models.PayloadSchemaType.KEYWORD),
            (f"{METADATA_PATH}.project", models.PayloadSchemaType.KEYWORD),
        ):
            try:
                client.create_payload_index(
                    collection_name=args.target, field_name=field, field_schema=schema
                )
            except Exception as exc:  # index already there
                log(f"  indice {field} no creado: {exc}")

    checkpoint_path = Path(args.checkpoint)
    done: set[str] = set()
    if checkpoint_path.exists():
        done = set(json.loads(checkpoint_path.read_text()))
        log(f"checkpoint: {len(done)} ya migrados")

    pending = [p for p in kept if p["id"] not in done]
    log(f"pendientes: {len(pending)}")

    started = time.time()
    for start in range(0, len(pending), BATCH):
        chunk = pending[start : start + BATCH]
        texts = [p["content"] for p in chunk]
        vectors = [v.tolist() for v in model.passage_embed(texts)]

        structs = []
        for point, vector in zip(chunk, vectors):
            created = parse_created_at(point["metadata"])
            structs.append(
                models.PointStruct(
                    id=point["id"],
                    vector={vector_name: vector},
                    payload={
                        DOCUMENT_PATH: point["content"],
                        METADATA_PATH: point["metadata"],
                        CONTENT_HASH_PATH: point["fingerprint"],
                        CREATED_AT_PATH: created,
                        UPDATED_AT_PATH: created,
                    },
                )
            )
        client.upsert(collection_name=args.target, points=structs, wait=True)

        done.update(p["id"] for p in chunk)
        checkpoint_path.write_text(json.dumps(sorted(done)))

        processed = start + len(chunk)
        elapsed = time.time() - started
        rate = processed / elapsed if elapsed else 0
        remaining = (len(pending) - processed) / rate if rate else 0
        log(
            f"{processed}/{len(pending)} | {rate:.1f} docs/s | "
            f"faltan {remaining / 60:.1f} min"
        )

    total = client.count(collection_name=args.target, exact=True).count
    log(f"migracion terminada | puntos en {args.target}: {total}")
    log(f"esperado: {len(kept)} | diferencia: {total - len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
