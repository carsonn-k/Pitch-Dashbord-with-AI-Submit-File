from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

from pitch_staff_dashboard.data import DEFAULT_CACHE_DIR, iter_date_chunks
from pitch_staff_dashboard.pitch_data import ALL_PITCHES_LAST5_PARQUET_PATH, last_five_year_window, prepare_pitch_storage_frame, write_pitch_parquet


def parse_args() -> argparse.Namespace:
    default_start, default_end = last_five_year_window()
    parser = argparse.ArgumentParser(description="Build one large Parquet file with MLB Statcast pitch-level rows for a date range.")
    parser.add_argument("--start", default=default_start.isoformat(), help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=default_end.isoformat(), help="End date, YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=7, help="Date batch size for Baseball Savant pulls")
    parser.add_argument("--output", default=str(ALL_PITCHES_LAST5_PARQUET_PATH), help="Output Parquet path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output Parquet file")
    parser.add_argument("--retries", type=int, default=3, help="Retries per date chunk")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between chunks")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Directory for reusable per-chunk Parquet cache files")
    parser.add_argument("--refresh-cache", action="store_true", help="Refetch chunks even when a cached Parquet chunk exists")
    return parser.parse_args()


def align_to_schema(df: pd.DataFrame, schema: pa.Schema) -> pd.DataFrame:
    aligned = df.copy()
    for col in schema.names:
        if col not in aligned.columns:
            aligned[col] = pd.NA
    return aligned[schema.names]


def iter_calendar_week_chunks(start_dt: str, end_dt: str) -> list[tuple[str, str]]:
    """Use partial edge chunks plus Monday-Sunday weeks to maximize cache reuse."""
    start = pd.to_datetime(start_dt).date()
    end = pd.to_datetime(end_dt).date()
    chunks: list[tuple[str, str]] = []
    cursor = start

    if cursor.weekday() != 0:
        first_end = min(cursor + timedelta(days=6 - cursor.weekday()), end)
        chunks.append((cursor.isoformat(), first_end.isoformat()))
        cursor = first_end + timedelta(days=1)

    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=6), end)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)

    return chunks


def build_chunks(start_dt: str, end_dt: str, chunk_days: int) -> list[tuple[str, str]]:
    if chunk_days == 7:
        return iter_calendar_week_chunks(start_dt, end_dt)
    return list(iter_date_chunks(start_dt, end_dt, chunk_days=chunk_days))


def pull_with_retries(statcast, chunk_start: str, chunk_end: str, retries: int) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return statcast(start_dt=chunk_start, end_dt=chunk_end, parallel=False)
        except TypeError:
            return statcast(start_dt=chunk_start, end_dt=chunk_end)
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{retries} failed: {exc}", flush=True)
            time.sleep(min(10, attempt * 2))
    assert last_exc is not None
    raise last_exc


def chunk_cache_path(cache_dir: Path, chunk_start: str, chunk_end: str) -> Path:
    return cache_dir / f"statcast_{chunk_start}_{chunk_end}.parquet"


def cache_chunk_is_readable(path: Path) -> bool:
    try:
        columns = pq.ParquetFile(path).schema_arrow.names
        probe_cols = [col for col in ["pitcher", "game_date"] if col in columns]
        pd.read_parquet(path, columns=probe_cols[:1] or None)
        return True
    except Exception as exc:
        print(f"    cached chunk unreadable; refetching {path.name}: {exc}", flush=True)
        return False


def read_or_fetch_chunk(
    statcast,
    cache_dir: Path,
    chunk_start: str,
    chunk_end: str,
    *,
    retries: int,
    compression: str,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_path = chunk_cache_path(cache_dir, chunk_start, chunk_end)
    if cache_path.exists() and not refresh_cache:
        if cache_chunk_is_readable(cache_path):
            print(f"    using cached {cache_path}", flush=True)
            return pd.read_parquet(cache_path)
        cache_path.unlink()

    pulled = pull_with_retries(statcast, chunk_start, chunk_end, retries)
    if pulled is None or pulled.empty:
        return pd.DataFrame()

    prepared = prepare_pitch_storage_frame(pulled)
    write_pitch_parquet(prepared, cache_path)
    return prepared


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    cache_dir = Path(args.cache_dir)

    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} already exists. Use --overwrite to rebuild it.")
    if output.exists() and output.is_dir():
        raise SystemExit(f"{output} is a directory. Remove it or choose a file path before rebuilding.")

    output.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()

    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError("pybaseball is not installed. Run `pip install -r requirements.txt`.") from exc

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    total_rows = 0
    chunk_count = 0

    try:
        for chunk_start, chunk_end in build_chunks(args.start, args.end, args.chunk_days):
            chunk_count += 1
            print(f"[{chunk_count}] Loading {chunk_start} to {chunk_end}...", flush=True)
            chunk_df = read_or_fetch_chunk(
                statcast,
                cache_dir,
                chunk_start,
                chunk_end,
                retries=args.retries,
                compression=args.compression,
                refresh_cache=args.refresh_cache,
            )

            if chunk_df.empty:
                print("    0 rows", flush=True)
                continue

            chunk_df = prepare_pitch_storage_frame(chunk_df)
            if schema is None:
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(partial, schema=schema, compression=args.compression)
            else:
                table = pa.Table.from_pandas(align_to_schema(chunk_df, schema), schema=schema, preserve_index=False)

            writer.write_table(table)
            total_rows += len(chunk_df)
            print(f"    wrote {len(chunk_df):,} rows; total {total_rows:,}", flush=True)
            time.sleep(args.sleep)
    finally:
        if writer is not None:
            writer.close()

    if schema is None:
        pd.DataFrame().to_parquet(partial, index=False, compression=args.compression)

    partial.replace(output)
    print(f"Wrote {total_rows:,} rows to {output}", flush=True)


if __name__ == "__main__":
    main()
