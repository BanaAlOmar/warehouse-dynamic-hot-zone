"""Prepare the two public inputs used by the warehouse re-slotting experiments.

The Warehouse Science picks file contains unquoted thousands separators in the
last two quantity fields.  This parser preserves the seven leading fields and
reconstructs the two quantities without silently dropping malformed rows.
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PICK_COLUMNS = [
    "PICKER",
    "DT_START",
    "SKU",
    "FROM_LOC",
    "TO_LOC",
    "ORDER_ID",
    "ORDER_TYPE",
    "ACT_QTY",
    "REQ_QTY",
]


def valid_number_parts(parts: list[str]) -> bool:
    if not parts:
        return False
    first = parts[0].strip().lstrip("+-")
    if not first.replace(".", "", 1).isdigit():
        return False
    return all(p.strip().isdigit() and len(p.strip()) == 3 for p in parts[1:])


def split_quantities(parts: list[str]) -> tuple[str, str]:
    candidates: list[tuple[int, str, str]] = []
    for cut in range(1, len(parts)):
        left, right = parts[:cut], parts[cut:]
        if valid_number_parts(left) and valid_number_parts(right):
            a, b = "".join(left), "".join(right)
            score = int(a == b)
            candidates.append((score, a, b))
    if not candidates:
        raise ValueError(f"Cannot split quantity fields: {parts!r}")
    candidates.sort(reverse=True)
    _, actual, requested = candidates[0]
    return actual, requested


def split_sku_numeric_fields(parts: list[str]) -> list[str]:
    """Recover the 16 numeric item-master fields from unquoted CSV tokens."""

    solutions: list[tuple[float, list[str]]] = []

    def visit(position: int, column: int, values: list[str], score: float) -> None:
        remaining_tokens = len(parts) - position
        remaining_columns = 16 - column
        if remaining_tokens < remaining_columns:
            return
        if column == 16:
            if position == len(parts):
                solutions.append((score, values.copy()))
            return
        max_width = min(3, remaining_tokens - (remaining_columns - 1))
        for width in range(1, max_width + 1):
            group = parts[position : position + width]
            stripped = [x.strip() for x in group]
            if width == 1:
                if stripped[0] and not stripped[0].lstrip("+-").replace(
                    ".", "", 1
                ).isdigit():
                    continue
            elif not valid_number_parts(group):
                continue
            value = "".join(stripped)
            local = 0.0
            # Case/pallet dimensions are strong alignment anchors in this file.
            anchors = {7: 135.0, 10: 120.0, 13: 125.0}
            if column in anchors and value:
                number = float(value)
                local += 12.0 if number == anchors[column] else -8.0
            if width > 1:
                # Thousands separators occur overwhelmingly in quantities/weights.
                local += 2.0 if column <= 4 else -3.0
            visit(position + width, column + 1, values + [value], score + local)

    visit(0, 0, [], 0.0)
    if not solutions:
        raise ValueError(f"Cannot recover item-master numeric fields: {parts!r}")
    solutions.sort(key=lambda item: item[0], reverse=True)
    return solutions[0][1]


def prepare_warehouse_science() -> None:
    source = DATA / "picks.csv"
    target = DATA / "picks_clean.csv"
    rows_written = 0
    pickers: set[str] = set()
    product_codes: set[str] = set()
    origin_codes: set[str] = set()
    first_timestamp = ""
    last_timestamp = ""
    with source.open(newline="", encoding="utf-8-sig") as src, target.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        if header != PICK_COLUMNS:
            raise ValueError(f"Unexpected picks header: {header!r}")
        writer.writerow(header)
        for line_number, row in enumerate(reader, start=2):
            # The source's second physical row contains type/description labels.
            if line_number == 2 and row[:3] == ["VTFS_JE", "DT_START", "TLV"]:
                continue
            if len(row) < 9:
                raise ValueError(f"Short row {line_number}: {row!r}")
            actual, requested = split_quantities(row[7:])
            writer.writerow(row[:7] + [actual, requested])
            rows_written += 1
            pickers.add(row[0])
            product_codes.add(row[2])
            origin_codes.add(row[3])
            if not first_timestamp:
                first_timestamp = row[1]
            last_timestamp = row[1]
    if rows_written != 2_079_011:
        raise ValueError(f"Expected 2,079,011 pick rows, wrote {rows_written:,}")

    parquet_target = DATA / "picks_clean.parquet"
    if parquet_target.exists():
        parquet_target.unlink()
    column_types = {column: pa.string() for column in PICK_COLUMNS}
    stream = pacsv.open_csv(
        target,
        read_options=pacsv.ReadOptions(block_size=4 << 20),
        convert_options=pacsv.ConvertOptions(column_types=column_types),
    )
    parquet_writer = None
    try:
        for batch in stream:
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(parquet_target, batch.schema)
            parquet_writer.write_batch(batch)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    sku_source = DATA / "SKUs.csv"
    sku_target = DATA / "SKUs_clean.csv"
    with sku_source.open(newline="", encoding="utf-8-sig") as src, sku_target.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader)
        writer.writerow(header)
        for line_number, row in enumerate(reader, start=2):
            if line_number == 2 and row[0] == "TLV":
                continue
            if len(row) < 20:
                raise ValueError(f"Short item-master row {line_number}: {row!r}")
            numeric = split_sku_numeric_fields(row[4:])
            writer.writerow(row[:4] + numeric)

    print(
        "Warehouse Science:",
        f"{rows_written:,} rows, {len(pickers)} pickers,",
        f"{len(product_codes):,} SKUs,",
        f"{len(origin_codes):,} origin codes,",
        f"{first_timestamp} to {last_timestamp}",
    )


def _convert_retail_sheet(source: str, sheet_name: str, target: str) -> None:
    frame = pd.read_excel(source, sheet_name=sheet_name, engine="calamine")
    for column in ["Invoice", "StockCode", "Description", "Country"]:
        frame[column] = frame[column].astype("string")
    for column in ["Quantity", "Price", "Customer ID"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame.to_parquet(target, index=False)


def prepare_retail_ii() -> None:
    source = DATA / "online_retail_II.xlsx"
    target = DATA / "retail2_raw.parquet"
    temporary = [DATA / "retail2_2009.parquet", DATA / "retail2_2010.parquet"]
    sheets = ["Year 2009-2010", "Year 2010-2011"]
    context = mp.get_context("spawn")
    for sheet_name, temporary_path in zip(sheets, temporary):
        process = context.Process(
            target=_convert_retail_sheet,
            args=(str(source), sheet_name, str(temporary_path)),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Retail II conversion failed for {sheet_name}")

    if target.exists():
        target.unlink()
    first = pq.ParquetFile(temporary[0])
    writer = pq.ParquetWriter(target, first.schema_arrow)
    total_rows = 0
    try:
        for temporary_path in temporary:
            parquet = pq.ParquetFile(temporary_path)
            for batch in parquet.iter_batches(batch_size=50_000):
                writer.write_batch(batch)
                total_rows += batch.num_rows
    finally:
        writer.close()
    print(
        "Online Retail II:",
        f"{total_rows:,} rows across {len(sheets)} sheets -> {target.name}",
    )


if __name__ == "__main__":
    prepare_warehouse_science()
    prepare_retail_ii()
