#!/usr/bin/env python3
"""Copy files whose names contain sample IDs listed in an Excel file."""

import argparse
import os
import shutil
from pathlib import Path

from openpyxl import load_workbook
from tqdm import tqdm


SAMPLE_COLUMN = "样本编号"


def read_sample_ids(excel_path, column_name=SAMPLE_COLUMN):
    """Read non-empty sample IDs from the specified Excel column."""
    excel_file = Path(excel_path)
    if not excel_file.is_file():
        raise FileNotFoundError("Excel 文件不存在: {}".format(excel_file))

    workbook = load_workbook(str(excel_file), read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        rows = worksheet.iter_rows(values_only=True)
        try:
            header = next(rows)
        except StopIteration:
            raise ValueError("Excel 文件为空: {}".format(excel_file))

        try:
            column_index = list(header).index(column_name)
        except ValueError:
            columns = ", ".join(str(name) for name in header if name is not None)
            raise ValueError(
                "Excel 中未找到列 '{}'; 当前列为: {}".format(column_name, columns)
            )

        sample_ids = []
        seen = set()
        for row in rows:
            value = row[column_index] if column_index < len(row) else None
            if value is None:
                continue
            sample_id = str(value).strip()
            if sample_id and sample_id not in seen:
                sample_ids.append(sample_id)
                seen.add(sample_id)
    finally:
        workbook.close()

    if not sample_ids:
        raise ValueError("Excel 列 '{}' 中没有有效样本编号".format(column_name))

    return sample_ids


def build_destination_path(destination_dir, filename):
    """Return a destination path, adding a suffix if a file already exists."""
    destination = destination_dir / filename
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = destination_dir / "{}_{}{}".format(stem, counter, suffix)
        if not candidate.exists():
            return candidate
        counter += 1


def copy_matched_files(excel_path, search_path, copy_path, column_name=SAMPLE_COLUMN):
    """Copy files whose file names contain any sample ID from the Excel file."""
    sample_ids = read_sample_ids(excel_path, column_name=column_name)

    search_dir = Path(search_path)
    if not search_dir.is_dir():
        raise NotADirectoryError("查找文件路径不是有效目录: {}".format(search_dir))

    destination_dir = Path(copy_path)
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for current_dir, _, filenames in tqdm(
        os.walk(str(search_dir)), desc="查找文件", unit="目录"
    ):
        current_path = Path(current_dir)
        for filename in filenames:
            if any(sample_id in filename for sample_id in sample_ids):
                source = current_path / filename
                destination = build_destination_path(destination_dir, filename)
                shutil.copy2(str(source), str(destination))
                copied_files.append((source, destination))

    return copied_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 Excel 的样本编号列递归查找并复制匹配文件"
    )
    parser.add_argument("excel_path", help="Excel 文件路径")
    parser.add_argument("search_path", help="需要递归查找的文件夹路径")
    parser.add_argument("copy_path", help="匹配文件复制到的文件夹路径")
    parser.add_argument(
        "--column",
        default=SAMPLE_COLUMN,
        help="Excel 中的样本编号列名，默认: {}".format(SAMPLE_COLUMN),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    copied_files = copy_matched_files(
        args.excel_path,
        args.search_path,
        args.copy_path,
        column_name=args.column,
    )
    print("完成，已复制 {} 个文件到 {}".format(len(copied_files), args.copy_path))


if __name__ == "__main__":
    main()
