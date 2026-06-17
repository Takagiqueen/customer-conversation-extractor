"""
命令行入口 —— 从客服对话 JSON 中提取结构化信息。

用法:
    python -m src.extractor.extractor \
        --input data/task2_conversations.json \
        --output outputs/extracted_results.json \
        --mode mock
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="从客服对话 JSON 提取结构化信息",
    )
    parser.add_argument(
        "--input", required=True,
        help="输入 JSON 文件路径，如 data/task2_conversations.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="输出 JSON 文件路径，如 outputs/extracted_results.json",
    )
    parser.add_argument(
        "--mode", default="mock", choices=["mock", "llm"],
        help="运行模式: mock（规则抽取）| llm（LLM 抽取，未实现）",
    )
    args = parser.parse_args(argv)

    # 读取输入
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件不存在 — {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        print("错误：输入 JSON 应为对话数组", file=sys.stderr)
        sys.exit(1)

    # 选择提取器
    if args.mode == "mock":
        from .mock_extractor import MockExtractor
        extractor = MockExtractor()
    else:
        print("错误：LLM 模式尚未实现，请使用 --mode mock", file=sys.stderr)
        sys.exit(1)

    # 批量提取
    result = extractor.extract_batch(conversations)

    # 写入输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(result.model_dump_json(indent=2, ensure_ascii=False))

    print(f"提取完成：{result.success_count}/{result.total_count} 条成功"
          + (f"，{result.failed_count} 条失败" if result.failed_count else ""))
    print(f"结果已写入 {output_path.resolve()}")


if __name__ == "__main__":
    main()
