"""Dependency-light, structure-preserving document parsers for RAG ingestion."""
from __future__ import annotations

import io
import json
import pathlib
import re
import unicodedata
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".docx", ".pdf"}
PARSER_VERSION = "coding-plan-structure-v2"
_WORD_NS_URI = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WORD_NS = f"{{{_WORD_NS_URI}}}"
_WORD_VAL = f"{_WORD_NS}val"
_LIST_PATTERN = re.compile(
    r"^\s*(?:[-*+•]|(?:\d+|[一二三四五六七八九十]+)[.)、．])\s+"
)
_KNOWN_HEADINGS = {
    "摘要", "关键词", "引言", "绪论", "说明", "范围", "步骤", "限制", "结论",
    "结语", "参考文献", "致谢", "abstract", "keywords", "introduction",
    "conclusion", "conclusions", "references",
}


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be converted into text blocks."""


def parse_uploaded_document(filename: str, payload: bytes) -> List[Dict[str, Any]]:
    """Return normalized documents accepted by :class:`KnowledgeBase`."""
    safe_name = pathlib.Path(filename or "unknown").name
    extension = pathlib.Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentParseError(
            f"不支持的文件格式 {extension or '(无扩展名)'}；"
            "支持 txt、md、json、docx、pdf"
        )
    if extension == ".json":
        return _parse_json(safe_name, payload)
    if extension == ".docx":
        text, blocks = _parse_docx(payload)
    elif extension == ".pdf":
        return _parse_pdf(safe_name, payload)
    else:
        text = payload.decode("utf-8", errors="ignore").strip()
        if extension == ".md":
            blocks = _parse_markdown_blocks(text)
        else:
            blocks, _ = _parse_plain_blocks(text)
    return [_document(safe_name, text, blocks)]


def _parse_json(filename: str, payload: bytes) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as ex:
        raise DocumentParseError(f"JSON 解析失败: {ex}") from ex
    if not isinstance(raw, list):
        raise DocumentParseError("JSON 文件应为数组格式: [{title, content}, ...]")

    documents: List[Dict[str, Any]] = []
    allowed = {
        "title", "content", "source_uri", "section", "document_id", "blocks",
        "knowledge_key", "fact_value", "knowledge_version", "version",
        "authority", "effective_at", "expires_at", "reviewed_at",
        "freshness_ttl_days", "deprecated", "supersedes_document_id",
        "scope", "audience",
    }
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise DocumentParseError(f"JSON 第 {index} 项不是对象")
        normalized = {key: item.get(key) for key in allowed if key in item}
        normalized["title"] = str(
            normalized.get("title") or f"{pathlib.Path(filename).stem}-{index}"
        )
        normalized["source_uri"] = str(
            normalized.get("source_uri") or filename
        )
        content = str(normalized.get("content") or "").strip()
        blocks = _normalize_supplied_blocks(normalized.get("blocks"))
        if not content and blocks:
            content = "\n\n".join(str(block["text"]) for block in blocks)
        if not content:
            raise DocumentParseError(f"JSON 第 {index} 项 content 为空")
        if not blocks:
            blocks, _ = _parse_plain_blocks(
                content,
                initial_heading=str(normalized.get("section") or ""),
            )
        normalized["content"] = content
        normalized["blocks"] = _assign_offsets(blocks)
        normalized["parser_version"] = PARSER_VERSION
        documents.append(normalized)
    return documents


def _parse_docx(payload: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as ex:
        raise DocumentParseError(f"DOCX 解析失败: {ex}") from ex

    body = root.find(f".//{_WORD_NS}body")
    if body is None:
        raise DocumentParseError("DOCX 中没有正文结构")

    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    for child in list(body):
        if child.tag == f"{_WORD_NS}p":
            text = _word_text(child)
            if not text:
                continue
            style = _word_paragraph_style(child)
            heading_level = _word_heading_level(style) or _heading_level(text)
            if heading_level is not None:
                heading_stack = _update_heading_stack(
                    heading_stack,
                    heading_level,
                    text,
                )
                block_type = "heading"
            elif child.find(f"./{_WORD_NS}pPr/{_WORD_NS}numPr") is not None:
                block_type = "list"
            elif "code" in style.lower() or "代码" in style:
                block_type = "code"
            else:
                block_type = "paragraph"
            blocks.append(_block(text, block_type, heading_stack))
        elif child.tag == f"{_WORD_NS}tbl":
            rows: List[str] = []
            for row in child.findall(f"./{_WORD_NS}tr"):
                cells = [
                    _word_text(cell)
                    for cell in row.findall(f"./{_WORD_NS}tc")
                ]
                if any(cells):
                    rows.append("| " + " | ".join(
                        cell.strip() for cell in cells
                    ) + " |")
            if rows:
                blocks.append(_block("\n".join(rows), "table", heading_stack))

    blocks = _assign_offsets(blocks)
    text = "\n\n".join(block["text"] for block in blocks).strip()
    if not text:
        raise DocumentParseError("DOCX 中没有可提取文本")
    return text, blocks


def _parse_pdf(filename: str, payload: bytes) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
        from pypdf import PdfReader
    except ImportError as ex:
        raise DocumentParseError(
            "PDF 解析需要安装 requirements.txt 中的 pypdf 和 pdfplumber"
        ) from ex
    try:
        reader = PdfReader(io.BytesIO(payload))
        plumber = pdfplumber.open(io.BytesIO(payload))
        blocks: List[Dict[str, Any]] = []
        heading_stack: List[str] = []
        extracted_pages = 0
        extractors = set()
        try:
            for page_number, page in enumerate(reader.pages, start=1):
                pypdf_text = (page.extract_text() or "").strip()
                text = pypdf_text
                extractor = "pypdf"
                plumber_page = plumber.pages[page_number - 1]
                if _looks_two_column(plumber_page):
                    text, used_columns = _extract_pdfplumber_page(
                        plumber_page,
                        page_number,
                    )
                    extractor = (
                        "pdfplumber-columns" if used_columns else "pdfplumber"
                    )
                elif not text or _is_fragmented_pdf_text(text):
                    text, used_columns = _extract_pdfplumber_page(
                        plumber_page,
                        page_number,
                    )
                    extractor = (
                        "pdfplumber-columns" if used_columns else "pdfplumber"
                    )
                text = _normalize_pdf_text(text)
                if not text:
                    continue
                extractors.add(extractor)
                extracted_pages += 1
                page_blocks, heading_stack = _parse_plain_blocks(
                    text,
                    page_number=page_number,
                    heading_stack=heading_stack,
                )
                blocks.extend(page_blocks)
        finally:
            plumber.close()
    except Exception as ex:
        raise DocumentParseError(f"PDF 解析失败: {ex}") from ex
    if not blocks:
        raise DocumentParseError(
            "PDF 中没有可提取文本；扫描件需要 OCR，当前版本暂不支持"
        )
    blocks = _assign_offsets(blocks)
    content = "\n\n".join(block["text"] for block in blocks)
    document = _document(filename, content, blocks)
    document["page_count"] = len(reader.pages)
    document["text_page_count"] = extracted_pages
    document["pdf_extractors"] = ",".join(sorted(extractors))
    return [document]


def _is_fragmented_pdf_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 80:
        return False
    short_ratio = sum(len(line) <= 10 for line in lines) / len(lines)
    ordered = sorted(len(line) for line in lines)
    median = ordered[len(ordered) // 2]
    return short_ratio >= 0.45 or median <= 8


def _extract_pdfplumber_page(page: Any, page_number: int) -> Tuple[str, bool]:
    use_columns = page_number > 1 and _looks_two_column(page)
    if use_columns:
        midpoint = float(page.width) / 2.0
        left = page.crop((0, 0, midpoint, float(page.height))).extract_text() or ""
        right = page.crop(
            (midpoint, 0, float(page.width), float(page.height))
        ).extract_text() or ""
        combined = "\n\n".join(
            part.strip() for part in (left, right) if part and part.strip()
        )
        if combined:
            return combined, True
    return (page.extract_text() or "").strip(), False


def _looks_two_column(page: Any) -> bool:
    try:
        width = float(page.width)
        height = float(page.height)
        words = [
            word
            for word in page.extract_words()
            if height * 0.08
            < (float(word["top"]) + float(word["bottom"])) / 2
            < height * 0.92
        ]
    except Exception:
        return False
    if len(words) < 80:
        return False
    centers = [
        (float(word["x0"]) + float(word["x1"])) / 2 for word in words
    ]
    left = sum(center < width * 0.45 for center in centers)
    gutter = sum(width * 0.45 <= center <= width * 0.55 for center in centers)
    right = sum(center > width * 0.55 for center in centers)
    if left >= 35 and right >= 35 and gutter / len(words) <= 0.085:
        return True

    band_height = max(80.0, height / 8.0)
    two_column_bands = 0
    for band_start in range(0, int(height), int(band_height)):
        band = [
            word
            for word in words
            if band_start
            <= (float(word["top"]) + float(word["bottom"])) / 2
            < band_start + band_height
        ]
        if len(band) < 20:
            continue
        band_centers = [
            (float(word["x0"]) + float(word["x1"])) / 2 for word in band
        ]
        band_left = sum(center < width * 0.45 for center in band_centers)
        band_gutter = sum(
            width * 0.45 <= center <= width * 0.55 for center in band_centers
        )
        band_right = sum(center > width * 0.55 for center in band_centers)
        if (
            band_left >= 5
            and band_right >= 5
            and band_gutter / len(band) <= 0.10
        ):
            two_column_bands += 1
    return two_column_bands >= 2


def _normalize_pdf_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u200b", "").replace("\ufeff", "")
    normalized_lines: List[str] = []
    for raw in value.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        line = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", line)
        line = re.sub(r"\s+([,.;:!?，。；：！？、])", r"\1", line)
        line = re.sub(r"([（(])\s+", r"\1", line)
        normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def _parse_markdown_blocks(text: str) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    blocks: List[Dict[str, Any]] = []
    heading_stack: List[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            heading_text = heading.group(2).strip()
            heading_stack = _update_heading_stack(
                heading_stack,
                len(heading.group(1)),
                heading_text,
            )
            blocks.append(_block(heading_text, "heading", heading_stack))
            index += 1
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            code_lines = [line]
            index += 1
            while index < len(lines):
                code_lines.append(lines[index])
                if lines[index].strip().startswith(fence):
                    index += 1
                    break
                index += 1
            blocks.append(_block("\n".join(code_lines), "code", heading_stack))
            continue

        if (
            index + 1 < len(lines)
            and "|" in line
            and _is_markdown_separator(lines[index + 1])
        ):
            table_lines = [line, lines[index + 1]]
            index += 2
            while (
                index < len(lines)
                and lines[index].strip()
                and "|" in lines[index]
            ):
                table_lines.append(lines[index])
                index += 1
            blocks.append(_block("\n".join(table_lines), "table", heading_stack))
            continue

        if _LIST_PATTERN.match(line):
            list_lines = [stripped]
            index += 1
            while index < len(lines) and (
                _LIST_PATTERN.match(lines[index])
                or lines[index].startswith(("  ", "\t"))
            ):
                list_lines.append(lines[index].strip())
                index += 1
            blocks.append(_block("\n".join(list_lines), "list", heading_stack))
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if not candidate.strip():
                break
            if (
                re.match(r"^\s{0,3}#{1,6}\s+", candidate)
                or candidate.strip().startswith(("```", "~~~"))
                or _LIST_PATTERN.match(candidate)
            ):
                break
            if (
                index + 1 < len(lines)
                and "|" in candidate
                and _is_markdown_separator(lines[index + 1])
            ):
                break
            paragraph_lines.append(candidate.strip())
            index += 1
        blocks.append(_block(
            _join_prose_lines(paragraph_lines),
            "paragraph",
            heading_stack,
        ))
    return _assign_offsets(blocks)


def _parse_plain_blocks(
    text: str,
    page_number: Optional[int] = None,
    heading_stack: Optional[Sequence[str]] = None,
    initial_heading: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    stack = list(heading_stack or ([initial_heading] if initial_heading else []))
    blocks: List[Dict[str, Any]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if re.fullmatch(r"(?:\d{1,4}|[ivxlcdmIVXLCDM]{1,8})", stripped):
            index += 1
            continue

        heading_level = _heading_level(stripped)
        if heading_level is not None:
            stack = _update_heading_stack(stack, heading_level, stripped)
            blocks.append(_block(stripped, "heading", stack, page_number))
            index += 1
            continue

        if _LIST_PATTERN.match(stripped):
            items = [stripped]
            index += 1
            while index < len(lines) and _LIST_PATTERN.match(lines[index].strip()):
                items.append(lines[index].strip())
                index += 1
            blocks.append(_block("\n".join(items), "list", stack, page_number))
            continue

        if "\t" in stripped and len([
            cell for cell in stripped.split("\t") if cell.strip()
        ]) >= 2:
            rows = [stripped]
            index += 1
            while index < len(lines) and "\t" in lines[index]:
                rows.append(lines[index].strip())
                index += 1
            blocks.append(_block("\n".join(rows), "table", stack, page_number))
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if _heading_level(candidate) is not None or _LIST_PATTERN.match(candidate):
                break
            if "\t" in candidate and len([
                cell for cell in candidate.split("\t") if cell.strip()
            ]) >= 2:
                break
            paragraph_lines.append(candidate)
            index += 1
        blocks.append(_block(
            _join_prose_lines(paragraph_lines),
            "paragraph",
            stack,
            page_number,
        ))
    return blocks, stack


def _heading_level(text: str) -> Optional[int]:
    value = re.sub(r"\s+", " ", text.strip())
    if not value or len(value) > 80:
        return None
    if re.search(
        r"(?:学报|期刊|journal|vol\.?|no\.?|\d{4}年)",
        value,
        flags=re.IGNORECASE,
    ):
        return None
    markdown = re.match(r"^(#{1,6})\s+", value)
    if markdown:
        return len(markdown.group(1))
    normalized = value.strip(" ：:。. ").lower()
    if normalized in _KNOWN_HEADINGS:
        return 1
    if value.endswith(("。", "，", ",", "；", ";", "！", "!", "？", "?")):
        return None
    chapter = re.match(r"^第[一二三四五六七八九十百\d]+([章节篇])", value)
    if chapter:
        return 1 if chapter.group(1) in {"章", "篇"} else 2
    if re.match(r"^[一二三四五六七八九十]+[、.．]\s*\S+", value):
        return 1
    numbered = re.match(r"^(\d+(?:\.\d+){0,3})[\s、．]+\S+", value)
    if numbered:
        number = numbered.group(1)
        remainder = value[numbered.end(1):].lstrip(" 、．")
        if "." not in number and int(number) > 20:
            return None
        if not remainder or remainder[0].isdigit():
            return None
        if re.search(r"[|=∑Σ]", remainder) or any(
            "\ue000" <= char <= "\uf8ff" for char in remainder
        ):
            return None
        if re.search(r"\.{4,}\s*\d*\s*$", remainder):
            return None
        if len(re.findall(r"\d+(?:\.\d+)?", remainder)) >= 2:
            return None
        if not re.search(r"[A-Za-z\u3400-\u9fff]", remainder):
            return None
        return min(4, number.count(".") + 1)
    return None


def _word_text(element: ElementTree.Element) -> str:
    return "".join(
        node.text or "" for node in element.iter(f"{_WORD_NS}t")
    ).strip()


def _word_paragraph_style(paragraph: ElementTree.Element) -> str:
    style = paragraph.find(f"./{_WORD_NS}pPr/{_WORD_NS}pStyle")
    return str(style.attrib.get(_WORD_VAL) or "") if style is not None else ""


def _word_heading_level(style: str) -> Optional[int]:
    match = re.search(r"(?:heading|标题)\s*([1-6])", style, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _block(
    text: str,
    block_type: str,
    heading_stack: Sequence[str],
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "block_type": block_type,
        "heading_path": " > ".join(item for item in heading_stack if item),
        "page_number": page_number,
    }


def _assign_offsets(blocks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    cursor = 0
    for raw in blocks:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        block = dict(raw)
        block["text"] = text
        block["block_index"] = len(normalized)
        block["source_start"] = cursor
        block["source_end"] = cursor + len(text)
        normalized.append(block)
        cursor += len(text) + 2
    return normalized


def _normalize_supplied_blocks(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    blocks: List[Dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict) or not str(raw.get("text") or "").strip():
            continue
        blocks.append({
            "text": str(raw.get("text") or "").strip(),
            "block_type": str(raw.get("block_type") or "paragraph"),
            "heading_path": str(raw.get("heading_path") or ""),
            "page_number": raw.get("page_number"),
        })
    return blocks


def _update_heading_stack(
    stack: Sequence[str],
    level: int,
    text: str,
) -> List[str]:
    safe_level = max(1, min(6, level))
    updated = list(stack[:safe_level - 1])
    while len(updated) < safe_level - 1:
        updated.append("")
    updated.append(text.strip())
    return updated


def _join_prose_lines(lines: Sequence[str]) -> str:
    result = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not result:
            result = line
        elif result.endswith("-") and re.match(r"^[A-Za-z]", line):
            result = result[:-1] + line
        elif _cjk_boundary(result[-1], line[0]):
            result += line
        else:
            result += " " + line
    return result


def _cjk_boundary(left: str, right: str) -> bool:
    return bool(
        re.match(r"[\u3400-\u9fff]", left)
        and re.match(r"[\u3400-\u9fff]", right)
    )


def _is_markdown_separator(line: str) -> bool:
    stripped = line.strip().strip("|")
    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in cells
    )


def _document(
    filename: str,
    text: str,
    blocks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    content = (text or "").strip()
    if not content:
        raise DocumentParseError("文档中没有可导入文本")
    normalized_blocks = _assign_offsets(blocks or [])
    return {
        "title": pathlib.Path(filename).stem,
        "content": content,
        "source_uri": filename,
        "blocks": normalized_blocks,
        "parser_version": PARSER_VERSION,
    }
