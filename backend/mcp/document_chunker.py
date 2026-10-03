"""Deterministic, structure-aware chunking for Coding Plan documents.

Parsers emit ordered blocks with page/heading coordinates.  This module owns
only boundary selection, short-block refinement and controlled overlap, so it
can be regression-tested without a model or vector database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


_PROSE_TYPES = {"heading", "paragraph"}
_ATOMIC_TYPES = {"table", "code", "list"}
_CONNECTOR_PREFIXES = (
    "因此", "所以", "此外", "同时", "其中", "进一步", "另外", "然而", "但是",
    "该方法", "该模型", "该功能", "该套餐", "上述", "由此", "具体而言", "例如",
    "then", "therefore", "however", "moreover", "furthermore",
)
_INCOMPLETE_SUFFIXES = (
    "，", ",", "：", ":", "、", "（", "(", "以及", "并且", "与", "和",
)


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, "", -1, "-1"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ChunkingConfig:
    """Length and refinement limits measured in normalized characters."""

    target_chars: int = 420
    max_chars: int = 512
    min_chars: int = 120
    overlap_chars: int = 60
    continuity_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError("chunk limits must satisfy 0 < min <= target <= max")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars must be between 0 and max_chars")


@dataclass(frozen=True)
class DocumentBlock:
    """An ordered structural block emitted by a format-specific parser."""

    text: str
    block_type: str = "paragraph"
    heading_path: str = ""
    page_number: Optional[int] = None
    block_index: int = 0
    source_start: int = 0
    source_end: int = 0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        fallback_index: int,
        fallback_heading: str = "",
        fallback_start: int = 0,
    ) -> "DocumentBlock":
        text = str(value.get("text") or "").strip()
        block_type = str(value.get("block_type") or "paragraph").strip().lower()
        if block_type not in {"heading", "paragraph", "list", "table", "code"}:
            block_type = "paragraph"
        source_start = _optional_int(value.get("source_start"))
        if source_start is None:
            source_start = fallback_start
        source_end = _optional_int(value.get("source_end"))
        if source_end is None or source_end < source_start:
            source_end = source_start + len(text)
        block_index = _optional_int(value.get("block_index"))
        return cls(
            text=text,
            block_type=block_type,
            heading_path=str(
                value.get("heading_path") or fallback_heading
            ).strip(),
            page_number=_optional_int(value.get("page_number")),
            block_index=fallback_index if block_index is None else block_index,
            source_start=source_start,
            source_end=source_end,
        )


@dataclass
class DocumentChunk:
    """A retrieval unit plus source coordinates needed for citations."""

    content: str
    heading_path: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    block_start: int = 0
    block_end: int = 0
    source_start: int = 0
    source_end: int = 0
    block_types: List[str] = field(default_factory=list)
    overlap_from_previous: int = 0

    @property
    def chunk_type(self) -> str:
        types = set(self.block_types)
        if "table" in types:
            return "table"
        if "code" in types:
            return "code"
        if "list" in types:
            return "list"
        if types == {"heading"}:
            return "heading"
        return "prose"


class DocumentChunker:
    """Two-stage, structure-first document chunker."""

    VERSION = "coding-plan-structure-v2"

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def split(self, document: Mapping[str, Any]) -> List[DocumentChunk]:
        blocks = self._load_blocks(document)
        if not blocks:
            return []

        units: List[DocumentBlock] = []
        for block in blocks:
            units.extend(self._split_oversized_block(block))

        chunks = self._initial_pack(units)
        chunks = self._merge_short_chunks(chunks)
        return self._add_neighbor_overlap(chunks)

    def split_text(self, text: str) -> List[str]:
        return [item.content for item in self.split({"content": text})]

    def continuity_score(self, left: DocumentChunk, right: DocumentChunk) -> float:
        """Return an explainable score for whether adjacent chunks may merge."""
        if not self._merge_compatible(left, right):
            return 0.0
        score = 0.55
        if self._pages_touch(left, right):
            score += 0.10
        if left.content.rstrip().lower().endswith(_INCOMPLETE_SUFFIXES):
            score += 0.20
        if right.content.lstrip().lower().startswith(_CONNECTOR_PREFIXES):
            score += 0.15
        score += min(
            0.15,
            self._char_bigram_jaccard(left.content, right.content) * 0.30,
        )
        return min(1.0, score)

    def _load_blocks(self, document: Mapping[str, Any]) -> List[DocumentBlock]:
        raw_blocks = document.get("blocks")
        fallback_heading = str(document.get("section") or "").strip()
        blocks: List[DocumentBlock] = []
        cursor = 0
        if isinstance(raw_blocks, Sequence) and not isinstance(raw_blocks, (str, bytes)):
            for index, raw in enumerate(raw_blocks):
                if not isinstance(raw, Mapping):
                    continue
                block = DocumentBlock.from_mapping(
                    raw,
                    index,
                    fallback_heading,
                    cursor,
                )
                if not block.text:
                    continue
                blocks.append(block)
                cursor = max(cursor, block.source_end + 1)
        if blocks:
            return blocks

        text = str(document.get("content") or "").strip()
        if not text:
            return []
        parts = [
            part.strip()
            for part in re.split(r"\n\s*\n+", text)
            if part.strip()
        ] or [text]
        for index, part in enumerate(parts):
            position = text.find(part, cursor)
            if position < 0:
                position = cursor
            blocks.append(DocumentBlock(
                text=part,
                block_type="paragraph",
                heading_path=fallback_heading,
                block_index=index,
                source_start=position,
                source_end=position + len(part),
            ))
            cursor = position + len(part)
        return blocks

    def _split_oversized_block(self, block: DocumentBlock) -> List[DocumentBlock]:
        normal_limit = (
            self.config.max_chars
            if block.block_type in _ATOMIC_TYPES
            else self.config.target_chars
        )
        if len(block.text) <= normal_limit:
            return [block]
        if block.block_type == "table":
            pieces = self._split_table(block.text)
        elif block.block_type == "code":
            pieces = self._split_lines(block.text, preserve_oversized_line=True)
        else:
            pieces = self._split_prose(block.text)
        return self._derive_blocks(block, pieces)

    def _split_prose(self, text: str) -> List[str]:
        sentences = self._split_after(text, r"(?<=[。！？!?；;])\s*|\n+")
        pieces: List[str] = []
        for sentence in sentences:
            if len(sentence) <= self.config.target_chars:
                pieces.append(sentence)
                continue
            clauses = self._split_after(sentence, r"(?<=[，,、：:])\s*")
            for clause in clauses:
                if len(clause) <= self.config.target_chars:
                    pieces.append(clause)
                else:
                    pieces.extend(self._hard_windows(clause))
        return self._pack_text_pieces(pieces)

    def _split_table(self, text: str) -> List[str]:
        rows = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(rows) <= 1:
            return [text]
        header_count = (
            2 if len(rows) > 1 and self._is_markdown_separator(rows[1]) else 1
        )
        header = rows[:header_count]
        groups: List[str] = []
        current = list(header)
        for row in rows[header_count:]:
            candidate = "\n".join(current + [row])
            if len(candidate) > self.config.max_chars and len(current) > header_count:
                groups.append("\n".join(current))
                current = list(header)
            current.append(row)
        if len(current) > header_count or not groups:
            groups.append("\n".join(current))
        return groups

    def _split_lines(self, text: str, preserve_oversized_line: bool) -> List[str]:
        lines = text.splitlines()
        groups: List[str] = []
        current: List[str] = []
        for line in lines:
            candidate = "\n".join(current + [line])
            if current and len(candidate) > self.config.max_chars:
                groups.append("\n".join(current).strip())
                current = []
            if len(line) > self.config.max_chars and not preserve_oversized_line:
                groups.extend(self._hard_windows(line))
            else:
                current.append(line)
        if current:
            groups.append("\n".join(current).strip())
        return [group for group in groups if group]

    def _pack_text_pieces(self, pieces: Iterable[str]) -> List[str]:
        packed: List[str] = []
        current = ""
        for raw in pieces:
            piece = raw.strip()
            if not piece:
                continue
            separator = (
                " "
                if current and current[-1].isascii() and piece[0].isascii()
                else ""
            )
            candidate = f"{current}{separator}{piece}"
            if current and len(candidate) > self.config.target_chars:
                packed.append(current)
                current = piece
            else:
                current = candidate
        if current:
            packed.append(current)
        return packed

    def _hard_windows(self, text: str) -> List[str]:
        size = self.config.target_chars
        return [
            text[index:index + size].strip()
            for index in range(0, len(text), size)
            if text[index:index + size].strip()
        ]

    @staticmethod
    def _split_after(text: str, pattern: str) -> List[str]:
        return [
            part.strip() for part in re.split(pattern, text)
            if part and part.strip()
        ]

    def _derive_blocks(
        self,
        source: DocumentBlock,
        pieces: Sequence[str],
    ) -> List[DocumentBlock]:
        result: List[DocumentBlock] = []
        cursor = 0
        for piece in pieces:
            local_start = source.text.find(piece, cursor)
            if local_start < 0:
                local_start = cursor
            local_end = local_start + len(piece)
            result.append(DocumentBlock(
                text=piece,
                block_type=source.block_type,
                heading_path=source.heading_path,
                page_number=source.page_number,
                block_index=source.block_index,
                source_start=source.source_start + local_start,
                source_end=source.source_start + local_end,
            ))
            cursor = local_end
        return result

    def _initial_pack(self, blocks: Sequence[DocumentBlock]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        current: Optional[DocumentChunk] = None
        for block in blocks:
            if block.block_type in _ATOMIC_TYPES:
                if current is not None:
                    if (
                        current.chunk_type == "heading"
                        and current.heading_path == block.heading_path
                    ):
                        if (
                            len(current.content) + 2 + len(block.text)
                            <= self.config.max_chars
                        ):
                            self._append_block(current, block)
                            chunks.append(current)
                        else:
                            chunks.append(self._chunk_from_block(block))
                        current = None
                        continue
                    chunks.append(current)
                    current = None
                chunks.append(self._chunk_from_block(block))
                continue

            if block.block_type == "heading" and current is not None:
                if current.chunk_type == "heading":
                    self._append_block(current, block)
                    current.heading_path = block.heading_path
                    continue
                chunks.append(current)
                current = None

            if current is None:
                current = self._chunk_from_block(block)
                continue
            if self._can_pack(current, block):
                self._append_block(current, block)
            else:
                chunks.append(current)
                current = self._chunk_from_block(block)
        if current is not None:
            chunks.append(current)
        return chunks

    def _can_pack(self, current: DocumentChunk, block: DocumentBlock) -> bool:
        if current.chunk_type in _ATOMIC_TYPES or block.block_type in _ATOMIC_TYPES:
            return False
        if current.heading_path != block.heading_path:
            return False
        if not self._page_numbers_touch(current.page_end, block.page_number):
            return False
        combined = len(current.content) + 2 + len(block.text)
        if combined > self.config.max_chars:
            return False
        return (
            combined <= self.config.target_chars
            or len(current.content) < self.config.min_chars
        )

    @staticmethod
    def _chunk_from_block(block: DocumentBlock) -> DocumentChunk:
        return DocumentChunk(
            content=block.text,
            heading_path=block.heading_path,
            page_start=block.page_number,
            page_end=block.page_number,
            block_start=block.block_index,
            block_end=block.block_index,
            source_start=block.source_start,
            source_end=block.source_end,
            block_types=[block.block_type],
        )

    @staticmethod
    def _append_block(chunk: DocumentChunk, block: DocumentBlock) -> None:
        chunk.content = f"{chunk.content}\n\n{block.text}"
        chunk.block_end = max(chunk.block_end, block.block_index)
        chunk.source_end = max(chunk.source_end, block.source_end)
        chunk.block_types.append(block.block_type)
        if block.page_number is not None:
            chunk.page_start = (
                block.page_number
                if chunk.page_start is None
                else min(chunk.page_start, block.page_number)
            )
            chunk.page_end = (
                block.page_number
                if chunk.page_end is None
                else max(chunk.page_end, block.page_number)
            )

    def _merge_short_chunks(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> List[DocumentChunk]:
        refined: List[DocumentChunk] = []
        index = 0
        while index < len(chunks):
            current = chunks[index]
            if len(current.content) < self.config.min_chars and index + 1 < len(chunks):
                right = chunks[index + 1]
                if self._can_merge_chunks(current, right):
                    current = self._merge_chunks(current, right)
                    index += 1
            if len(current.content) < self.config.min_chars and refined:
                left = refined[-1]
                if self._can_merge_chunks(left, current):
                    refined[-1] = self._merge_chunks(left, current)
                    index += 1
                    continue
            refined.append(current)
            index += 1
        return refined

    def _can_merge_chunks(self, left: DocumentChunk, right: DocumentChunk) -> bool:
        if len(left.content) + 2 + len(right.content) > self.config.max_chars:
            return False
        return self.continuity_score(left, right) >= self.config.continuity_threshold

    def _merge_compatible(self, left: DocumentChunk, right: DocumentChunk) -> bool:
        if (
            left.chunk_type not in {"prose", "heading"}
            or right.chunk_type not in {"prose", "heading"}
        ):
            return False
        if left.heading_path != right.heading_path:
            return False
        return self._page_numbers_touch(left.page_end, right.page_start)

    @staticmethod
    def _merge_chunks(left: DocumentChunk, right: DocumentChunk) -> DocumentChunk:
        pages = [
            page
            for page in (
                left.page_start,
                left.page_end,
                right.page_start,
                right.page_end,
            )
            if page is not None
        ]
        return DocumentChunk(
            content=f"{left.content}\n\n{right.content}",
            heading_path=left.heading_path,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            block_start=min(left.block_start, right.block_start),
            block_end=max(left.block_end, right.block_end),
            source_start=min(left.source_start, right.source_start),
            source_end=max(left.source_end, right.source_end),
            block_types=left.block_types + right.block_types,
        )

    def _add_neighbor_overlap(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> List[DocumentChunk]:
        if self.config.overlap_chars <= 0:
            return list(chunks)
        result: List[DocumentChunk] = []
        for current in chunks:
            if result:
                previous = result[-1]
                available = self.config.max_chars - len(current.content) - 1
                if available > 0 and self._merge_compatible(previous, current):
                    overlap = self._overlap_tail(
                        previous.content,
                        min(self.config.overlap_chars, available),
                    )
                    if overlap:
                        current = DocumentChunk(
                            content=f"{overlap}\n{current.content}",
                            heading_path=current.heading_path,
                            page_start=current.page_start,
                            page_end=current.page_end,
                            block_start=current.block_start,
                            block_end=current.block_end,
                            source_start=max(0, current.source_start - len(overlap)),
                            source_end=current.source_end,
                            block_types=list(current.block_types),
                            overlap_from_previous=len(overlap),
                        )
            result.append(current)
        return result

    @staticmethod
    def _overlap_tail(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        candidate = text[-max_chars:].strip()
        if not candidate:
            return ""
        boundary = re.search(r"[。！？!?；;\n]", candidate)
        if boundary:
            suffix = candidate[boundary.end():].strip()
            if len(suffix) >= min(20, max_chars // 2):
                candidate = suffix
        return candidate

    @staticmethod
    def _page_numbers_touch(left: Optional[int], right: Optional[int]) -> bool:
        if left is None or right is None:
            return True
        return 0 <= right - left <= 1

    @staticmethod
    def _pages_touch(left: DocumentChunk, right: DocumentChunk) -> bool:
        return DocumentChunker._page_numbers_touch(left.page_end, right.page_start)

    @staticmethod
    def _char_bigram_jaccard(left: str, right: str) -> float:
        def grams(value: str) -> set:
            normalized = re.sub(r"\s+", "", value.lower())
            return {
                normalized[index:index + 2]
                for index in range(max(0, len(normalized) - 1))
            }

        left_grams = grams(left)
        right_grams = grams(right)
        if not left_grams or not right_grams:
            return 0.0
        return len(left_grams & right_grams) / len(left_grams | right_grams)

    @staticmethod
    def _is_markdown_separator(line: str) -> bool:
        stripped = line.strip().strip("|")
        cells = [cell.strip() for cell in stripped.split("|")]
        return bool(cells) and all(
            re.fullmatch(r":?-{3,}:?", cell) for cell in cells
        )
