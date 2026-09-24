"""请求/响应数据模型。"""
from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
)


class RecognizedChar(BaseModel):
    """单个识别字符及其起止时间（毫秒）。"""

    model_config = ConfigDict(extra="forbid")

    char: str = Field(..., description="识别字符，必须且只能包含 1 个字符")
    start: StrictInt = Field(..., description="起始时间（毫秒）")
    end: StrictInt = Field(..., description="结束时间（毫秒）")


class AlignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recognized: list[RecognizedChar] = Field(
        ..., description="按时间排序的识别字符序列"
    )
    total_duration: StrictInt = Field(
        ..., alias="total_duration",
        description="录音总时长（毫秒），须为非负整数",
    )
    revised_text: str = Field(..., description="人工修订后的完整文本")


class RevisedCharTimeline(BaseModel):
    char: str = Field(..., description="修订文本中的字符")
    start: int = Field(..., description="起始时间（毫秒）")
    end: int = Field(..., description="结束时间（毫秒）")
    source: str = Field(
        ...,
        description="matched=直接匹配到识别字符；computed=按空档均匀推算的新增字符",
    )
    matched_index: int | None = Field(
        None, description="source=matched 时对应的识别字符下标（从 0 开始）"
    )
    recognized_char: str | None = Field(
        None, description="source=matched 时对应的识别字符（替换纠错时与 char 不同）"
    )


class AlignResponse(BaseModel):
    total_duration: int
    chars: list[RevisedCharTimeline]
    inserted_count: int = Field(..., description="新增（推算）字符数")
    deleted_count: int = Field(..., description="被删除、未输出的识别字符数")
    matched_count: int = Field(..., description="匹配/替换到识别字符的修订字符数")
    backend: str = Field(..., description="对齐内核：native(C) 或 python")


class LocateRequest(AlignRequest):
    """/locate-quote 请求：沿用 /align 的校时输入，另收引语与编辑次数上限。"""

    quote: str = Field(..., description="要定位的引语，必须为非空字符串")
    max_edits: StrictInt = Field(
        ..., description="允许的最小编辑次数（插入/删除/替换），取值 0～2")

    @field_validator("max_edits")
    @classmethod
    def _check_max_edits(cls, value: int) -> int:
        if value < 0 or value > 2:
            raise ValueError("max_edits 必须为 0～2 的整数")
        return value


class QuoteHit(BaseModel):
    start: int = Field(..., description="命中片段在修订文本中的起始字符下标（含）")
    end: int = Field(..., description="命中片段在修订文本中的结束字符下标（不含）")
    text: str = Field(..., description="命中片段的实际文字")
    edits: int = Field(..., description="实际编辑次数")
    start_ms: int = Field(..., description="由首字符时间轴确定的起始毫秒")
    end_ms: int = Field(..., description="由末字符时间轴确定的结束毫秒")


class LocateResponse(BaseModel):
    hits: list[QuoteHit]


class HealthResponse(BaseModel):
    status: str
    backend: str
