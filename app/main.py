"""采访逐字稿校时 API。"""
from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import alignment
from .errors import InvalidTimelineError
from .schemas import AlignRequest, AlignResponse, HealthResponse
from .timeline import build_timeline, validate_request


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_RECOGNIZED_CHARS = _env_int("MAX_RECOGNIZED_CHARS", 20000)
MAX_REVISED_LENGTH = _env_int("MAX_REVISED_LENGTH", 20000)

app = FastAPI(
    title="采访逐字稿校时 API",
    description=(
        "按最少编辑代价将人工修订文本与按时间排序的识别字符全局对齐，"
        "为每个修订字符生成起止毫秒时间轴。"
    ),
    version="1.0.0",
)


@app.exception_handler(InvalidTimelineError)
async def invalid_timeline_handler(_request: Request,
                                   exc: InvalidTimelineError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request,
                             exc: RequestValidationError) -> JSONResponse:
    """把 Pydantic 校验错误转成明确的中文提示。"""
    messages = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"] if p != "body") or "请求体"
        etype = err["type"]
        if etype == "missing":
            messages.append(f"缺少必填字段：{loc}")
        elif etype in ("int_type", "int_parsing"):
            messages.append(f"字段 {loc} 必须是整数毫秒时间戳")
        elif etype == "string_type":
            messages.append(f"字段 {loc} 必须是字符串")
        elif etype == "extra_forbidden":
            messages.append(f"存在不允许的字段：{loc}")
        else:
            messages.append(f"字段 {loc} 无效：{err['msg']}")
    detail = "；".join(messages) if messages else "请求参数无效"
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health() -> HealthResponse:
    """健康检查。"""
    return HealthResponse(status="ok", backend=alignment.backend_name())


@app.post("/align", response_model=AlignResponse, tags=["校时对齐"])
async def align(req: AlignRequest) -> AlignResponse:
    """为人工修订文本生成逐字时间轴。"""
    validate_request(req, MAX_RECOGNIZED_CHARS, MAX_REVISED_LENGTH)
    result = build_timeline(req, alignment.backend_name())
    return AlignResponse(**result)
