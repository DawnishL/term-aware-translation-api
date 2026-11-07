from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class HealthResponse(BaseModel):
    status: str = Literal["ok"]
    version: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="需要检索术语的原文",
        examples=["The battery supports fast charging."],
    )

    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="向量检索最多返回的参考术语数量",
    )


class TermHit(BaseModel):
    source: str = Field(..., description="术语原文")
    target: str = Field(..., description="指定译法")
    match_type: Literal["exact", "vector"] = Field(
        ..., description="exact 为精确匹配命中，vector 为向量检索命中"
    )
    score: float | None = Field(None, description="向量相似度，精确匹配时为空")


class SearchResponse(BaseModel):
    query: str
    must_use: list[TermHit] = Field(..., description="必须使用的术语")
    reference: list[TermHit] = Field(..., description="参考术语")

class TranslateRequest(SearchRequest):
    """/translate 接口的请求格式，和 /search 相同。"""


class TranslateResponse(BaseModel):
    """/translate 接口的回复格式。"""

    query: str
    translation: str = Field(..., description="按术语约束生成的译文")
    must_use: list[TermHit] = Field(..., description="要求模型必须使用的术语")
    reference: list[TermHit] = Field(..., description="提供给模型的参考术语")