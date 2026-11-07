from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from app.auth import verify_api_key
from app.config import get_settings
from app.retrieval.service import RetrievalService, get_retrieval_service
from app.schemas import (
    HealthResponse, 
    SearchRequest, 
    SearchResponse,
    TranslateRequest,
    TranslateResponse,
    )
from app.translation import TranslationService, get_translation_service

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时先创建检索服务，避免第一个请求等很久。"""
    get_retrieval_service()
    get_translation_service
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="检索术语并按术语翻译的 API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """检查服务是否正常运行。"""
    return HealthResponse(status="ok", version=settings.app_version)

@app.post("/search", 
          response_model=SearchResponse,
          dependencies=[Depends(verify_api_key)])
def search(request: SearchRequest,
           service: RetrievalService = Depends(get_retrieval_service)) -> SearchResponse:
    """输入原文，返回必须使用的术语和参考术语。"""
    return service.search(request.text, request.top_k)

@app.post("/translate",
          response_model=TranslateResponse,
          dependencies=[Depends(verify_api_key)])
def translate(request: TranslateRequest,
           service: TranslationService = Depends(get_translation_service)) -> TranslateResponse:
    return service.translate(request.text, request.top_k)