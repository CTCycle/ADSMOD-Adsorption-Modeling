# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
import threading
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from server.domain.deep_learning import (
    DeepLearningAction,
    DeepLearningLogResponse,
    DeepLearningOperationAccepted,
    DeepLearningOperationRequest,
    DeepLearningPlanResponse,
    DeepLearningStatusResponse,
)
from server.services.deep_learning import (
    SHUTDOWN_DELAY_SECONDS,
    DeepLearningManager,
    DeepLearningOperationError,
)


###############################################################################
def require_deep_learning(request: Request) -> None:
    """Router dependency: training endpoints only run on a verified installation."""
    manager: DeepLearningManager = request.app.state.deep_learning
    available, reason = manager.availability()
    if not available:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Deep learning is unavailable: {(reason or 'not installed').rstrip('.')}. "
            "Install or repair it in Settings > Deep Learning.",
        )


###############################################################################
def _manager(request: Request) -> DeepLearningManager:
    return request.app.state.deep_learning


###############################################################################
async def get_status(
    request: Request, refresh: Annotated[bool, Query()] = False
) -> DeepLearningStatusResponse:
    manager = _manager(request)
    if refresh:
        await asyncio.to_thread(manager.refresh)
    return await asyncio.to_thread(manager.status)


###############################################################################
async def get_plan(
    request: Request, action: Annotated[DeepLearningAction, Query()]
) -> DeepLearningPlanResponse:
    return await asyncio.to_thread(_manager(request).plan, action)


###############################################################################
async def start_operation(
    request: Request, payload: DeepLearningOperationRequest
) -> DeepLearningOperationAccepted:
    if not payload.confirm:
        raise HTTPException(
            status_code=422,
            detail="Explicit confirmation is required: ADSMOD closes while dependencies change.",
        )
    request_shutdown = getattr(request.app.state, "request_shutdown", None)
    if request_shutdown is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This ADSMOD process cannot restart itself; start it with the launcher or 'python -m server.cli'.",
        )
    try:
        operation_id = await asyncio.to_thread(_manager(request).begin, payload.action)
    except DeepLearningOperationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    # Let this response reach the client before the server begins closing.
    timer = threading.Timer(SHUTDOWN_DELAY_SECONDS, request_shutdown)
    timer.daemon = True
    timer.start()
    return DeepLearningOperationAccepted(operation_id=operation_id, action=payload.action)


###############################################################################
async def get_log(
    request: Request, lines: Annotated[int, Query(ge=1, le=500)] = 80
) -> DeepLearningLogResponse:
    return DeepLearningLogResponse(lines=_manager(request).log_tail(lines))


###############################################################################
def create_deep_learning_router() -> APIRouter:
    router = APIRouter(prefix="/deep-learning", tags=["deep-learning"])
    router.add_api_route("/status", get_status, methods=["GET"], response_model=DeepLearningStatusResponse)
    router.add_api_route("/plan", get_plan, methods=["GET"], response_model=DeepLearningPlanResponse)
    router.add_api_route(
        "/operations",
        start_operation,
        methods=["POST"],
        response_model=DeepLearningOperationAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    router.add_api_route("/log", get_log, methods=["GET"], response_model=DeepLearningLogResponse)
    return router


__all__ = ["create_deep_learning_router", "require_deep_learning"]
