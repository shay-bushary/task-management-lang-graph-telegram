"""Chat routes for LangGraph execution."""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from langchain_core.messages import HumanMessage

from ..deps import get_graph, get_websocket_manager
from ..graph.graph import CompiledGraph
from ..schemas import ChatResponse, UserMessage
from ..ws import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    user_message: UserMessage,
    graph: CompiledGraph = Depends(get_graph),
    ws_manager: WebSocketManager = Depends(get_websocket_manager),
) -> ChatResponse:
    """Process user message through LangGraph and return session ID for streaming.

    Args:
        user_message: User message data
        graph: Compiled LangGraph instance
        ws_manager: WebSocket manager for streaming

    Returns:
        Chat response with session ID for WebSocket streaming

    Raises:
        HTTPException: If message processing fails
    """
    try:
        # Generate unique session ID
        session_id = str(uuid4())

        logger.info(
            f"Starting chat session {session_id} for message: {user_message.message[:100]}..."
        )

        # Validate message
        if not user_message.message or not user_message.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty",
            )

        # Create initial state with user message
        initial_state = {
            "messages": [HumanMessage(content=user_message.message.strip())]
        }

        # Start async graph execution
        # This will be handled by the WebSocket connection for streaming
        # For now, we just return the session ID

        # Store the initial state and graph execution task in the WebSocket manager
        await ws_manager.prepare_session(session_id, initial_state, graph)

        logger.info(f"Chat session {session_id} prepared for streaming")

        return ChatResponse(
            session_id=session_id,
            message="Chat session started. Connect to WebSocket for streaming results.",
            status="started",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during message processing",
        )


@router.get("/sessions/{session_id}/status")
async def get_session_status(
    session_id: str, ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> dict:
    """Get the status of a chat session.

    Args:
        session_id: Session ID
        ws_manager: WebSocket manager

    Returns:
        Session status information

    Raises:
        HTTPException: If session not found
    """
    try:
        logger.debug(f"Getting status for session: {session_id}")

        status_info = await ws_manager.get_session_status(session_id)

        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        return status_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving session status",
        )


@router.delete("/sessions/{session_id}")
async def cleanup_session(
    session_id: str, ws_manager: WebSocketManager = Depends(get_websocket_manager)
) -> dict:
    """Clean up a chat session.

    Args:
        session_id: Session ID
        ws_manager: WebSocket manager

    Returns:
        Cleanup confirmation
    """
    try:
        logger.info(f"Cleaning up session: {session_id}")

        success = await ws_manager.cleanup_session(session_id)

        return {
            "session_id": session_id,
            "cleaned_up": success,
            "message": "Session cleaned up successfully"
            if success
            else "Session not found",
        }

    except Exception as e:
        logger.error(f"Error cleaning up session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during session cleanup",
        )
