"""WebSocket implementation for token streaming and real-time communication."""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect, Query
from langchain_core.messages import BaseMessage

from .graph.graph import CompiledGraph

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manager for WebSocket connections and streaming sessions."""
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        logger.info("WebSocket manager initialized")
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection and register it.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID for the connection
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        """Disconnect and clean up a WebSocket connection.
        
        Args:
            session_id: Session ID to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        if session_id in self.sessions:
            # Cancel any running tasks
            session_data = self.sessions[session_id]
            if 'task' in session_data and not session_data['task'].done():
                session_data['task'].cancel()
            del self.sessions[session_id]
        
        logger.info(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific WebSocket connection.
        
        Args:
            session_id: Session ID
            message: Message to send
        """
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to session {session_id}: {str(e)}")
                self.disconnect(session_id)
    
    async def broadcast_token(self, session_id: str, token: str):
        """Broadcast a token to the WebSocket connection.
        
        Args:
            session_id: Session ID
            token: Token to broadcast
        """
        await self.send_message(session_id, {
            "type": "token",
            "content": token,
            "session_id": session_id
        })
    
    async def broadcast_event(self, session_id: str, event_type: str, data: Any = None):
        """Broadcast an event to the WebSocket connection.
        
        Args:
            session_id: Session ID
            event_type: Type of event
            data: Optional event data
        """
        message = {
            "type": "event",
            "event_type": event_type,
            "session_id": session_id
        }
        
        if data is not None:
            message["data"] = data
        
        await self.send_message(session_id, message)
    
    async def broadcast_final_result(self, session_id: str, result: Any):
        """Broadcast the final result to the WebSocket connection.
        
        Args:
            session_id: Session ID
            result: Final result
        """
        await self.send_message(session_id, {
            "type": "final_result",
            "result": result,
            "session_id": session_id
        })
    
    async def broadcast_error(self, session_id: str, error: str):
        """Broadcast an error to the WebSocket connection.
        
        Args:
            session_id: Session ID
            error: Error message
        """
        await self.send_message(session_id, {
            "type": "error",
            "error": error,
            "session_id": session_id
        })
    
    async def prepare_session(self, session_id: str, initial_state: Dict[str, Any], graph: CompiledGraph):
        """Prepare a session for graph execution.
        
        Args:
            session_id: Session ID
            initial_state: Initial state for the graph
            graph: Compiled graph instance
        """
        self.sessions[session_id] = {
            "initial_state": initial_state,
            "graph": graph,
            "status": "prepared",
            "created_at": asyncio.get_event_loop().time()
        }
        logger.info(f"Session {session_id} prepared for execution")
    
    async def execute_graph(self, session_id: str):
        """Execute the graph for a session with streaming.
        
        Args:
            session_id: Session ID
        """
        if session_id not in self.sessions:
            await self.broadcast_error(session_id, "Session not found")
            return
        
        session_data = self.sessions[session_id]
        
        try:
            session_data["status"] = "running"
            await self.broadcast_event(session_id, "execution_started")
            
            graph = session_data["graph"]
            initial_state = session_data["initial_state"]
            
            # Create a streaming callback for tokens
            async def token_callback(token: str):
                await self.broadcast_token(session_id, token)
            
            # Create a streaming callback for events
            async def event_callback(event_type: str, data: Any = None):
                await self.broadcast_event(session_id, event_type, data)
            
            # Set up streaming callbacks in the session
            session_data["token_callback"] = token_callback
            session_data["event_callback"] = event_callback
            
            # Execute the graph with streaming
            final_state = None
            async for state in graph.astream(initial_state):
                # Broadcast intermediate states
                await self.broadcast_event(session_id, "state_update", {
                    "messages": [
                        {
                            "type": msg.__class__.__name__,
                            "content": msg.content if hasattr(msg, 'content') else str(msg)
                        }
                        for msg in state.get("messages", [])
                    ]
                })
                final_state = state
            
            # Broadcast final result
            if final_state:
                final_messages = final_state.get("messages", [])
                if final_messages:
                    last_message = final_messages[-1]
                    await self.broadcast_final_result(session_id, {
                        "type": last_message.__class__.__name__,
                        "content": last_message.content if hasattr(last_message, 'content') else str(last_message)
                    })
            
            session_data["status"] = "completed"
            await self.broadcast_event(session_id, "execution_completed")
            
        except Exception as e:
            logger.error(f"Error executing graph for session {session_id}: {str(e)}")
            session_data["status"] = "error"
            await self.broadcast_error(session_id, str(e))
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session status information or None if not found
        """
        if session_id not in self.sessions:
            return None
        
        session_data = self.sessions[session_id]
        return {
            "session_id": session_id,
            "status": session_data.get("status", "unknown"),
            "created_at": session_data.get("created_at"),
            "connected": session_id in self.active_connections
        }
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if session was cleaned up, False if not found
        """
        if session_id not in self.sessions:
            return False
        
        # Cancel any running tasks
        session_data = self.sessions[session_id]
        if 'task' in session_data and not session_data['task'].done():
            session_data['task'].cancel()
        
        # Remove session data
        del self.sessions[session_id]
        
        # Disconnect WebSocket if still connected
        if session_id in self.active_connections:
            try:
                websocket = self.active_connections[session_id]
                await websocket.close()
            except:
                pass
            del self.active_connections[session_id]
        
        logger.info(f"Session {session_id} cleaned up")
        return True


# Global WebSocket manager instance
_websocket_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance.
    
    Returns:
        WebSocket manager instance
    """
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Query(..., description="Session ID for the WebSocket connection")
):
    """WebSocket endpoint for streaming tokens and events.
    
    Args:
        websocket: WebSocket connection
        session_id: Session ID for the connection
    """
    manager = get_websocket_manager()
    
    try:
        await manager.connect(websocket, session_id)
        
        # Check if session exists and start execution
        session_status = await manager.get_session_status(session_id)
        if not session_status:
            await manager.broadcast_error(session_id, "Session not found")
            return
        
        # Start graph execution in background
        execution_task = asyncio.create_task(manager.execute_graph(session_id))
        
        # Keep connection alive and handle messages
        try:
            while True:
                # Wait for messages from client (for potential interaction)
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    # Handle client messages if needed
                    message = json.loads(data)
                    logger.debug(f"Received message from client: {message}")
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        await manager.send_message(session_id, {"type": "pong"})
                    
                except asyncio.TimeoutError:
                    # No message received, continue
                    continue
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received from session {session_id}")
                    continue
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session: {session_id}")
        
        # Wait for execution to complete or cancel it
        if not execution_task.done():
            execution_task.cancel()
            try:
                await execution_task
            except asyncio.CancelledError:
                pass
    
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint for session {session_id}: {str(e)}")
        await manager.broadcast_error(session_id, f"WebSocket error: {str(e)}")
    
    finally:
        manager.disconnect(session_id)

