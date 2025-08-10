"""Telegram service for message handling and webhook management."""

import asyncio
import json
import logging
from typing import Dict, Optional, Any

import aiohttp
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramRetryAfter
from aiogram.types import Message, WebhookInfo

from ..config import Settings

logger = logging.getLogger(__name__)


class TelegramService:
    """Service for Telegram bot operations and message management."""
    
    def __init__(self, settings: Settings):
        """Initialize the Telegram service.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
        )
        self.fastapi_base_url = "http://localhost:8000"  # Configurable
        
        logger.info("Telegram service initialized")
    
    async def send_message(
        self, 
        chat_id: int, 
        text: str, 
        parse_mode: Optional[str] = None,
        disable_web_page_preview: bool = True
    ) -> Optional[Message]:
        """Send a message to a Telegram chat.
        
        Args:
            chat_id: Telegram chat ID
            text: Message text
            parse_mode: Parse mode (Markdown, HTML, or None)
            disable_web_page_preview: Whether to disable web page preview
            
        Returns:
            Sent message or None if failed
        """
        try:
            message = await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
            
            logger.debug(f"Message sent to chat {chat_id}: {text[:50]}...")
            return message
        
        except TelegramRetryAfter as e:
            logger.warning(f"Rate limited, retry after {e.retry_after} seconds")
            await asyncio.sleep(e.retry_after)
            return await self.send_message(chat_id, text, parse_mode, disable_web_page_preview)
        
        except TelegramAPIError as e:
            logger.error(f"Telegram API error sending message: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error sending message: {str(e)}")
            return None
    
    async def edit_message(
        self, 
        chat_id: int, 
        message_id: int, 
        text: str,
        parse_mode: Optional[str] = None,
        disable_web_page_preview: bool = True
    ) -> bool:
        """Edit a Telegram message.
        
        Args:
            chat_id: Telegram chat ID
            message_id: Message ID to edit
            text: New message text
            parse_mode: Parse mode (Markdown, HTML, or None)
            disable_web_page_preview: Whether to disable web page preview
            
        Returns:
            True if message was edited successfully
        """
        try:
            await self.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=disable_web_page_preview
            )
            
            logger.debug(f"Message {message_id} edited in chat {chat_id}")
            return True
        
        except TelegramRetryAfter as e:
            logger.warning(f"Rate limited, retry after {e.retry_after} seconds")
            await asyncio.sleep(e.retry_after)
            return await self.edit_message(chat_id, message_id, text, parse_mode, disable_web_page_preview)
        
        except TelegramAPIError as e:
            # Handle common edit errors
            if "message is not modified" in str(e).lower():
                logger.debug(f"Message {message_id} content unchanged, skipping edit")
                return True
            elif "message to edit not found" in str(e).lower():
                logger.warning(f"Message {message_id} not found for editing")
                return False
            else:
                logger.error(f"Telegram API error editing message: {str(e)}")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error editing message: {str(e)}")
            return False
    
    async def delete_message(self, chat_id: int, message_id: int) -> bool:
        """Delete a Telegram message.
        
        Args:
            chat_id: Telegram chat ID
            message_id: Message ID to delete
            
        Returns:
            True if message was deleted successfully
        """
        try:
            await self.bot.delete_message(chat_id=chat_id, message_id=message_id)
            
            logger.debug(f"Message {message_id} deleted from chat {chat_id}")
            return True
        
        except TelegramAPIError as e:
            logger.error(f"Telegram API error deleting message: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting message: {str(e)}")
            return False
    
    async def set_webhook(self, webhook_url: str, secret_token: Optional[str] = None) -> bool:
        """Set webhook for the bot.
        
        Args:
            webhook_url: Webhook URL
            secret_token: Optional secret token for webhook security
            
        Returns:
            True if webhook was set successfully
        """
        try:
            await self.bot.set_webhook(
                url=webhook_url,
                secret_token=secret_token or self.settings.telegram_webhook_secret
            )
            
            logger.info(f"Webhook set to: {webhook_url}")
            return True
        
        except TelegramAPIError as e:
            logger.error(f"Telegram API error setting webhook: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error setting webhook: {str(e)}")
            return False
    
    async def delete_webhook(self) -> bool:
        """Delete webhook for the bot.
        
        Returns:
            True if webhook was deleted successfully
        """
        try:
            await self.bot.delete_webhook()
            logger.info("Webhook deleted")
            return True
        
        except TelegramAPIError as e:
            logger.error(f"Telegram API error deleting webhook: {str(e)}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error deleting webhook: {str(e)}")
            return False
    
    async def get_webhook_info(self) -> Optional[WebhookInfo]:
        """Get current webhook information.
        
        Returns:
            Webhook info or None if error
        """
        try:
            webhook_info = await self.bot.get_webhook_info()
            logger.debug(f"Webhook info: {webhook_info}")
            return webhook_info
        
        except TelegramAPIError as e:
            logger.error(f"Telegram API error getting webhook info: {str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Unexpected error getting webhook info: {str(e)}")
            return None
    
    async def handle_streaming_chat(
        self, 
        chat_id: int, 
        user_message: str,
        initial_message_id: Optional[int] = None
    ) -> Optional[Message]:
        """Handle streaming chat response with real-time message editing.
        
        Args:
            chat_id: Telegram chat ID
            user_message: User's message text
            initial_message_id: Optional initial message ID to edit
            
        Returns:
            Final message or None if failed
        """
        try:
            # Send initial processing message if no message ID provided
            if initial_message_id is None:
                initial_msg = await self.send_message(
                    chat_id=chat_id,
                    text="🤖 *Processing your message...*\n\n⏳ Thinking..."
                )
                if not initial_msg:
                    return None
                initial_message_id = initial_msg.message_id
            
            # Start chat session with FastAPI
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.fastapi_base_url}/chat/",
                    json={"message": user_message}
                ) as response:
                    
                    if response.status != 200:
                        error_data = await response.json()
                        await self.edit_message(
                            chat_id=chat_id,
                            message_id=initial_message_id,
                            text=f"❌ *Error Processing Message*\n\n{error_data.get('detail', 'Unknown error')}"
                        )
                        return None
                    
                    chat_result = await response.json()
                    session_id = chat_result.get('session_id')
                    
                    if not session_id:
                        await self.edit_message(
                            chat_id=chat_id,
                            message_id=initial_message_id,
                            text="❌ *Error: No session ID received*\n\nPlease try your message again."
                        )
                        return None
            
            # Connect to WebSocket for streaming
            return await self._handle_websocket_streaming(
                chat_id=chat_id,
                message_id=initial_message_id,
                session_id=session_id
            )
        
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            if initial_message_id:
                await self.edit_message(
                    chat_id=chat_id,
                    message_id=initial_message_id,
                    text="❌ *Error Processing Message*\n\nAn unexpected error occurred. Please try again."
                )
            return None
    
    async def _handle_websocket_streaming(
        self, 
        chat_id: int, 
        message_id: int, 
        session_id: str
    ) -> Optional[Message]:
        """Handle WebSocket streaming with message editing.
        
        Args:
            chat_id: Telegram chat ID
            message_id: Message ID to edit
            session_id: WebSocket session ID
            
        Returns:
            Final message or None if failed
        """
        try:
            ws_url = f"ws://localhost:8000/ws/stream?session_id={session_id}"
            
            accumulated_response = ""
            last_update_time = 0
            update_interval = 0.5  # Update every 500ms
            max_message_length = 4000  # Telegram message limit with some buffer
            
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(ws_url) as ws:
                    
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                                msg_type = data.get('type')
                                
                                if msg_type == 'token':
                                    # Accumulate tokens
                                    token = data.get('content', '')
                                    accumulated_response += token
                                    
                                    # Update message every 500ms
                                    current_time = asyncio.get_event_loop().time()
                                    if current_time - last_update_time >= update_interval:
                                        # Truncate if too long
                                        display_text = accumulated_response
                                        if len(display_text) > max_message_length:
                                            display_text = display_text[:max_message_length - 50] + "..."
                                        
                                        success = await self.edit_message(
                                            chat_id=chat_id,
                                            message_id=message_id,
                                            text=f"🤖 *AI Response:*\n\n{display_text}..."
                                        )
                                        
                                        if success:
                                            last_update_time = current_time
                                
                                elif msg_type == 'final_result':
                                    # Final response
                                    result = data.get('result', {})
                                    final_content = result.get('content', accumulated_response)
                                    
                                    # Truncate if too long
                                    if len(final_content) > max_message_length:
                                        final_content = final_content[:max_message_length - 50] + "..."
                                    
                                    await self.edit_message(
                                        chat_id=chat_id,
                                        message_id=message_id,
                                        text=f"🤖 *AI Response:*\n\n{final_content}"
                                    )
                                    break
                                
                                elif msg_type == 'error':
                                    # Error occurred
                                    error_msg = data.get('error', 'Unknown error')
                                    await self.edit_message(
                                        chat_id=chat_id,
                                        message_id=message_id,
                                        text=f"❌ *Error:*\n\n{error_msg}"
                                    )
                                    break
                                
                                elif msg_type == 'event':
                                    # Handle events (optional visual feedback)
                                    event_type = data.get('event_type')
                                    if event_type == 'execution_started':
                                        await self.edit_message(
                                            chat_id=chat_id,
                                            message_id=message_id,
                                            text="🤖 *AI Response:*\n\n🔄 Processing your request..."
                                        )
                            
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from WebSocket: {msg.data}")
                                continue
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
            
            return None  # Message was edited in place
        
        except Exception as e:
            logger.error(f"Error in WebSocket streaming: {str(e)}")
            await self.edit_message(
                chat_id=chat_id,
                message_id=message_id,
                text="❌ *Error during response streaming*\n\nThe response was interrupted. Please try again."
            )
            return None
    
    async def handle_document_upload(
        self, 
        chat_id: int, 
        file_id: str, 
        filename: str
    ) -> Optional[Message]:
        """Handle document upload and processing.
        
        Args:
            chat_id: Telegram chat ID
            file_id: Telegram file ID
            filename: Original filename
            
        Returns:
            Response message or None if failed
        """
        try:
            # Validate file type
            if not filename.lower().endswith('.pdf'):
                return await self.send_message(
                    chat_id=chat_id,
                    text="❌ *Only PDF files are supported*\n\nPlease upload a PDF document to add it to the knowledge base."
                )
            
            # Send processing message
            processing_msg = await self.send_message(
                chat_id=chat_id,
                text=f"📄 *Processing PDF: {filename}*\n\n⏳ Downloading and analyzing document...\nThis may take a few moments."
            )
            
            if not processing_msg:
                return None
            
            # Download file from Telegram
            file_info = await self.bot.get_file(file_id)
            file_content = await self.bot.download_file(file_info.file_path)
            
            # Prepare multipart form data
            form_data = aiohttp.FormData()
            form_data.add_field(
                'file',
                file_content,
                filename=filename,
                content_type='application/pdf'
            )
            
            # Send to FastAPI ingestion endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.fastapi_base_url}/ingest/pdf",
                    data=form_data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Update message with success
                        await self.edit_message(
                            chat_id=chat_id,
                            message_id=processing_msg.message_id,
                            text=(
                                f"✅ *PDF Processed Successfully!*\n\n"
                                f"📄 **File:** {result.get('filename', filename)}\n"
                                f"📊 **Pages:** {result.get('document_count', 'unknown')}\n"
                                f"🔤 **Text Chunks:** {result.get('chunk_count', 'unknown')}\n\n"
                                f"💡 You can now ask questions about this document!"
                            )
                        )
                        return processing_msg
                    else:
                        error_data = await response.json()
                        await self.edit_message(
                            chat_id=chat_id,
                            message_id=processing_msg.message_id,
                            text=(
                                f"❌ *Error Processing PDF*\n\n"
                                f"Error: {error_data.get('detail', 'Unknown error')}\n\n"
                                f"Please try uploading the PDF again."
                            )
                        )
                        return processing_msg
        
        except Exception as e:
            logger.error(f"Error handling document upload: {str(e)}")
            if 'processing_msg' in locals():
                await self.edit_message(
                    chat_id=chat_id,
                    message_id=processing_msg.message_id,
                    text=(
                        f"❌ *Error Processing PDF*\n\n"
                        f"An unexpected error occurred while processing your document.\n"
                        f"Please try again later."
                    )
                )
                return processing_msg
            else:
                return await self.send_message(
                    chat_id=chat_id,
                    text="❌ *Error Processing PDF*\n\nAn unexpected error occurred. Please try again."
                )
    
    async def get_bot_info(self) -> Optional[Dict[str, Any]]:
        """Get bot information.
        
        Returns:
            Bot information or None if failed
        """
        try:
            me = await self.bot.get_me()
            return {
                "id": me.id,
                "username": me.username,
                "first_name": me.first_name,
                "is_bot": me.is_bot,
                "can_join_groups": me.can_join_groups,
                "can_read_all_group_messages": me.can_read_all_group_messages,
                "supports_inline_queries": me.supports_inline_queries
            }
        
        except Exception as e:
            logger.error(f"Error getting bot info: {str(e)}")
            return None
    
    async def close(self):
        """Close the bot session."""
        try:
            await self.bot.session.close()
            logger.info("Telegram service closed")
        except Exception as e:
            logger.error(f"Error closing Telegram service: {str(e)}")


# Global Telegram service instance - will be initialized during app startup
_telegram_service: Optional[TelegramService] = None


def get_telegram_service() -> Optional[TelegramService]:
    """Get the global Telegram service instance.
    
    Returns:
        Telegram service instance or None if not initialized
    """
    return _telegram_service


def initialize_telegram_service(settings: Settings) -> TelegramService:
    """Initialize the global Telegram service instance.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized Telegram service
    """
    global _telegram_service
    _telegram_service = TelegramService(settings)
    return _telegram_service

