"""Telegram bot implementation using aiogram v3 with webhook and polling support."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    BotCommand,
    Document,
    Message,
    Update,
    WebhookInfo
)
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.config import Settings
from app.deps import get_settings

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot with aiogram v3 supporting both webhook and polling modes."""
    
    def __init__(self, settings: Settings):
        """Initialize the Telegram bot.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN)
        )
        self.dp = Dispatcher()
        self.fastapi_base_url = f"http://localhost:8000"  # Configurable
        
        # Setup handlers
        self._setup_handlers()
        
        logger.info("Telegram bot initialized")
    
    def _setup_handlers(self):
        """Setup message handlers for the bot."""
        
        # Start command handler
        @self.dp.message(CommandStart())
        async def start_handler(message: Message):
            """Handle /start command."""
            await message.answer(
                "🤖 *Welcome to Task Management RAG Bot!*\n\n"
                "I can help you:\n"
                "• 📝 Manage your tasks with natural language\n"
                "• 📄 Process PDF documents for knowledge retrieval\n"
                "• 💬 Answer questions based on your uploaded documents\n\n"
                "Just send me a message or upload a PDF to get started!\n\n"
                "Commands:\n"
                "/help - Show this help message\n"
                "/status - Check bot status"
            )
        
        # Help command handler
        @self.dp.message(Command("help"))
        async def help_handler(message: Message):
            """Handle /help command."""
            await message.answer(
                "🔧 *Task Management RAG Bot Help*\n\n"
                "*Text Messages:*\n"
                "• Send any message to interact with the AI assistant\n"
                "• Ask questions about your documents\n"
                "• Create, update, or manage tasks naturally\n\n"
                "*Document Upload:*\n"
                "• Upload PDF files to add them to the knowledge base\n"
                "• The bot will process and index the content\n"
                "• You can then ask questions about the uploaded documents\n\n"
                "*Examples:*\n"
                "• \"Create a task to review the quarterly report\"\n"
                "• \"What does the document say about project timelines?\"\n"
                "• \"Show me my pending tasks\"\n"
                "• \"Mark task XYZ as completed\""
            )
        
        # Status command handler
        @self.dp.message(Command("status"))
        async def status_handler(message: Message):
            """Handle /status command."""
            try:
                # Check FastAPI health
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.fastapi_base_url}/healthz") as response:
                        if response.status == 200:
                            health_data = await response.json()
                            status_text = (
                                "✅ *Bot Status: Online*\n\n"
                                f"🔧 API Status: {health_data.get('status', 'unknown').title()}\n"
                                f"📊 Services: {len(health_data.get('services', {}))}\n"
                                f"📁 Directories: {len(health_data.get('directories', {}))}\n"
                                f"🕐 Last Check: {health_data.get('timestamp', 'unknown')}"
                            )
                        else:
                            status_text = "⚠️ *Bot Status: API Unavailable*"
            except Exception as e:
                logger.error(f"Error checking status: {str(e)}")
                status_text = "❌ *Bot Status: Error checking API*"
            
            await message.answer(status_text)
        
        # Document upload handler
        @self.dp.message(F.document)
        async def document_handler(message: Message):
            """Handle document uploads (PDF processing)."""
            document: Document = message.document
            
            # Check if it's a PDF
            if not document.file_name or not document.file_name.lower().endswith('.pdf'):
                await message.answer(
                    "❌ *Only PDF files are supported*\n\n"
                    "Please upload a PDF document to add it to the knowledge base."
                )
                return
            
            # Send processing message
            processing_msg = await message.answer(
                f"📄 *Processing PDF: {document.file_name}*\n\n"
                "⏳ Downloading and analyzing document...\n"
                "This may take a few moments."
            )
            
            try:
                # Download file from Telegram
                file_info = await self.bot.get_file(document.file_id)
                file_path = file_info.file_path
                
                # Download file content
                file_content = await self.bot.download_file(file_path)
                
                # Prepare multipart form data
                form_data = aiohttp.FormData()
                form_data.add_field(
                    'file',
                    file_content,
                    filename=document.file_name,
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
                            await processing_msg.edit_text(
                                f"✅ *PDF Processed Successfully!*\n\n"
                                f"📄 **File:** {result.get('filename', document.file_name)}\n"
                                f"📊 **Pages:** {result.get('document_count', 'unknown')}\n"
                                f"🔤 **Text Chunks:** {result.get('chunk_count', 'unknown')}\n\n"
                                f"💡 You can now ask questions about this document!"
                            )
                        else:
                            error_data = await response.json()
                            await processing_msg.edit_text(
                                f"❌ *Error Processing PDF*\n\n"
                                f"Error: {error_data.get('detail', 'Unknown error')}\n\n"
                                f"Please try uploading the PDF again."
                            )
            
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                await processing_msg.edit_text(
                    f"❌ *Error Processing PDF*\n\n"
                    f"An unexpected error occurred while processing your document.\n"
                    f"Please try again later."
                )
        
        # Text message handler (chat with AI)
        @self.dp.message(F.text)
        async def text_handler(message: Message):
            """Handle text messages (forward to chat endpoint)."""
            user_text = message.text
            
            # Send initial response
            response_msg = await message.answer(
                "🤖 *Processing your message...*\n\n"
                "⏳ Thinking..."
            )
            
            try:
                # Send message to FastAPI chat endpoint
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.fastapi_base_url}/chat/",
                        json={"message": user_text}
                    ) as response:
                        
                        if response.status == 200:
                            chat_result = await response.json()
                            session_id = chat_result.get('session_id')
                            
                            if session_id:
                                # Connect to WebSocket for streaming
                                await self._handle_streaming_response(
                                    session_id, response_msg, user_text
                                )
                            else:
                                await response_msg.edit_text(
                                    "❌ *Error: No session ID received*\n\n"
                                    "Please try your message again."
                                )
                        else:
                            error_data = await response.json()
                            await response_msg.edit_text(
                                f"❌ *Error Processing Message*\n\n"
                                f"Error: {error_data.get('detail', 'Unknown error')}"
                            )
            
            except Exception as e:
                logger.error(f"Error handling text message: {str(e)}")
                await response_msg.edit_text(
                    "❌ *Error Processing Message*\n\n"
                    "An unexpected error occurred. Please try again."
                )
    
    async def _handle_streaming_response(
        self, 
        session_id: str, 
        message: Message, 
        original_text: str
    ):
        """Handle streaming response from WebSocket.
        
        Args:
            session_id: WebSocket session ID
            message: Telegram message to edit
            original_text: Original user message
        """
        try:
            # Connect to WebSocket
            ws_url = f"ws://localhost:8000/ws/stream?session_id={session_id}"
            
            accumulated_response = ""
            last_update_time = 0
            update_interval = 0.5  # Update every 500ms
            
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
                                        try:
                                            await message.edit_text(
                                                f"🤖 *AI Response:*\n\n{accumulated_response}..."
                                            )
                                            last_update_time = current_time
                                        except Exception as edit_error:
                                            # Handle rate limiting or other edit errors
                                            logger.warning(f"Message edit error: {edit_error}")
                                
                                elif msg_type == 'final_result':
                                    # Final response
                                    result = data.get('result', {})
                                    final_content = result.get('content', accumulated_response)
                                    
                                    await message.edit_text(
                                        f"🤖 *AI Response:*\n\n{final_content}"
                                    )
                                    break
                                
                                elif msg_type == 'error':
                                    # Error occurred
                                    error_msg = data.get('error', 'Unknown error')
                                    await message.edit_text(
                                        f"❌ *Error:*\n\n{error_msg}"
                                    )
                                    break
                            
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON from WebSocket: {msg.data}")
                                continue
                        
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
        
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            await message.edit_text(
                "❌ *Error during response streaming*\n\n"
                "The response was interrupted. Please try again."
            )
    
    async def set_webhook(self) -> bool:
        """Set webhook for the bot.
        
        Returns:
            True if webhook was set successfully
        """
        try:
            webhook_url = f"{self.settings.telegram_webhook_url}/webhook"
            
            await self.bot.set_webhook(
                url=webhook_url,
                secret_token=self.settings.telegram_webhook_secret
            )
            
            logger.info(f"Webhook set to: {webhook_url}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting webhook: {str(e)}")
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
        
        except Exception as e:
            logger.error(f"Error deleting webhook: {str(e)}")
            return False
    
    async def get_webhook_info(self) -> Optional[WebhookInfo]:
        """Get current webhook information.
        
        Returns:
            Webhook info or None if error
        """
        try:
            return await self.bot.get_webhook_info()
        except Exception as e:
            logger.error(f"Error getting webhook info: {str(e)}")
            return None
    
    async def set_commands(self):
        """Set bot commands for the menu."""
        commands = [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="help", description="Show help message"),
            BotCommand(command="status", description="Check bot status"),
        ]
        
        await self.bot.set_my_commands(commands)
        logger.info("Bot commands set")
    
    async def start_polling(self):
        """Start the bot in polling mode."""
        try:
            logger.info("Starting bot in polling mode...")
            
            # Delete webhook if exists
            await self.delete_webhook()
            
            # Set commands
            await self.set_commands()
            
            # Start polling
            await self.dp.start_polling(self.bot)
        
        except Exception as e:
            logger.error(f"Error in polling mode: {str(e)}")
            raise
    
    async def start_webhook(self, host: str = "0.0.0.0", port: int = 8001):
        """Start the bot in webhook mode.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            logger.info(f"Starting bot in webhook mode on {host}:{port}...")
            
            # Set webhook
            if not await self.set_webhook():
                raise Exception("Failed to set webhook")
            
            # Set commands
            await self.set_commands()
            
            # Create aiohttp application
            app = web.Application()
            
            # Setup webhook handler
            webhook_requests_handler = SimpleRequestHandler(
                dispatcher=self.dp,
                bot=self.bot,
                secret_token=self.settings.telegram_webhook_secret
            )
            webhook_requests_handler.register(app, path="/webhook")
            
            # Setup application
            setup_application(app, self.dp, bot=self.bot)
            
            # Start server
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, host, port)
            await site.start()
            
            logger.info(f"Webhook server started on {host}:{port}")
            
            # Keep running
            try:
                await asyncio.Future()  # Run forever
            except KeyboardInterrupt:
                logger.info("Shutting down webhook server...")
            finally:
                await runner.cleanup()
        
        except Exception as e:
            logger.error(f"Error in webhook mode: {str(e)}")
            raise


async def main():
    """Main function to run the bot."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Get settings
        settings = get_settings()
        
        # Create bot
        bot = TelegramBot(settings)
        
        # Determine mode (webhook vs polling)
        if settings.telegram_webhook_url:
            logger.info("Starting in webhook mode")
            await bot.start_webhook()
        else:
            logger.info("Starting in polling mode")
            await bot.start_polling()
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

