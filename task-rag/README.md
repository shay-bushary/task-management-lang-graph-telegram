# Task Management RAG Template

A comprehensive task management system with RAG (Retrieval-Augmented Generation) capabilities, LangGraph orchestration, and Telegram bot integration. This template provides a complete solution for managing tasks through natural language interactions while leveraging document knowledge bases.

## 🚀 Features

### Core Capabilities
- **Natural Language Task Management**: Create, update, and manage tasks using conversational AI
- **RAG-Powered Knowledge Base**: Upload PDF documents and query them for contextual information
- **Real-time Streaming**: WebSocket-based token streaming for responsive user experience
- **Telegram Bot Integration**: Complete Telegram bot with message editing for streaming UX
- **RESTful API**: Comprehensive FastAPI backend with full CRUD operations
- **Vector Search**: Chroma-based vector database for semantic document search

### Technical Features
- **LangGraph Orchestration**: Advanced workflow management with state machines
- **Async Architecture**: Full async/await support for high performance
- **Comprehensive Testing**: Complete test suite with mocking for external services
- **Production Ready**: Logging, error handling, and monitoring capabilities
- **Flexible Deployment**: Support for both development and production environments

## 📋 Requirements

- Python 3.11+
- OpenAI API key
- Telegram Bot Token (optional, for bot functionality)

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd task-management-lang-graph-telegram/task-rag
make quickstart
```

This will:
- Install all dependencies
- Create `.env` file from template
- Set up required directories
- Validate configuration

### 2. Configure Environment

Edit the `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Optional (defaults provided)
TELEGRAM_WEBHOOK_SECRET=your_webhook_secret
TELEGRAM_WEBHOOK_URL=https://your-domain.com
MODEL_NAME=gpt-3.5-turbo
EMBEDDINGS_MODEL=text-embedding-3-small
LOG_LEVEL=INFO
```

### 3. Start Development

```bash
# Terminal 1: Start API server
make dev

# Terminal 2: Start Telegram bot
make bot
```

Visit http://localhost:8000/docs for API documentation.

## 📚 Environment Configuration

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM and embeddings | `sk-...` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather | `123456:ABC-DEF...` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TELEGRAM_WEBHOOK_SECRET` | Webhook security token | `None` |
| `TELEGRAM_WEBHOOK_URL` | Webhook URL for production | `None` |
| `MODEL_NAME` | OpenAI model for chat | `gpt-3.5-turbo` |
| `EMBEDDINGS_MODEL` | OpenAI embeddings model | `text-embedding-3-small` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENVIRONMENT` | Environment name | `development` |
| `CHUNK_SIZE` | Text chunk size for RAG | `1000` |
| `CHUNK_OVERLAP` | Text chunk overlap | `200` |
| `RETRIEVAL_K` | Number of documents to retrieve | `5` |

### Directory Configuration

The application uses these directories (created automatically):

- `data/uploads/` - PDF file uploads
- `data/chroma/` - Vector database storage
- `logs/` - Application logs

## 🔧 Development Commands

### Essential Commands

```bash
make help          # Show all available commands
make dev           # Start FastAPI development server
make bot           # Start Telegram bot in polling mode
make test          # Run all tests
make fmt           # Format code with ruff and black
```

### Testing Commands

```bash
make test          # Run all tests
make test-cov      # Run tests with coverage report
make test-watch    # Run tests in watch mode
```

### Code Quality

```bash
make fmt           # Format code
make lint          # Run linting checks
make check         # Run all quality checks (lint + test)
```

### Utilities

```bash
make clean         # Clean temporary files and caches
make check-env     # Validate environment configuration
make logs          # View application logs
make health        # Check application health
make shell         # Start Python shell with app context
```

## 📖 API Documentation

### Core Endpoints

#### Health Check
```http
GET /healthz
```
Returns system health status and service information.

#### Task Management
```http
GET    /tasks/              # List tasks with filtering
POST   /tasks/              # Create new task
GET    /tasks/{id}          # Get specific task
PATCH  /tasks/{id}          # Update task
DELETE /tasks/{id}          # Delete task
GET    /tasks/search/?q=... # Search tasks
GET    /tasks/stats/        # Get task statistics
```

#### Document Ingestion
```http
POST /ingest/pdf           # Upload and process PDF
POST /ingest/tasks         # Bulk create tasks
GET  /ingest/status        # Get ingestion status
```

#### Chat Interface
```http
POST /chat/                           # Start chat session
GET  /chat/sessions/{id}/status       # Get session status
DELETE /chat/sessions/{id}            # Clean up session
```

#### WebSocket Streaming
```
WS /ws/stream?session_id=...
```
Real-time token streaming for chat responses.

### Request/Response Examples

#### Create Task
```bash
curl -X POST "http://localhost:8000/tasks/" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Review quarterly report",
    "description": "Analyze Q3 financial performance"
  }'
```

#### Upload PDF
```bash
curl -X POST "http://localhost:8000/ingest/pdf" \
  -F "file=@document.pdf"
```

#### Chat Request
```bash
curl -X POST "http://localhost:8000/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a task to review the uploaded document"
  }'
```

## 🤖 Telegram Bot Usage

### Bot Commands

- `/start` - Welcome message and feature overview
- `/help` - Detailed help and usage examples
- `/status` - Check bot and API status

### Interactions

#### Text Messages
Send any message to interact with the AI assistant:
- "Create a task to review the quarterly report"
- "What does the document say about project timelines?"
- "Show me my pending tasks"
- "Mark task XYZ as completed"

#### Document Upload
Upload PDF files to add them to the knowledge base:
1. Send a PDF file to the bot
2. Bot will process and index the content
3. You can then ask questions about the document

### Streaming Experience
The bot provides real-time streaming responses:
- Messages update every 500ms as tokens arrive
- Visual indicators show processing status
- Final formatted response when complete

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Telegram Bot  │    │   FastAPI App   │    │   LangGraph     │
│                 │    │                 │    │                 │
│ • Message       │◄──►│ • REST API      │◄──►│ • Orchestration │
│   Handling      │    │ • WebSocket     │    │ • Tool Calling  │
│ • File Upload   │    │ • Streaming     │    │ • State Mgmt    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Services      │    │   Vector Store  │    │   OpenAI API    │
│                 │    │                 │    │                 │
│ • Task Service  │    │ • Chroma DB     │    │ • GPT Models    │
│ • RAG Service   │    │ • Embeddings    │    │ • Embeddings    │
│ • Telegram Svc  │    │ • Search        │    │ • Streaming     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **User Input** → Telegram Bot or API
2. **Message Processing** → FastAPI routes
3. **LangGraph Execution** → Tool calling and orchestration
4. **RAG Retrieval** → Vector search in Chroma
5. **LLM Generation** → OpenAI API with streaming
6. **Response Delivery** → WebSocket streaming to client

### Key Design Patterns

- **Dependency Injection**: Clean service management
- **Repository Pattern**: Abstracted data access
- **Observer Pattern**: WebSocket event broadcasting
- **Strategy Pattern**: Pluggable LLM and vector store backends
- **Factory Pattern**: Application and service initialization

## 🧪 Testing

### Test Structure

```
tests/
├── conftest.py          # Shared fixtures and mocks
├── test_ingest.py       # PDF processing and ingestion
├── test_rag.py          # RAG retrieval and search
└── test_tasks.py        # Task CRUD and LangGraph tools
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
python -m pytest tests/test_tasks.py -v

# Run tests in watch mode
make test-watch
```

### Test Coverage

The test suite covers:
- ✅ All API endpoints with success/error scenarios
- ✅ Service layer with business logic validation
- ✅ LangGraph tools and orchestration
- ✅ PDF processing and RAG functionality
- ✅ Telegram bot interactions
- ✅ WebSocket streaming
- ✅ Error handling and edge cases
- ✅ Thread safety and concurrency

### Mocking Strategy

External services are comprehensively mocked:
- OpenAI API (embeddings, chat completions)
- Chroma vector database
- Telegram Bot API
- WebSocket connections
- File system operations

## 🚀 Deployment

### Development Deployment

```bash
# Start all services
make dev    # Terminal 1
make bot    # Terminal 2
```

### Production Deployment

#### Option 1: Direct Deployment

```bash
# Install production dependencies
make prod-install

# Start production server
make prod-start
```

#### Option 2: Docker Deployment

```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

#### Option 3: Process Manager (PM2)

```bash
# Install PM2
npm install -g pm2

# Start API server
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8000" --name task-api

# Start Telegram bot
pm2 start "python -m bots.telegram_bot" --name telegram-bot
```

### Environment-Specific Configuration

#### Development
- Polling mode for Telegram bot
- Debug logging enabled
- Auto-reload for code changes

#### Production
- Webhook mode for Telegram bot
- Optimized logging levels
- Multiple worker processes
- Error monitoring integration

### Webhook Configuration

For production Telegram bot deployment:

1. Set webhook URL in environment:
```bash
TELEGRAM_WEBHOOK_URL=https://your-domain.com
TELEGRAM_WEBHOOK_SECRET=your-secret-token
```

2. Configure reverse proxy (nginx example):
```nginx
location /webhook {
    proxy_pass http://localhost:8001;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

## 📊 Monitoring and Logging

### Log Files

- `logs/app.log` - General application logs
- `logs/error.log` - Error-specific logs
- Console output - Development logging

### Log Levels

- `DEBUG` - Detailed debugging information
- `INFO` - General operational messages
- `WARNING` - Warning conditions
- `ERROR` - Error conditions
- `CRITICAL` - Critical error conditions

### Health Monitoring

```bash
# Check application health
make health

# View real-time logs
make logs

# Check system status
curl http://localhost:8000/healthz
```

### Performance Monitoring

Key metrics to monitor:
- API response times
- WebSocket connection count
- Vector search latency
- Memory usage
- Error rates

## 🔧 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors

**Problem**: `openai.AuthenticationError: Incorrect API key`

**Solution**:
```bash
# Check API key in .env file
make check-env

# Verify API key format (starts with sk-)
echo $OPENAI_API_KEY
```

#### 2. Telegram Bot Not Responding

**Problem**: Bot doesn't respond to messages

**Solutions**:
```bash
# Check bot token
make check-env

# Verify bot is running
ps aux | grep telegram_bot

# Check logs for errors
make logs

# Test bot token
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
```

#### 3. PDF Processing Fails

**Problem**: PDF upload returns error

**Solutions**:
- Ensure PDF is valid and not corrupted
- Check file size (max 50MB)
- Verify uploads directory permissions
- Check OpenAI embeddings API status

#### 4. Vector Search Returns No Results

**Problem**: RAG queries return empty results

**Solutions**:
```bash
# Check if documents are indexed
curl http://localhost:8000/ingest/status

# Verify Chroma database
ls -la data/chroma/

# Re-upload documents if needed
```

#### 5. WebSocket Connection Issues

**Problem**: Streaming doesn't work

**Solutions**:
- Check if FastAPI server is running
- Verify WebSocket endpoint accessibility
- Check browser console for connection errors
- Ensure session ID is valid

#### 6. High Memory Usage

**Problem**: Application consumes too much memory

**Solutions**:
- Monitor vector database size
- Clear old sessions periodically
- Optimize chunk size settings
- Consider database cleanup

### Debug Mode

Enable debug logging:
```bash
# Set in .env file
LOG_LEVEL=DEBUG

# Restart services
make dev
```

### Performance Optimization

#### Vector Database
```bash
# Optimize chunk size for your documents
CHUNK_SIZE=800
CHUNK_OVERLAP=150

# Adjust retrieval parameters
RETRIEVAL_K=3
```

#### API Performance
- Use connection pooling for external APIs
- Implement caching for frequent queries
- Monitor and optimize slow endpoints

### Getting Help

1. **Check Logs**: Always start with `make logs`
2. **Validate Config**: Run `make check-env`
3. **Test Components**: Use individual test files
4. **Health Check**: Verify with `make health`
5. **Documentation**: Review API docs at `/docs`

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Format code: `make fmt`
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints throughout
- Write comprehensive tests
- Document public APIs
- Use meaningful commit messages

### Testing Requirements

All contributions must include:
- Unit tests for new functionality
- Integration tests for API changes
- Mock external service dependencies
- Maintain test coverage above 90%

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [aiogram](https://aiogram.dev/) - Telegram Bot framework
- [Chroma](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - Language models and embeddings

## 📞 Support

For support and questions:
- Check the troubleshooting section above
- Review the API documentation at `/docs`
- Check application logs with `make logs`
- Validate configuration with `make check-env`

---

**Happy coding! 🚀**

