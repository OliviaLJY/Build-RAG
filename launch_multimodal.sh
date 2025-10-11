#!/bin/bash

# Launch script for Multimodal RAG System
# This script starts the API server and provides helpful information

echo "🎨 =================================="
echo "   Multimodal RAG System Launcher"
echo "====================================🎨"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual environment found"
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠${NC}  No virtual environment found"
    echo "   Consider creating one: python -m venv venv"
fi

# Check if requirements are installed
echo ""
echo "🔍 Checking dependencies..."

if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} PyTorch installed"
else
    echo -e "${RED}✗${NC} PyTorch not found"
    echo "   Install with: pip install torch torchvision"
    exit 1
fi

if python -c "import transformers" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Transformers installed"
else
    echo -e "${RED}✗${NC} Transformers not found"
    echo "   Install with: pip install transformers"
    exit 1
fi

if python -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} FastAPI installed"
else
    echo -e "${RED}✗${NC} FastAPI not found"
    echo "   Install with: pip install -r requirements.txt"
    exit 1
fi

# Check for OpenAI API key
echo ""
echo "🔑 Checking API configuration..."

if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY" .env; then
        echo -e "${GREEN}✓${NC} OpenAI API key configured"
    else
        echo -e "${YELLOW}⚠${NC}  OpenAI API key not found in .env"
        echo "   Add: OPENAI_API_KEY=sk-your-key-here"
    fi
else
    echo -e "${YELLOW}⚠${NC}  No .env file found"
    echo "   Create one with: OPENAI_API_KEY=sk-your-key-here"
fi

# Check if documents exist
echo ""
echo "📚 Checking document store..."

if [ -d "data/vectorstore" ]; then
    echo -e "${GREEN}✓${NC} Vector store found"
else
    echo -e "${YELLOW}⚠${NC}  No vector store found"
    echo "   Ingest documents first or system will use general knowledge"
fi

# Create necessary directories
echo ""
echo "📁 Setting up directories..."
mkdir -p data logs documents/uploaded

# Check if port 8000 is available
echo ""
echo "🔌 Checking port availability..."

if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}✗${NC} Port 8000 is already in use"
    echo "   Kill existing process or use a different port"
    exit 1
else
    echo -e "${GREEN}✓${NC} Port 8000 is available"
fi

# All checks passed
echo ""
echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo "=================================="
echo "🚀 Starting Multimodal API Server"
echo "=================================="
echo ""

# Display helpful information
echo -e "${BLUE}📖 Quick Reference:${NC}"
echo ""
echo "  API Documentation:  http://localhost:8000/docs"
echo "  Health Check:       http://localhost:8000/health"
echo "  Frontend:           file://$(pwd)/frontend_multimodal.html"
echo ""
echo -e "${BLUE}🔑 Create API Key:${NC}"
echo "  curl -X POST http://localhost:8000/api/keys \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\":\"my-key\",\"user_id\":\"test\"}'"
echo ""
echo -e "${BLUE}💡 Example Query:${NC}"
echo "  curl -X POST http://localhost:8000/api/query \\"
echo "    -H 'X-API-Key: YOUR_KEY' \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\":\"What is machine learning?\"}'"
echo ""
echo "=================================="
echo ""

# Ask user to confirm
read -p "Press Enter to start the server (or Ctrl+C to cancel)..."

# Start the server
echo ""
echo "🎬 Starting server..."
echo ""

# Run the server with auto-reload
python api_server_multimodal.py

# If server stops
echo ""
echo "🛑 Server stopped"

