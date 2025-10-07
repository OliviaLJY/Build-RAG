#!/bin/bash
# Startup script for Production RAG API Server

echo "=================================="
echo "Production RAG API Server"
echo "=================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed!"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "❌ FastAPI not installed. Run: pip install fastapi uvicorn"
    exit 1
}

echo "✅ Dependencies OK"
echo ""

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p documents
mkdir -p logs

echo "📁 Directories ready"
echo ""

# Start the server
echo "🚀 Starting Production RAG API Server..."
echo ""
echo "Server will be available at:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run with logging
uvicorn api_server_production:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info \
    2>&1 | tee logs/server_$(date +%Y%m%d_%H%M%S).log

