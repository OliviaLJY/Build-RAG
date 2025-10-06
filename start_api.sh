#!/bin/bash
# Quick start script for the REST API server

echo "🚀 Starting RAG API Server..."
echo ""
echo "📖 API Documentation will be available at:"
echo "   http://localhost:8000/docs"
echo ""
echo "🔗 Interactive API Explorer:"
echo "   http://localhost:8000/redoc"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Start the server with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

