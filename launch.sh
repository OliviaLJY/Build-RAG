#!/bin/bash
# Launch script for RAG System with Frontend

echo "🚀 Launching RAG System..."
echo ""

# Check if server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "✅ Backend server already running on port 8000"
else
    echo "🔧 Starting backend server..."
    uvicorn api_server_production:app --reload --port 8000 > logs/server.log 2>&1 &
    SERVER_PID=$!
    echo "✅ Backend started (PID: $SERVER_PID)"
    sleep 3
fi

echo ""
echo "🌐 Opening frontend in browser..."
echo ""

# Open frontend in default browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open frontend.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open frontend.html
else
    # Windows Git Bash
    start frontend.html
fi

echo "✅ Frontend opened!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RAG System is Ready!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Frontend: file://$(pwd)/frontend.html"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Keep script running
tail -f logs/server.log 2>/dev/null || tail -f /dev/null

