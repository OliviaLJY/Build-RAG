#!/bin/bash

# Launch script for RAG Web Frontend

echo "=================================================="
echo "  RAG AI Assistant - Web Frontend"
echo "=================================================="
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit>=1.28.0
    echo ""
fi

echo "🚀 Starting web server..."
echo ""
echo "📖 The app will open in your browser automatically"
echo "   URL: http://localhost:8501"
echo ""
echo "💡 To stop the server, press Ctrl+C"
echo ""
echo "=================================================="
echo ""

# Launch streamlit
streamlit run app.py

