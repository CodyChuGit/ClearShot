#!/usr/bin/env bash
set -e

echo "🚀 Starting ClearShot Servers..."

# Function to cleanly shut down both servers on CTRL+C
cleanup() {
    echo ""
    echo "🛑 Shutting down ClearShot servers..."
    if [ -n "$BACKEND_PID" ]; then kill -TERM "$BACKEND_PID" 2>/dev/null || true; fi
    if [ -n "$FRONTEND_PID" ]; then kill -TERM "$FRONTEND_PID" 2>/dev/null || true; fi
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "-> Starting FastAPI backend on http://localhost:8001"
python3 server.py &
BACKEND_PID=$!

echo "-> Starting Vite React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both servers are running!"
echo "   Frontend UI: http://localhost:5173"
echo "   Backend API: http://localhost:8001"
echo "Press CTRL+C to stop both servers."
echo ""

# Wait for background processes
wait $BACKEND_PID
wait $FRONTEND_PID
