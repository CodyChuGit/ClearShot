#!/usr/bin/env bash
set -e

echo "======================================"
echo "    ClearShot - Installation Script   "
echo "======================================"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

# Check for Node.js / npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed."
    echo "Please install Node.js (which includes npm) and try again."
    exit 1
fi

echo "✅ System requirements met."

echo ""
echo "📦 Installing Python backend dependencies..."
pip3 install -r requirements.txt

echo ""
echo "📦 Installing React frontend dependencies..."
cd frontend
npm install
cd ..

echo ""
echo "🎉 Installation complete!"
echo "To start ClearShot, simply run: ./start.sh"
