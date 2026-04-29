# Hannah/start.sh
#!/usr/bin/env fish

fuser -k 3000/tcp 2>/dev/null
fuser -k 8000/tcp 2>/dev/null
fuser -k 8001/tcp 2>/dev/null
fuser -k 8002/tcp 2>/dev/null

echo "🚀 Arrancando Hannah..."

# Model server - Hannah 360M
cd ~/Github/Hannah/backend-hannah
uvicorn server.hannah_model_server:app --port 8001 &

# Model server - Qwen
uvicorn server.qwen_model_server:app --port 8002 &

# Backend FastAPI
uvicorn app:app --host 0.0.0.0 --reload --port 8000 &

# Frontend
cd ~/Github/Hannah/frontend-hannah
python -m http.server 3000 --bind 0.0.0.0 &

echo "✅ Todo corriendo:"
echo "   Frontend  → http://localhost:3000"
echo "   Backend   → http://localhost:8000"
echo "   Hannah    → http://localhost:8001"
echo "   Qwen      → http://localhost:8002"
echo ""
echo "Ctrl+C para detener todo"

wait
