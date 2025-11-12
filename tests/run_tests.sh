#!/usr/bin/env bash
# ==========================================================
# Finance Audit RAG – Automated Test Script
# ==========================================================
# Make sure:
# - The virtual environment is active
# - FastAPI dependencies are installed
# - Redis is running locally
# ==========================================================

set -e  # stop on first error
set -o pipefail

echo "🔹 [1/8] Checking prerequisites..."
if ! command -v curl &> /dev/null; then
  echo "❌ curl not found. Please install curl."
  exit 1
fi

if [ ! -f "tests/sample_sox.txt" ]; then
  echo "❌ Missing tests/sample_sox.txt file."
  exit 1
fi

echo "✅ Prerequisites OK."

# ----------------------------------------------------------
echo "🔹 [2/8] Starting FastAPI server in background..."
python api.py > tests/api_test.log 2>&1 &
API_PID=$!
sleep 5  # wait for the server to start

cleanup() {
  echo "🧹 Cleaning up..."
  kill $API_PID 2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------------------------------------
echo "🔹 [3/8] Testing /ingest endpoint..."
curl -s -X POST -F "file=@tests/sample_sox.txt" http://localhost:8000/ingest | tee tests/output_ingest.json
echo
if grep -q '"status": "ok"' tests/output_ingest.json; then
  echo "✅ Ingest successful."
else
  echo "❌ Ingest failed."; exit 1
fi

# ----------------------------------------------------------
echo "🔹 [4/8] Testing /query (first call, no cache)..."
QUERY='What are key SOX controls for revenue recognition?'
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\"}" | tee tests/output_query1.json
echo
if grep -q '"answer":' tests/output_query1.json; then
  echo "✅ Query 1 successful."
else
  echo "❌ Query 1 failed."; exit 1
fi

# ----------------------------------------------------------
echo "🔹 [5/8] Testing /query (cached Redis response)..."
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY\"}" | tee tests/output_query2.json
echo
if grep -q '"source": "cache"' tests/output_query2.json; then
  echo "✅ Cache hit verified."
else
  echo "⚠️ Cache miss (check Redis running)."
fi

# ----------------------------------------------------------
echo "🔹 [6/8] Testing /update_sec endpoint..."
UPDATE_TEXT="New SEC guidance emphasizes continuous monitoring of SOX control effectiveness."
curl -s -X POST -F "text=$UPDATE_TEXT" http://localhost:8000/update_sec | tee tests/output_update.json
echo
if grep -q '"status": "ok"' tests/output_update.json; then
  echo "✅ SEC update tool executed."
else
  echo "❌ SEC update failed."; exit 1
fi

# ----------------------------------------------------------
echo "🔹 [7/8] Testing retrieval after SEC update..."
QUERY2='What does the new SEC guidance mention about SOX control monitoring?'
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$QUERY2\"}" | tee tests/output_query3.json
echo
echo "✅ Query after SEC update executed."

# ----------------------------------------------------------
echo "🔹 [8/8] Summary:"
echo "   - Logs: tests/api_test.log"
echo "   - Outputs: tests/output_*.json"
echo "✅ All tests completed successfully!"
