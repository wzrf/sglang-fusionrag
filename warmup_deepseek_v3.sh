#!/bin/bash

# Warmup script for DeepSeek-V3.2 sglang server
# This script sends test requests to warm up the model

SERVER_URL="http://localhost:30000"
MODEL_NAME="DeepSeek-V3.2"

echo "=========================================="
echo "DeepSeek-V3.2 Server Warmup Script"
echo "=========================================="
echo "Server: $SERVER_URL"
echo "Model: $MODEL_NAME"
echo ""

# Check if server is ready
echo "[1/5] Checking server health..."
if ! curl -s "$SERVER_URL/health" > /dev/null; then
    echo "❌ Error: Server is not responding at $SERVER_URL"
    echo "Please start the server first: /mnt/data/shm/sglang-fusionrag/launch_deepseek_v3.sh"
    exit 1
fi
echo "✅ Server is healthy"
echo ""

# Test 1: Short prompt (warm up basic inference)
echo "[2/5] Warmup test 1: Short prompt..."
curl -s "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq -r '.choices[0].text' | head -3
echo "✅ Short prompt test completed"
echo ""

# # Test 2: Medium prompt (warm up KV cache)
# echo "[3/5] Warmup test 2: Medium prompt..."
# curl -s "$SERVER_URL/v1/completions" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "'"$MODEL_NAME"'",
#     "prompt": "Explain the concept of machine learning in simple terms. Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
#     "max_tokens": 100,
#     "temperature": 0.7
#   }' | jq -r '.choices[0].text' | head -3
# echo "✅ Medium prompt test completed"
# echo ""

# Test 3: Long context (warm up hierarchical cache)
# echo "[4/5] Warmup test 3: Long context..."
# LONG_CONTEXT="The history of artificial intelligence began in antiquity with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe human thinking as a symbolic system. But the field of AI wasn't formally founded until 1956, at a conference at Dartmouth College, in Hanover, New Hampshire, where the term artificial intelligence was coined. In the following decades, AI research experienced several waves of optimism and disappointment. The field went through multiple AI winters when funding and interest waned. However, recent advances in computing power, big data, and machine learning algorithms have led to a renaissance in AI research and applications."

# curl -s "$SERVER_URL/v1/completions" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "'"$MODEL_NAME"'",
#     "prompt": "'"$LONG_CONTEXT"' Based on this text, what are the key milestones in AI history?",
#     "max_tokens": 150,
#     "temperature": 0.7
#   }' | jq -r '.choices[0].text' | head -5
# echo "✅ Long context test completed"
# echo ""

# Test 4: Chat completion format
echo "[5/5] Warmup test 4: Chat completion..."
curl -s "$SERVER_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_NAME"'",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq -r '.choices[0].message.content'
echo "✅ Chat completion test completed"
echo ""

# Get server stats
echo "=========================================="
echo "Server Statistics:"
echo "=========================================="
curl -s "$SERVER_URL/get_model_info" | jq '.'
echo ""

echo "=========================================="
echo "✅ Warmup completed successfully!"
echo "=========================================="
echo "The server is now ready for production use."
echo "HiCache and KV cache should be warmed up."
