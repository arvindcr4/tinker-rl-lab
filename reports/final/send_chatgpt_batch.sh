#!/bin/bash
# Batch send questions to ChatGPT Pro sessions
# Usage: ./send_chatgpt_batch.sh <session_name> <batch_number> "<questions_text>"

SESSION=$1
BATCH=$2
QUESTIONS=$3
OUTPUT_DIR="/home/arvind/tinker-rl-lab/reports/final/chatgpt_responses"
mkdir -p "$OUTPUT_DIR"

echo "[Batch $BATCH] Navigating to new chat..."
BROWSE_SESSION=$SESSION bb browse open "https://chatgpt.com" 2>&1 > /dev/null
sleep 2

echo "[Batch $BATCH] Selecting Pro model..."
BROWSE_SESSION=$SESSION bb browse snapshot 2>&1 > /dev/null
# Click model selector
MODEL_REF=$(BROWSE_SESSION=$SESSION bb browse snapshot 2>&1 | python3 -c "
import json, sys, re
data = json.loads(sys.stdin.read())
tree = data.get('tree','')
# Find 'Model selector' button ref
m = re.search(r'\[(\d+-\d+)\] button: Model selector', tree)
if m: print(m.group(1))
else: print('NOT_FOUND')
")

if [ "$MODEL_REF" != "NOT_FOUND" ]; then
    BROWSE_SESSION=$SESSION bb browse click "$MODEL_REF" 2>&1 > /dev/null
    sleep 1
    # Find Pro menuitem
    PRO_REF=$(BROWSE_SESSION=$SESSION bb browse snapshot 2>&1 | python3 -c "
import json, sys, re
data = json.loads(sys.stdin.read())
tree = data.get('tree','')
m = re.search(r'\[(\d+-\d+)\] menuitem: Pro', tree)
if m: print(m.group(1))
else: print('NOT_FOUND')
")
    if [ "$PRO_REF" != "NOT_FOUND" ]; then
        BROWSE_SESSION=$SESSION bb browse click "$PRO_REF" 2>&1 > /dev/null
        echo "[Batch $BATCH] Pro model selected"
        sleep 1
    fi
fi

echo "[Batch $BATCH] Typing questions..."
# Use eval to set text content reliably
ESCAPED=$(echo "$QUESTIONS" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
BROWSE_SESSION=$SESSION bb browse eval "
var ta = document.querySelector('textarea, [contenteditable=\"true\"]');
if (ta) {
  ta.focus();
  document.execCommand('selectAll');
  document.execCommand('insertText', false, $ESCAPED);
  ta.dispatchEvent(new Event('input', {bubbles: true}));
  'filled'
} else { 'no textarea' }
" 2>&1 > /dev/null

sleep 1

echo "[Batch $BATCH] Sending..."
# Find and click send button
SEND_REF=$(BROWSE_SESSION=$SESSION bb browse snapshot 2>&1 | python3 -c "
import json, sys, re
data = json.loads(sys.stdin.read())
tree = data.get('tree','')
m = re.search(r'\[(\d+-\d+)\] button: Send prompt', tree)
if m: print(m.group(1))
else: print('NOT_FOUND')
")

if [ "$SEND_REF" != "NOT_FOUND" ]; then
    BROWSE_SESSION=$SESSION bb browse click "$SEND_REF" 2>&1 > /dev/null
    echo "[Batch $BATCH] Sent! Waiting for response..."
else
    echo "[Batch $BATCH] ERROR: Send button not found"
    exit 1
fi

# Poll for completion (max 25 min)
for i in $(seq 1 50); do
    sleep 30
    RESULT=$(BROWSE_SESSION=$SESSION bb browse eval "
var s=document.querySelector('button[aria-label=\"Stop streaming\"]');
var t=document.querySelectorAll('[data-message-author-role=\"assistant\"]');
var last = t.length > 0 ? t[t.length-1].innerText.length : 0;
JSON.stringify({gen:!!s, turns:t.length, len:last})
" 2>&1 | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['result'])" 2>/dev/null)

    GEN=$(echo "$RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('gen',True))")
    LEN=$(echo "$RESULT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('len',0))")

    if [ "$GEN" = "False" ] && [ "$LEN" -gt 100 ] 2>/dev/null; then
        echo "[Batch $BATCH] Response complete! ($LEN chars)"
        # Extract response
        BROWSE_SESSION=$SESSION bb browse eval "
var t=document.querySelectorAll('[data-message-author-role=\"assistant\"]');
t[t.length-1].innerText
" 2>&1 | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['result'])" > "$OUTPUT_DIR/batch_${BATCH}.txt"
        echo "[Batch $BATCH] Saved to $OUTPUT_DIR/batch_${BATCH}.txt"
        exit 0
    fi
    echo "[Batch $BATCH] Still generating... (${i}x30s, $LEN chars)"
done

echo "[Batch $BATCH] TIMEOUT after 25 min"
exit 1
