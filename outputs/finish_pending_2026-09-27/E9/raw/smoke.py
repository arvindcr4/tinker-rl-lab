import json, time, urllib.request
B = "http://127.0.0.1:18019/v1"
for i in range(30):
    try:
        urllib.request.urlopen(B + "/models", timeout=900).read(); break
    except Exception as e:
        print("wait", repr(e)[:100]); time.sleep(2)
req = urllib.request.Request(B + "/chat/completions", data=json.dumps({"model": "x", "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 10, "temperature": 0}).encode(), headers={"content-type": "application/json"})
print(urllib.request.urlopen(req, timeout=900).read()[:400])
