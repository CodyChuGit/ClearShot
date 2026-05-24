import asyncio
import websockets
import json
import requests
import time

async def test():
    # 1. upload URL
    res = requests.post("http://127.0.0.1:8000/api/download-url", json={"url": "https://www.youtube.com/watch?v=LXb3EKWsInQ"})
    job = res.json()
    job_id = job["job_id"]
    print(f"Job ID: {job_id}")

    async with websockets.connect(f"ws://127.0.0.1:8000/ws/{job_id}") as ws:
        print("Connected. Sending download...")
        await ws.send(json.dumps({"action": "download", "format_id": "bestvideo"}))
        
        # Read a few messages
        for _ in range(3):
            msg = await ws.recv()
            print("Received:", msg)
            
        print("Sending abort...")
        await ws.send(json.dumps({"action": "abort"}))
        
        # Read remaining
        while True:
            msg = await ws.recv()
            print("Received:", msg)
            if json.loads(msg).get("type") in ["download_aborted", "download_complete", "error"]:
                break

asyncio.run(test())
