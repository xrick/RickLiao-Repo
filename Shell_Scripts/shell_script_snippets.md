lsof -Pi :8000 -sTCP:LISTEN -t

===================================================

# Kill all processes on port
local all_pids=$(lsof -Pi :$port -sTCP:LISTEN -t)
for p in $all_pids; do
    kill $p 2>/dev/null
done
sleep 2

===================================================

# Verify and force kill if needed
local remaining_pids=$(lsof -Pi :$port -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$remaining_pids" ]; then
    for p in $remaining_pids; do
        kill -9 $p 2>/dev/null
    done
fi

====================================================
Server Status:
$ curl http://localhost:8000/health
{"status": "healthy", "app_name": "DocAI", "version": "1.0.0"}

====================================================
source docaienv/bin/activate && nohup python main.py > logs/server.log 2>&1 &

====================================================