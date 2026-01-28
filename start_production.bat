@echo off
setlocal
call venv\Scripts\activate.bat
echo Starting Face Detection App in production mode...
python -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=8080, threads=16, backlog=200, connection_limit=500, max_request_body_size=26843545600, channel_timeout=1200)"
pause
