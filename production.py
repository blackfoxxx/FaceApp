from waitress import serve
from app import app
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    print("Starting Face Detection App in production mode with DEBUG enabled...")
    print("Server will be accessible at http://0.0.0.0:8080")
    print("Use your computer's IP address to access from other devices on the network")
    print("DEBUG MODE: Detailed error messages and logging enabled")
    serve(app, host='0.0.0.0', port=8080, threads=16, backlog=200, connection_limit=500, max_request_body_size=26843545600, channel_timeout=1200)
