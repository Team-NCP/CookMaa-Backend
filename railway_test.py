#!/usr/bin/env python3

"""
Ultra minimal Railway test server
"""

import os
from http.server import HTTPServer, BaseHTTPRequestHandler

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"message":"Railway test server"}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}")
    print(f"PORT env var: {os.environ.get('PORT', 'NOT SET')}")
    
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    print(f"Server running on 0.0.0.0:{port}")
    server.serve_forever()