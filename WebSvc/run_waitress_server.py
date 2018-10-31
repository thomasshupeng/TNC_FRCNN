import os
from waitress import serve
from tncApp import app

PORT_NUMBER = 8080
serve(app, host="0.0.0.0", port=os.environ.get('SERVER_PORT', PORT_NUMBER))
