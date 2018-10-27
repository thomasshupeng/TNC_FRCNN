import os
from waitress import serve
from tncApp import app

serve(app, host="0.0.0.0", port=os.environ["SERVER_PORT"])
