import os
import time
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on("search_event")
def handle_search_event(data):
    search_text = data.get("search_text")
    print(f"Received search text: {search_text}")
    result = process_search(search_text)
    emit("search_response", {"result": result})

def process_search(query):
    return f"Processed query: {query.upper()}"


if __name__ == '__main__':
    socketio.run(app, debug=True)
