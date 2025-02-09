from flask import Flask, render_template, url_for
from flask_cors import CORS
from flask_socketio import SocketIO


app = Flask(__name__)
CORS(app)
socket = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socket.run(app=app, debug=True)