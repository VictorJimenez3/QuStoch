from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from stockData import get_stock_info

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on("search_event")
def handle_search_event(data):
    search_text = data.get("search_text")
    print(f"Received ticker: {search_text}")
    
    try:
        result = get_stock_info(search_text) #assumes stock ticker is valid, yfinance will throw otherwise
        result.update({"status": "success"}) 
        emit("search_response", result)

    except Exception as e:
        print(e)
        emit("search_response", {"status": "INVALID TICKER, TRY AGAIN"})

@socketio.on("backend_simulaton_event")
def handle_simulation_submission(data):
    pass

if __name__ == '__main__':
    socketio.run(app, debug=True)
