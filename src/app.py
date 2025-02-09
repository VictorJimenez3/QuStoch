#packages
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

#files
from stockData import get_stock_info
from quantumMonteCarloStochastic import Q_sim_start

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
    
    print(data)
    
    stock_data = get_stock_info(data["stock_ticker"])
    
    print(stock_data)

    if not all((stock_data["latest_price"], #some data is None
        data["striking_price"],
        data["maturity_time"],
        stock_data["risk_free_return"],
        stock_data["historical_volatility"])):
        print("NONETYPE FOUND: ", stock_data["latest_price"],
        data["striking_price"],
        data["maturity_time"],
        stock_data["risk_free_return"],
        stock_data["historical_volatility"])

        return

    graph_path, detailed_output = Q_sim_start(
        round(float(stock_data["latest_price"]), 2),
        round(float(data["striking_price"].replace("$", "")), 2),
        int(data["maturity_time"]),
        round(float(stock_data["risk_free_return"]), 2),
        round(float(stock_data["historical_volatility"]), 2)
    )
    
    returned_data = {
        "graph_path" : graph_path,
        "detailed_output" : detailed_output,
        "status" : "success"
    }
    
    #TODO status unsucessful

    print("sending....")

    emit("backend_simulation_event", returned_data)


if __name__ == '__main__':
    socketio.run(app, debug=True)
