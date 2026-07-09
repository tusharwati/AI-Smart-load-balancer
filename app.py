"""
app.py  —  AI Smart Load Balancer  (Flask Backend)
"""

from flask import Flask, render_template, jsonify, request
import pickle, random, time, os
import numpy as np

app = Flask(__name__)

# ── Load ML model ─────────────────────────────────────────────────────────────
MODEL_PATH = "models/load_balancer_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("✅ ML model loaded.")
else:
    print("⚠️  Model not found. Run train_model.py first!")


# ── Simulated servers (in-memory state) ───────────────────────────────────────
servers = {
    1: {"name": "Server 1", "load": 20.0, "latency": 80.0,  "connections": 5,  "requests": 0},
    2: {"name": "Server 2", "load": 45.0, "latency": 120.0, "connections": 12, "requests": 0},
    3: {"name": "Server 3", "load": 30.0, "latency": 95.0,  "connections": 8,  "requests": 0},
}

request_log   = []   # [{method, server, latency, timestamp}, ...]
MAX_LOG       = 50


# ── Helper: fluctuate server metrics naturally ─────────────────────────────────
def fluctuate_servers():
    for sid, s in servers.items():
        s["load"]        = max(5,  min(95, s["load"]        + random.uniform(-5, 5)))
        s["latency"]     = max(30, min(500, s["latency"]    + random.uniform(-10, 10)))
        s["connections"] = max(0,  min(49, s["connections"] + random.randint(-2, 2)))


# ── AI routing decision ────────────────────────────────────────────────────────
def ai_route():
    if model is None:
        return round_robin_route()   # fallback
    features = np.array([[
        servers[1]["load"], servers[1]["latency"], servers[1]["connections"],
        servers[2]["load"], servers[2]["latency"], servers[2]["connections"],
        servers[3]["load"], servers[3]["latency"], servers[3]["connections"],
    ]])
    return int(model.predict(features)[0])


# ── Traditional routing methods ───────────────────────────────────────────────
_rr_counter = [0]

def round_robin_route():
    _rr_counter[0] = (_rr_counter[0] % 3) + 1
    return _rr_counter[0]

def least_connections_route():
    return min(servers, key=lambda sid: servers[sid]["connections"])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    fluctuate_servers()
    return jsonify({
        "servers": [
            {
                "id":          sid,
                "name":        s["name"],
                "load":        round(s["load"], 1),
                "latency":     round(s["latency"], 1),
                "connections": s["connections"],
                "requests":    s["requests"],
            }
            for sid, s in servers.items()
        ]
    })


@app.route("/api/send_request", methods=["POST"])
def send_request():
    method = request.json.get("method", "ai")   # ai | round_robin | least_conn

    fluctuate_servers()

    if method == "ai":
        chosen = ai_route()
    elif method == "round_robin":
        chosen = round_robin_route()
    else:
        chosen = least_connections_route()

    # Simulate processing
    s = servers[chosen]
    s["requests"]    += 1
    s["connections"]  = min(49, s["connections"] + 1)
    s["load"]         = min(95, s["load"] + random.uniform(0.5, 2.0))
    sim_latency       = round(s["latency"] + random.uniform(-5, 15), 1)

    entry = {
        "method":    method,
        "server":    chosen,
        "server_name": s["name"],
        "latency":   sim_latency,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    request_log.insert(0, entry)
    if len(request_log) > MAX_LOG:
        request_log.pop()

    return jsonify({"success": True, "routed_to": chosen, "latency": sim_latency, "server_name": s["name"]})


@app.route("/api/log")
def get_log():
    return jsonify({"log": request_log[:20]})


@app.route("/api/reset")
def reset():
    for sid, s in servers.items():
        s["load"]        = random.uniform(15, 40)
        s["latency"]     = random.uniform(60, 130)
        s["connections"] = random.randint(3, 15)
        s["requests"]    = 0
    request_log.clear()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
