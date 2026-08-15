from flask import Flask, render_template, request, jsonify
from integration import VeriLensEngine

app = Flask(__name__)

engine = VeriLensEngine()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No data received."
            }), 400

        claim = data.get("claim", "").strip()

        if not claim:
            return jsonify({
                "success": False,
                "error": "Please enter a claim."
            }), 400

        result = engine.analyze(claim)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        print("ERROR:", e)

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )