"""
Prabha Enterprises — Voice Invoice Agent
Backend: Flask + Groq (Llama 3.3 70B)

The GROQ_API_KEY is read from environment variables (set in Render dashboard).
Never hardcode API keys in source files.
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")

# Groq client — reads GROQ_API_KEY from environment automatically
client = Groq()

SYSTEM_PROMPT = """You are an intelligent invoice assistant for Prabha Enterprises, Aurangabad, Maharashtra, India.
You help fill and edit GST Tax Invoices by understanding natural language in English, Hindi, or Hinglish.

You receive the current invoice state as JSON + the user's command.
Return ONLY valid JSON — no prose, no markdown, no explanation.

LANGUAGE UNDERSTANDING:
- paanch sau=500, ek hazaar=1000, do hazaar=2000, teen hazaar=3000, das=10, bis=20, pachaas=50, sau=100
- sawa paanch=5.25, dedh=1.5, adha=0.5, paune do=1.75, sawa=1.25
- rupaye/rupaiya/rs = price in rupees
- piece/pcs/nos/nag = unit Nos
- last item / aakhri item = last item in the list
- pehla=1st, doosra=2nd, teesra=3rd, chautha=4th
- bill to / party / customer / client = Bill To section
- same address / same as bill = copy bill address to ship address
- delete / hatao / remove = delete item
- change / update / badlo / set / karo = update a field
- HSN 4821 = default for printed materials (manuals, stickers, labels, booklets)
- Default GST = 18%, default unit = Nos

Return this exact JSON shape:
{
  "message": "one sentence describing what you did",
  "actions": [
    {"type":"add_item","name":"ITEM NAME UPPERCASE","hsn":"4821","qty":100,"unit":"Nos","price":5.50,"gst":18},
    {"type":"update_item","index":1,"field":"qty","value":500},
    {"type":"update_item","index":1,"field":"price","value":6.50},
    {"type":"update_item","index":1,"field":"name","value":"NEW NAME"},
    {"type":"update_item","index":1,"field":"gst","value":12},
    {"type":"update_item","index":1,"field":"unit","value":"Kg"},
    {"type":"delete_item","index":2},
    {"type":"clear_items"},
    {"type":"update_field","field":"billName","value":"PARTY NAME"},
    {"type":"update_field","field":"billAddr","value":"address here"},
    {"type":"update_field","field":"billGST","value":"27XXXXX1234X1ZX"},
    {"type":"update_field","field":"shipName","value":""},
    {"type":"update_field","field":"shipAddr","value":""},
    {"type":"update_field","field":"invNo","value":"25/26-444"},
    {"type":"update_field","field":"invDate","value":"2026-02-15"},
    {"type":"update_field","field":"bankDetails","value":"bank info"},
    {"type":"copy_bill_to_ship"}
  ]
}"""


@app.route("/")
def index():
    # Serve the main HTML page
    return send_from_directory("static", "index.html")


@app.route("/agent", methods=["POST"])
def agent():
    body          = request.get_json(force=True)
    user_text     = body.get("text", "").strip()
    invoice_state = body.get("invoice_state", {})

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    # Build the message we send to the AI
    user_content = (
        f"Current invoice state:\n{json.dumps(invoice_state, indent=2)}\n\n"
        f"User command: {user_text}"
    )

    try:
        # Call Groq API — Llama 3.3 70B is the best free model for structured output
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            temperature=0.1,      # Low temperature = consistent, predictable JSON output
            max_tokens=900,
            response_format={"type": "json_object"}  # Forces Groq to return valid JSON
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return jsonify(parsed)

    except json.JSONDecodeError:
        return jsonify({"error": f"AI returned invalid JSON: {raw[:200]}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    # Simple health check — Render uses this to know the app is alive
    return jsonify({"status": "ok", "model": "llama-3.3-70b-versatile"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
