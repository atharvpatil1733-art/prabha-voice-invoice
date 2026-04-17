"""
Prabha Enterprises — Voice Invoice Agent
Backend: Flask + Groq (Llama 3.3 70B)
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")
client = Groq()

SYSTEM_PROMPT = """You are an intelligent invoice assistant for Prabha Enterprises, Aurangabad, Maharashtra, India.
You help fill and edit GST Tax Invoices by understanding natural language commands in English.

You receive the current invoice state as JSON + the user's command.
Return ONLY valid JSON — no prose, no markdown, no explanation.

=== ONE-SENTENCE FULL INVOICE (most important feature) ===
When user says something like:
  "Invoice for Lambodar, 500 Tata manuals at 5.50 and 600 BE stickers at 0.90"
  "Make invoice for ABC Company, 1000 CG manuals at 0.95 rupees"
  "New invoice for XYZ, 200 Godrej manuals 2 rupees each, 100 stickers at 1 rupee"

You MUST return ALL actions together in one response:
  1. update_field for billName (and shipName same)
  2. clear_items (to start fresh)
  3. add_item for EACH item mentioned

Party name rules:
  - "for [name]" or "to [name]" = billName and shipName
  - If address/GSTIN not mentioned, leave them as-is (don't clear them)
  - Always copy billName to shipName unless user says different ship address

Item parsing rules:
  - "[qty] [item name] at [price]" or "[qty] [item name] [price] rupees"
  - Numbers: 500=500, 1000=1000, 0.90=0.90, 5.50=5.50
  - "each" / "rupees" / "at" / "per piece" all mean price
  - If GST not mentioned, default 18%
  - If HSN not mentioned, default 4821
  - If unit not mentioned, default Nos
  - Item names ALWAYS UPPERCASE

=== SINGLE COMMANDS ===
- "bill to ABC Company" -> update_field billName only
- "add Tata manual 500 qty 5.50 price" -> add_item only
- "item 2 quantity 2000" -> update_item index 2
- "delete item 3" -> delete_item index 3
- "invoice number 25/26-450" -> update_field invNo
- "same address for shipping" -> copy_bill_to_ship
- "clear all items" -> clear_items

=== FIELD NAMES (exact, case-sensitive) ===
billName, billAddr, billGST, shipName, shipAddr, invNo, invDate, bankDetails

=== OUTPUT FORMAT ===
{
  "message": "one sentence describing what you did",
  "actions": [
    {"type":"update_field","field":"billName","value":"PARTY NAME UPPERCASE"},
    {"type":"update_field","field":"shipName","value":"PARTY NAME UPPERCASE"},
    {"type":"clear_items"},
    {"type":"add_item","name":"ITEM NAME UPPERCASE","hsn":"4821","qty":500,"unit":"Nos","price":5.50,"gst":18},
    {"type":"add_item","name":"SECOND ITEM UPPERCASE","hsn":"4821","qty":600,"unit":"Nos","price":0.90,"gst":18},
    {"type":"update_item","index":1,"field":"qty","value":500},
    {"type":"update_item","index":1,"field":"price","value":6.50},
    {"type":"update_item","index":1,"field":"name","value":"NEW NAME"},
    {"type":"update_item","index":1,"field":"gst","value":12},
    {"type":"update_item","index":1,"field":"unit","value":"Kg"},
    {"type":"delete_item","index":2},
    {"type":"update_field","field":"billAddr","value":"address here"},
    {"type":"update_field","field":"billGST","value":"27XXXXX1234X1ZX"},
    {"type":"update_field","field":"invNo","value":"25/26-444"},
    {"type":"update_field","field":"invDate","value":"2026-02-15"},
    {"type":"copy_bill_to_ship"}
  ]
}

CRITICAL: For one-sentence full invoice commands, ALWAYS include clear_items + all add_item actions.
CRITICAL: Party names and item names ALWAYS in UPPERCASE.
CRITICAL: Return ONLY the JSON object, nothing else."""


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    try:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=("audio.webm", audio_file.read(), "audio/webm"),
            language="en",
            response_format="text"
        )
        return jsonify({"text": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/agent", methods=["POST"])
def agent():
    body          = request.get_json(force=True)
    user_text     = body.get("text", "").strip()
    invoice_state = body.get("invoice_state", {})

    if not user_text:
        return jsonify({"error": "No text provided"}), 400

    user_content = (
        f"Current invoice state:\n{json.dumps(invoice_state, indent=2)}\n\n"
        f"User command: {user_text}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            temperature=0.1,
            max_tokens=1200,
            response_format={"type": "json_object"}
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
    return jsonify({"status": "ok", "model": "llama-3.3-70b-versatile"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
