"""
Prabha Enterprises — Voice Invoice Agent v2
Backend: Flask + Groq (Llama 3.3 70B) + Groq Whisper
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")
client = Groq()

# ── WHISPER PROMPT ─────────────────────────────────────────────────────────────
# This tells Whisper the vocabulary it will hear — massively improves accuracy
# for Indian names, product names, GST terms, and numbers spoken in English
WHISPER_PROMPT = (
    "Invoice for Prabha Enterprises Aurangabad. "
    "Customers: Lambodar Moulders, Tata, Kenstar, Godrej, CG, BE, Croma. "
    "Products: Tata Croma installation manual, BE sticker Kenstar, CG manual, "
    "Godrej fridge manual, instruction booklet, label, sticker. "
    "Numbers: quantity, price per piece, rupees, HSN 4821, GST 18 percent, SGST, CGST. "
    "Commands: invoice for, add item, delete item, update quantity, update price, "
    "bill to, ship to, invoice number, same address."
)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert invoice agent for Prabha Enterprises, Aurangabad, Maharashtra, India.
You understand spoken English commands and fill GST Tax Invoices intelligently.
Return ONLY valid JSON. No prose, no markdown.

=== PRABHA'S REAL BUSINESS CONTEXT ===
Common customers:
- LAMBODAR MOULDERS PVT LTD | Address: GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD | GSTIN: 27AACCL7143A1ZV
- TATA, KENSTAR, GODREJ, CG, BE (brand names — use as item prefixes)

Common items (HSN 4821, GST 18% default):
- TATA CROMA INSTALLATION MANUAL — typically 5.50 rupees
- BE STICKER KENSTAR — typically 0.90 rupees
- CG MANUAL — typically 0.95 rupees
- GODREJ FRIDGE MANUAL — typically 0.95 rupees
- INSTRUCTION BOOKLET — varies
- LABEL / STICKER — varies

=== ONE-SENTENCE FULL INVOICE (KEY FEATURE) ===
"Invoice for Lambodar, 500 Tata manuals at 5.50 and 600 BE stickers at 0.90"
→ update_field billName + update_field shipName + clear_items + add_item x N

"Make invoice for Lambodar" (no items mentioned)
→ update_field billName + update_field shipName + copy known address+GSTIN if customer recognized

SMART RULES:
- If customer name sounds like "Lambodar" / "Lambodar Moulders" → use full name + known address + GSTIN
- If item sounds like "Tata manual" / "installation manual" → use "TATA CROMA INSTALLATION MANUAL"
- If price not mentioned for known items → use typical price from context above
- If GST not mentioned → 18%
- If HSN not mentioned → 4821
- If unit not mentioned → Nos
- Always UPPERCASE for names and item names
- "and" between items = multiple add_item actions

=== SINGLE COMMANDS ===
- "bill to [name]" → update_field billName
- "add [item] [qty] at [price]" → add_item
- "item [n] quantity [x]" → update_item
- "delete item [n]" → delete_item
- "invoice number [x]" → update_field invNo
- "same address" → copy_bill_to_ship
- "clear all" → clear_items

=== FIELD NAMES (exact) ===
billName, billAddr, billGST, shipName, shipAddr, invNo, invDate, bankDetails

=== OUTPUT JSON ===
{
  "message": "one sentence of what you did",
  "actions": [
    {"type":"update_field","field":"billName","value":"LAMBODAR MOULDERS PVT LTD"},
    {"type":"update_field","field":"billAddr","value":"GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD"},
    {"type":"update_field","field":"billGST","value":"27AACCL7143A1ZV"},
    {"type":"update_field","field":"shipName","value":"LAMBODAR MOULDERS PVT LTD"},
    {"type":"update_field","field":"shipAddr","value":"GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD"},
    {"type":"clear_items"},
    {"type":"add_item","name":"TATA CROMA INSTALLATION MANUAL","hsn":"4821","qty":500,"unit":"Nos","price":5.50,"gst":18},
    {"type":"add_item","name":"BE STICKER KENSTAR","hsn":"4821","qty":600,"unit":"Nos","price":0.90,"gst":18},
    {"type":"update_item","index":1,"field":"qty","value":500},
    {"type":"delete_item","index":2},
    {"type":"copy_bill_to_ship"}
  ]
}

CRITICAL: For full invoice commands always: clear_items first, then add all items.
CRITICAL: If customer is recognized, always fill address and GSTIN too.
CRITICAL: Return ONLY the JSON object."""


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
            prompt=WHISPER_PROMPT,        # <-- KEY: business vocab injection
            response_format="text",
            temperature=0.0               # <-- deterministic, most accurate
        )
        return jsonify({"text": str(transcription).strip()})
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
