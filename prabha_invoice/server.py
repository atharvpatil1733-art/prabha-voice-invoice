"""
Prabha Enterprises — Voice Invoice Agent v3
Real business data extracted from 10 actual invoices
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")
client = Groq()

# ── WHISPER PROMPT ─────────────────────────────────────────────────────────────
# Teaches Whisper the exact words it will hear — improves accuracy massively
# All real customer names, product names, and business terms from Prabha's invoices
WHISPER_PROMPT = (
    "Invoice for Prabha Enterprises Aurangabad. "
    "Customers: Lambodar Moulders, Badve Autocoms, Badve Autotech, BMR Havac, "
    "Elegant Coating, Laxmi Metal Pressing Works, Nahars Engineering, Pavna Industries. "
    "Products: Doctor Tape 10MM 3 layer shot blasting, Doctor Tape 8MM, "
    "INST 121 manual, sticker PVC wiring, BE sticker Kenstar, history card book, "
    "wire tube condenser assembly, instructions label Indo Asian, instructions label Legrand, "
    "Tata Croma installation manual, Tata Croma rating label, "
    "service number sticker Kenstar, tested OK sticker, inst manual Pulse NIX, "
    "small multi color sticker, green square sticker, blue square sticker. "
    "Terms: quantity, price per piece, rupees, HSN 4821, HSN 4820, GST 18 percent, "
    "SGST, CGST, IGST, invoice number, bill to, ship to, same address, Nos."
)

# ── SYSTEM PROMPT — with ALL real data from 10 invoices ───────────────────────
# WHY: Instead of the AI guessing, it now KNOWS Prabha's exact customers and products.
# Saying "Lambodar" auto-fills full name + address + GSTIN. Saying "doctor tape" 
# auto-fills correct HSN, price, item name. This is "few-shot prompting" — 
# real examples = smarter AI, no extra cost, no retraining needed.
SYSTEM_PROMPT = """You are an expert invoice agent for Prabha Enterprises, Aurangabad, Maharashtra, India.
You understand spoken English commands and fill GST Tax Invoices intelligently.
Return ONLY valid JSON. No prose, no markdown, no explanation.

=== KNOWN CUSTOMERS (memorized from real invoices) ===
When customer name is recognized, always fill name + address + GSTIN automatically.

1. LAMBODAR / LAMBODAR MOULDERS:
   billName: LAMBODAR MOULDERS PVT LTD
   billAddr: GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD
   billGST: 27AACCL7143A1ZV
   placeSupply: 27-Maharashtra

2. BADVE AUTOCOMS / BADVE AUTOCOMS PVT LTD:
   billName: BADVE AUTOCOMS PVT LTD
   billAddr: A-3 CHAKAN TALEGAON ROAD MAHALUNGE TQ CHAKAN DI PUNE 410501
   billGST: 27AABCB3720G1Z3
   placeSupply: 27-Maharashtra

3. BADVE AUTOTECH / BADVE AUTOTECH 4P:
   billName: BADVE AUTOTECH PVT LTD 4P
   billAddr: PLOT 535 AT VITHLAPURA BECHRAJI ROAD TQ MANDAL DIST AHEMADABAD 382120
   billGST: 24AAFCB9076K1ZB
   placeSupply: 24-Gujarat

4. BMR HAVAC / BMR:
   billName: BMR HAVAC LTD PLANT III-A
   billAddr: PLOT NO F-212 MIDC SUPA PARNER IND PARK DI AHEMADNAGAR MAHARASHTRA 414301
   billGST: 27AABCH9456K1Z3
   placeSupply: 27-Maharashtra

5. ELEGANT COATING / ELEGANT:
   billName: ELEGANT COATING PVT LTD
   billAddr: ELEGANT COATINGS PRIVATE LIMITED GUT NO-10 FAROLA VILLAGE TAL - PAITHAN
   billGST: 27AAACE4983F1ZK
   placeSupply: 27-Maharashtra

6. LAXMI METAL / LAXMI:
   billName: LAXMI METAL PRESSING WORKS PVT LTD
   billAddr: WALUJ
   billGST: 27AAACL5640G1ZN
   placeSupply: 27-Maharashtra

7. NAHARS / NAHARS ENGINEERING:
   billName: NAHARS ENGINEERING INDIA PVT LTD (UNIT-II)
   billAddr: SY NO 56/4 &57 NARSAPUR VILLAGE DIST KOLAR-563133
   billGST: 29AABCE3808J1ZK
   placeSupply: 29-Karnataka

8. PAVNA / PAVNA INDUSTRIES:
   billName: PAVNA INDUSTRIES LIMITED
   billAddr: UNIT-VI, C-11 (PART-A), FIVE-STAR MIDC, SHENDRA VILLAGE - KUMBHEPAL, TALUKA AND DISTRICT, AURANGABAD - 4310001
   billGST: 27AACCP0664L1Z8
   placeSupply: 27-Maharashtra

=== KNOWN ITEMS (memorized from real invoices) ===
When item name is recognized, use exact name + HSN + typical price.
User may say it casually — match to the real item.

- "doctor tape 10mm" / "doctor tape 3 layer" / "shot blasting tape":
  name: DOCTOR TAPE 10MM (3 LAYER) FOR SHOT BLASTING | hsn: 4821 | price: 0.12 | gst: 18

- "doctor tape 8mm" / "doctor tape":
  name: DOCTOR TAPE 8 MM X 8MM | hsn: 4821 | price: 0.12 | gst: 18

- "inst 121 manual" / "121 manual":
  name: INST 121 MANUAL | hsn: 4820 | price: 0.80 | gst: 18

- "sticker pvc wiring" / "pvc wiring sticker" / "2 pin sticker":
  name: STICKER PVC WIRING DIA 2 PIN W/O 121 | hsn: 4821 | price: 0.50 | gst: 18

- "be sticker kenstar" / "kenstar sticker" / "be sticker":
  name: BE STICKER KENSTAR | hsn: 4821 | price: 0.90 | gst: 18

- "history card book" / "history card":
  name: HISTORY CARD BOOK | hsn: 4820 | price: 0.50 | gst: 18

- "wire tube condenser" / "condenser assembly" / "wire condenser":
  name: WIRE & TUBE CONDENSER ASSLY | hsn: 4821 | price: 0.65 | gst: 18

- "instructions label indo asian" / "indo asian label" / "indo asian":
  name: INSTRUCTIONS LABLE INDO ASIAN | hsn: 4821 | price: 0.20 | gst: 18

- "instructions label legrand" / "legrand label" / "legrand":
  name: INSTRUCTIONS LABLE LEGRAND | hsn: 4821 | price: 0.20 | gst: 18

- "tata croma manual" / "tata manual" / "croma manual" / "tata croma inst manual":
  name: TATA CROMA INST MANUAL 50L | hsn: 4820 | price: 5.50 | gst: 18

- "tata croma rating label" / "tata rating label" / "croma label":
  name: TATA CROMA RATING LABLE 50 L | hsn: 4821 | price: 0.70 | gst: 18

- "service number sticker kenstar" / "service sticker kenstar" / "kenstar number sticker":
  name: SERVICE NUMBER STICKER KENSTAR | hsn: 4821 | price: 0.25 | gst: 18

- "tested ok sticker" / "tested ok":
  name: TESTED OK STICKER | hsn: 4820 | price: 0.15 | gst: 18

- "inst manual pulse" / "pulse manual" / "pulse nix manual":
  name: Inst MANUAL PULSE 20& NIX | hsn: 4820 | price: 0.80 | gst: 18

- "small multi color sticker" / "multicolor sticker":
  name: SMALL MULTI COLOR STICKER | hsn: 4821 | price: 0.04 | gst: 18

- "green square sticker" / "nt green square":
  name: 52DM0209 38X20 NT GREEN SQUARE | hsn: 4821 | price: 0.12 | gst: 18

- "blue square sticker" / "nt blue square":
  name: 52DS0301 38X20 NT BLUE SQUARE | hsn: 4821 | price: 0.12 | gst: 18

=== ONE-SENTENCE FULL INVOICE ===
"Invoice for Lambodar, 500 Tata manuals and 8000 BE stickers"
→ fill customer (name+addr+gstin+placeSupply) + clear_items + add all items

"Invoice for Elegant, 52000 Indo Asian labels"
→ fill customer + clear_items + add item (use known price 0.20)

RULES:
- Always copy billName to shipName, billAddr to shipAddr (unless told otherwise)
- If price not mentioned for known item → use typical price from above
- If price mentioned → use the mentioned price (overrides typical)
- If qty not mentioned → ask in message field, but still add item with qty 1
- Always UPPERCASE for all names
- Default HSN: 4821, Default GST: 18%, Default unit: Nos

=== SINGLE COMMANDS ===
- "bill to [name]" → update_field billName + fill known address/gstin if recognized
- "add [item] [qty]" → add_item using known item data
- "item [n] quantity [x]" → update_item
- "delete item [n]" → delete_item
- "invoice number [x]" → update_field invNo
- "same address" → copy_bill_to_ship
- "clear all" → clear_items

=== FIELD NAMES (exact) ===
billName, billAddr, billGST, shipName, shipAddr, invNo, invDate, bankDetails, placeSupply

=== OUTPUT JSON ===
{
  "message": "one sentence of what you did",
  "actions": [
    {"type":"update_field","field":"billName","value":"LAMBODAR MOULDERS PVT LTD"},
    {"type":"update_field","field":"billAddr","value":"GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD"},
    {"type":"update_field","field":"billGST","value":"27AACCL7143A1ZV"},
    {"type":"update_field","field":"shipName","value":"LAMBODAR MOULDERS PVT LTD"},
    {"type":"update_field","field":"shipAddr","value":"GUT NO 926 SOMPURI ROAD BIDKIN AURANGABAD"},
    {"type":"update_field","field":"placeSupply","value":"27-Maharashtra"},
    {"type":"clear_items"},
    {"type":"add_item","name":"TATA CROMA INST MANUAL 50L","hsn":"4820","qty":500,"unit":"Nos","price":5.50,"gst":18},
    {"type":"add_item","name":"BE STICKER KENSTAR","hsn":"4821","qty":8000,"unit":"Nos","price":0.90,"gst":18},
    {"type":"update_item","index":1,"field":"qty","value":500},
    {"type":"delete_item","index":2},
    {"type":"update_field","field":"invNo","value":"26/27-25"},
    {"type":"copy_bill_to_ship"}
  ]
}

CRITICAL: Always fill placeSupply when customer is recognized.
CRITICAL: Always copy bill to ship unless told otherwise.
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
            prompt=WHISPER_PROMPT,
            response_format="text",
            temperature=0.0
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
            max_tokens=1500,
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
