"""
Prabha Enterprises — Voice Invoice Agent
Backend: Flask + Groq (Llama 3.3 70B + Whisper)
"""

import os
import json
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder="static")
client = Groq()

SYSTEM_PROMPT = """You are an intelligent invoice assistant for Prabha Enterprises, Aurangabad, Maharashtra, India.
You help fill and edit GST Tax Invoices by understanding natural language in English, Hindi, or Hinglish.

You receive the current invoice state as JSON + the user's command.
Return ONLY valid JSON — no prose, no markdown, no explanation.

CRITICAL DECISION RULE — READ THIS FIRST:
Before deciding add_item vs update_item, scan the existing items list carefully.
If the user mentions ANY item name that is similar (even slightly) to an existing item → ALWAYS update, NEVER add a new one.
Similarity examples:
  "duck tape" = "ductape" = "ducktape" = "DUCK TAPE" = "D TAPE" = "adhesive tape"
  "tata manual" = "tata croma manual" = "tata inst manual"
  "kenstar" = "kenstar sticker" = "BE STICKER KENSTAR"
  "cg" = "CG MANUAL"
  "godrej" = "GODREJ FRIDGE MANUAL"
Only use add_item if the item is clearly brand new and has NO match in the existing items list.

INTENT RULES:
- "edit / change / badlo / update / set / karo / kar" = update_item (NEVER add)
- "add / naya / new / daalo" = add_item (but still check for duplicates first)
- "delete / hatao / remove / nikalo" = delete_item
- "last item / aakhri item" = highest index in items list
- "pehla"=index 1, "doosra"=index 2, "teesra"=index 3, "chautha"=index 4

FIELD UPDATE RULES:
- "quantity / qty / matra / kitna" → field: qty
- "price / rate / daam / bhav / rupaye" → field: price
- "name / naam / title" → field: name
- "HSN / hsn code" → field: hsn
- "GST / tax" → field: gst
- "unit / measurement" → field: unit

NUMBER UNDERSTANDING:
- paanch sau=500, ek hazaar=1000, do hazaar=2000, teen hazaar=3000
- das=10, bis=20, pachaas=50, sau=100, paanch=5
- sawa paanch=5.25, dedh=1.5, adha=0.5, paune do=1.75, sawa=1.25
- rupaye/rupaiya/rs = price in rupees
- piece/pcs/nos/nag = unit Nos

PARTY FIELD RULES:
- "bill to / party / customer / client" = billName + billAddr + billGST
- "same address / same as bill / wahi address" = copy_bill_to_ship
- "ship to / delivery at" = shipName + shipAddr

DEFAULTS:
- HSN: 4821 (printed materials — manuals, stickers, labels, booklets)
- GST: 18%
- Unit: Nos
- Item names: ALWAYS UPPERCASE

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
    return send_from_directory("static", "index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("audio.webm", f, "audio/webm"),
                model="whisper-large-v3-turbo",
                language="hi",        # hi = Hindi+Hinglish, best for your use case
                response_format="text"
            )
        transcript = result.strip() if isinstance(result, str) else result.text.strip()
        return jsonify({"text": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/agent", methods=["POST"])
def agent():
    body = request.get_json(force=True)
    user_text = body.get("text", "").strip()
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
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            max_tokens=900,
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
    return jsonify({"status": "ok", "model": "llama-3.3-70b-versatile", "stt": "whisper-large-v3-turbo"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
