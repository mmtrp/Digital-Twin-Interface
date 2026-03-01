"""
digital_twin_api.py
────────────────────────────────────────────────────────────────
Flask backend that bridges your ML Digital Twin model with the
3D sensor visualisation frontend (digital_twin_viewer.html).

HOW TO RUN:
  1. pip install flask flask-cors
  2. python digital_twin_api.py
  3. Open digital_twin_viewer.html in your browser

ENDPOINTS:
  GET  /predict          → returns latest cached predictions (polled by 3D viewer)
  POST /run_prediction   → accepts sensor CSV data, runs model, returns predictions
  POST /reset            → resets all sensors to OK / reference values
  GET  /health           → simple health-check ping

CSV COLUMN NAMES MUST MATCH THESE SENSOR IDs EXACTLY:
  engine_temp, map_sensor, o2_sensor, maf_sensor,
  throttle_pos, crankshaft, camshaft, knock_sensor,
  fuel_pressure, egr_sensor, oil_pressure, iat_sensor
────────────────────────────────────────────────────────────────
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import time
import os

app = Flask(__name__, static_folder='static')
CORS(app)  # Allow the HTML frontend (any origin) to call this API


# ══════════════════════════════════════════════════════════════
#  STATIC FILE SERVING — serves static/car.glb (and any other
#  files in the static/ folder) to the HTML frontend.
# ══════════════════════════════════════════════════════════════
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


# ══════════════════════════════════════════════════════════════
#  SENSOR REGISTRY
#  Must stay in sync with the SENSORS array in digital_twin_viewer.html
# ══════════════════════════════════════════════════════════════
SENSOR_IDS = [
    'engine_temp',
    'map_sensor',
    'o2_sensor',
    'maf_sensor',
    'throttle_pos',
    'crankshaft',
    'camshaft',
    'knock_sensor',
    'fuel_pressure',
    'egr_sensor',
    'oil_pressure',
    'iat_sensor',
]

# Reference (healthy) values + operating range for each sensor
REFERENCE_VALUES = {
    'engine_temp':   {'value': 90,    'unit': '°C',  'low': 70,   'high': 105},
    'map_sensor':    {'value': 101.3, 'unit': 'kPa', 'low': 95,   'high': 108},
    'o2_sensor':     {'value': 0.45,  'unit': 'V',   'low': 0.1,  'high': 0.9},
    'maf_sensor':    {'value': 12.0,  'unit': 'g/s', 'low': 2,    'high': 20},
    'throttle_pos':  {'value': 15,    'unit': '%',   'low': 0,    'high': 100},
    'crankshaft':    {'value': 800,   'unit': 'RPM', 'low': 600,  'high': 7000},
    'camshaft':      {'value': 0.0,   'unit': '°',   'low': -5,   'high': 5},
    'knock_sensor':  {'value': 10,    'unit': 'mV',  'low': 0,    'high': 30},
    'fuel_pressure': {'value': 3.5,   'unit': 'bar', 'low': 2.8,  'high': 4.2},
    'egr_sensor':    {'value': 18,    'unit': '%',   'low': 10,   'high': 30},
    'oil_pressure':  {'value': 45,    'unit': 'PSI', 'low': 25,   'high': 80},
    'iat_sensor':    {'value': 25,    'unit': '°C',  'low': -10,  'high': 60},
}

# In-memory cache of the most recent prediction results
latest_predictions = []


# ══════════════════════════════════════════════════════════════
#  ROUTE 1 — GET /predict
#  The 3D viewer polls this every 3 seconds (when polling is enabled).
#  Returns the latest cached prediction set.
# ══════════════════════════════════════════════════════════════
@app.route('/predict', methods=['GET'])
def get_predictions():
    """Return the latest cached sensor predictions."""
    return jsonify({
        'predictions': latest_predictions,
        'count':       len(latest_predictions),
        'timestamp':   time.time(),
    })


# ══════════════════════════════════════════════════════════════
#  ROUTE 2 — POST /run_prediction
#  Called by the frontend when a CSV file is uploaded.
#
#  Expected JSON body:
#  {
#    "sensor_data": {
#      "engine_temp":   115.0,
#      "map_sensor":    98.2,
#      "o2_sensor":     0.10,
#      ...
#    }
#  }
#
#  Returns:
#  {
#    "predictions": [
#      { "sensorId": "engine_temp", "status": "fault",
#        "curVal": 115.0, "anomalyScore": 0.89 },
#      ...
#    ],
#    "timestamp": 1234567890.0
#  }
# ══════════════════════════════════════════════════════════════
@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    global latest_predictions

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({'error': 'Invalid JSON body'}), 400

    sensor_data = body.get('sensor_data', {})
    if not isinstance(sensor_data, dict):
        return jsonify({'error': 'sensor_data must be a JSON object'}), 400

    # ── PLUG YOUR MODEL IN HERE ───────────────────────────────
    #
    # Replace the call below with your actual ML model.
    # Your model should accept a dict of { sensor_id: float }
    # and return a dict of { sensor_id: anomaly_score (0.0–1.0) }
    #
    # Example:
    #   from your_model import DigitalTwinModel
    #   model = DigitalTwinModel.load('model.pkl')
    #   anomaly_scores = model.predict(sensor_data)
    #
    # ─────────────────────────────────────────────────────────
    anomaly_scores = _placeholder_model(sensor_data)

    # Convert anomaly scores → frontend prediction format
    predictions = []
    for sid in SENSOR_IDS:
        ref   = REFERENCE_VALUES.get(sid, {})
        score = float(anomaly_scores.get(sid, 0.0))
        cur   = sensor_data.get(sid, ref.get('value'))

        # Classify status from score
        if score >= 0.7:
            status = 'fault'
        elif score >= 0.4:
            status = 'warn'
        else:
            status = 'ok'

        predictions.append({
            'sensorId':     sid,
            'status':       status,
            'curVal':       round(float(cur), 4) if cur is not None else ref.get('value'),
            'anomalyScore': round(score, 4),
        })

    latest_predictions = predictions

    return jsonify({
        'predictions': predictions,
        'timestamp':   time.time(),
    })


# ══════════════════════════════════════════════════════════════
#  ROUTE 3 — POST /reset
#  Clears all faults — resets every sensor to OK at reference value.
# ══════════════════════════════════════════════════════════════
@app.route('/reset', methods=['POST'])
def reset():
    global latest_predictions
    latest_predictions = [
        {
            'sensorId':     sid,
            'status':       'ok',
            'curVal':       REFERENCE_VALUES[sid]['value'],
            'anomalyScore': 0.0,
        }
        for sid in SENSOR_IDS
    ]
    return jsonify({'status': 'reset', 'timestamp': time.time()})


# ══════════════════════════════════════════════════════════════
#  ROUTE 4 — GET /health
#  Simple ping to confirm the server is running.
# ══════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': time.time()})


# ══════════════════════════════════════════════════════════════
#  PLACEHOLDER MODEL
#  ── REPLACE THIS WITH YOUR ACTUAL ML MODEL ──
#
#  This uses simple threshold deviation logic to mimic anomaly
#  detection. It is here only so the system works out of the box.
#
#  Input:  sensor_data  dict  { sensor_id: float_value }
#  Output: scores       dict  { sensor_id: anomaly_score 0.0–1.0 }
#
#  To plug in your own model, replace the body of this function:
#
#      def _placeholder_model(sensor_data):
#          from your_model import predict_anomalies
#          return predict_anomalies(sensor_data, reference_twin)
# ══════════════════════════════════════════════════════════════
def _placeholder_model(sensor_data: dict) -> dict:
    scores = {}
    for sid, ref in REFERENCE_VALUES.items():
        cur = sensor_data.get(sid)

        if cur is None:
            scores[sid] = 0.0
            continue

        lo      = ref['low']
        hi      = ref['high']
        ref_val = ref['value']
        span    = (hi - lo) / 2.0 + 1e-9   # half-range, avoid divide-by-zero

        # Normalised deviation from reference value (0 = perfect, 1 = at range edge)
        deviation = abs(float(cur) - ref_val) / span
        deviation = min(deviation, 1.0)

        # Apply a power curve so small deviations stay low but large ones spike
        scores[sid] = round(deviation ** 1.5, 4)

    return scores


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print()
    print('  ╔══════════════════════════════════════════════╗')
    print('  ║   Engine Digital Twin — Flask API Server    ║')
    print('  ╚══════════════════════════════════════════════╝')
    print()
    print('  Running at:  http://localhost:5000')
    print()
    print('  Endpoints:')
    print('    GET  /health           → server health check')
    print('    GET  /predict          → latest cached predictions')
    print('    POST /run_prediction   → submit sensor data → get predictions')
    print('    POST /reset            → reset all sensors to OK')
    print()
    print('  Expected POST /run_prediction body:')
    print('    { "sensor_data": { "engine_temp": 115, "o2_sensor": 0.1, ... } }')
    print()
    print('  Sensor IDs accepted:')
    for sid in SENSOR_IDS:
        ref = REFERENCE_VALUES[sid]
        print(f'    {sid:<16} ref={ref["value"]} {ref["unit"]}')
    print()
    app.run(host='0.0.0.0', port=5000, debug=True)
