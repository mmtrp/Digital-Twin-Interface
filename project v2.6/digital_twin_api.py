"""
digital_twin_api.py
────────────────────────────────────────────────────────────────
Flask backend for the Engine Digital Twin 3D viewer.
Sensor IDs now match SUZUKI_Recording6.csv headers exactly.
────────────────────────────────────────────────────────────────
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import time

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

SENSOR_IDS = [
    'Engine Speed[rpm]','ECT[degree C]','MAP[kPa]','MAF[g/s]',
    'Throttle Position[%]','Target Throttle Posi[%]','Calculated Load[%]',
    'O2S B1 S1[V]','O2S B1 S2[V]','Short Term Fuel Trim[%]',
    'Long Term Fuel Trim[%]','Total Fuel Trim[%]','Inj Pulse Width[ms]',
    'Ignition Advance[deg BTDC]','IAC Throttle Opening[%]','Battery Voltage[V]',
    'Battery Temperature[degree C]','Generator Field Duty[%]',
    'Brake Booster Pressure[kPa]','Accelerator Position[%]',
    'APP Sensor 1 Voltage[V]','APP Sensor 2 Voltage[V]',
    'TP Sensor 1 Volt[V]','TP Sensor 2 Volt[V]',
    'EVAP Canist Prg Duty[%]','VVT Gap B1 EX[degree CA]',
    'Intake Air Temperature[degree C]',
]

REFERENCE_VALUES = {
    'Engine Speed[rpm]':               {'value':2100,  'unit':'rpm',   'low':800,   'high':5500  },
    'ECT[degree C]':                   {'value':94.0,  'unit':'°C',    'low':80.0,  'high':105.0 },
    'MAP[kPa]':                        {'value':38.0,  'unit':'kPa',   'low':28.0,  'high':100.0 },
    'MAF[g/s]':                        {'value':2.8,   'unit':'g/s',   'low':0.8,   'high':12.0  },
    'Throttle Position[%]':            {'value':9.0,   'unit':'%',     'low':5.0,   'high':80.0  },
    'Target Throttle Posi[%]':         {'value':9.2,   'unit':'%',     'low':5.0,   'high':80.0  },
    'Calculated Load[%]':              {'value':25.0,  'unit':'%',     'low':10.0,  'high':80.0  },
    'O2S B1 S1[V]':                    {'value':0.45,  'unit':'V',     'low':0.06,  'high':0.95  },
    'O2S B1 S2[V]':                    {'value':0.65,  'unit':'V',     'low':0.1,   'high':0.85  },
    'Short Term Fuel Trim[%]':         {'value':0.78,  'unit':'%',     'low':-15.0, 'high':15.0  },
    'Long Term Fuel Trim[%]':          {'value':-6.0,  'unit':'%',     'low':-20.0, 'high':15.0  },
    'Total Fuel Trim[%]':              {'value':-5.2,  'unit':'%',     'low':-25.0, 'high':20.0  },
    'Inj Pulse Width[ms]':             {'value':2.2,   'unit':'ms',    'low':0.5,   'high':8.0   },
    'Ignition Advance[deg BTDC]':      {'value':23.0,  'unit':'°BTDC', 'low':-5.0,  'high':40.0  },
    'IAC Throttle Opening[%]':         {'value':28.0,  'unit':'%',     'low':10.0,  'high':60.0  },
    'Battery Voltage[V]':              {'value':13.3,  'unit':'V',     'low':11.5,  'high':15.0  },
    'Battery Temperature[degree C]':   {'value':44.0,  'unit':'°C',    'low':15.0,  'high':65.0  },
    'Generator Field Duty[%]':         {'value':20.0,  'unit':'%',     'low':6.0,   'high':75.0  },
    'Brake Booster Pressure[kPa]':     {'value':66.5,  'unit':'kPa',   'low':45.0,  'high':75.0  },
    'Accelerator Position[%]':         {'value':3.0,   'unit':'%',     'low':0.0,   'high':100.0 },
    'APP Sensor 1 Voltage[V]':         {'value':0.98,  'unit':'V',     'low':0.74,  'high':4.65  },
    'APP Sensor 2 Voltage[V]':         {'value':0.49,  'unit':'V',     'low':0.37,  'high':2.33  },
    'TP Sensor 1 Volt[V]':             {'value':0.84,  'unit':'V',     'low':0.68,  'high':4.65  },
    'TP Sensor 2 Volt[V]':             {'value':4.14,  'unit':'V',     'low':0.50,  'high':4.32  },
    'EVAP Canist Prg Duty[%]':         {'value':14.0,  'unit':'%',     'low':0.0,   'high':100.0 },
    'VVT Gap B1 EX[degree CA]':        {'value':0.0,   'unit':'°CA',   'low':-3.0,  'high':3.0   },
    'Intake Air Temperature[degree C]':{'value':47.0,  'unit':'°C',    'low':-10.0, 'high':70.0  },
}

latest_predictions = []

@app.route('/predict', methods=['GET'])
def get_predictions():
    return jsonify({'predictions':latest_predictions,'count':len(latest_predictions),'timestamp':time.time()})

@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    global latest_predictions
    body=request.get_json(force=True,silent=True)
    if not body: return jsonify({'error':'Invalid JSON body'}),400
    sensor_data=body.get('sensor_data',{})
    if not isinstance(sensor_data,dict): return jsonify({'error':'sensor_data must be a JSON object'}),400
    anomaly_scores=_placeholder_model(sensor_data)
    predictions=[]
    for sid in SENSOR_IDS:
        ref=REFERENCE_VALUES.get(sid,{'value':0,'unit':''})
        score=float(anomaly_scores.get(sid,0.0))
        cur=sensor_data.get(sid,ref.get('value'))
        status='fault' if score>=0.7 else('warn' if score>=0.4 else 'ok')
        predictions.append({'sensorId':sid,'status':status,'curVal':round(float(cur),4) if cur is not None else ref.get('value'),'anomalyScore':round(score,4)})
    latest_predictions=predictions
    return jsonify({'predictions':predictions,'timestamp':time.time()})

@app.route('/reset', methods=['POST'])
def reset():
    global latest_predictions
    latest_predictions=[{'sensorId':sid,'status':'ok','curVal':REFERENCE_VALUES[sid]['value'],'anomalyScore':0.0} for sid in SENSOR_IDS]
    return jsonify({'status':'reset','timestamp':time.time()})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','timestamp':time.time()})

def _placeholder_model(sensor_data):
    """Threshold deviation scoring. Replace with your ML model."""
    scores={}
    for sid,ref in REFERENCE_VALUES.items():
        cur=sensor_data.get(sid)
        if cur is None: scores[sid]=0.0; continue
        span=(ref['high']-ref['low'])/2.0+1e-9
        scores[sid]=round(min(abs(float(cur)-ref['value'])/span**1.5,1.0),4)
    return scores

if __name__=='__main__':
    print('\n  Engine Digital Twin — Flask API')
    print(f'  Sensors: {len(SENSOR_IDS)} Suzuki OBD IDs')
    print('  http://localhost:5000\n')
    app.run(host='0.0.0.0',port=5000,debug=True)
