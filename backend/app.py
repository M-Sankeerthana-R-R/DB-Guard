# gui/app.py
#1.Run python app.py in /backend
#2.npm start
from flask import Flask, jsonify, send_file
import pandas as pd
import os
import json
from flask_cors import CORS
from routes.clientProxy import client_bp
from monitor.slow_query_analyzer import analyze_root_causes

app = Flask(__name__)
CORS(app)

LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'logs', 'activity_log.csv')
CONNECTED_FILE = os.path.join(os.path.dirname(__file__), 'logs', 'connected_clients.json')

@app.route('/api/dashboard')
def dashboard():
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"error": "Log file not found."}), 404

    try:
        df = pd.read_csv(LOG_FILE_PATH, dtype=str)

        if 'Result' not in df.columns:
            df['Result'] = ""

        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp'])
        else:
            df['Timestamp'] = pd.to_datetime('now')

        if 'ExecutionTime' in df.columns:
            df['ExecutionTime'] = pd.to_numeric(df['ExecutionTime'], errors='coerce').fillna(0)
        else:
            df['ExecutionTime'] = 0

        if 'SlowQuery' in df.columns:
            df['SlowQuery'] = df['SlowQuery'].astype(str).str.lower().isin(['true','1','yes'])
        else:
            df['SlowQuery'] = False

        total_unique_clients = int(df['ClientID'].nunique())
        client_counts = df['ClientID'].value_counts().to_dict()

        slow_counts = {
            'Slow': int(df['SlowQuery'].sum()),
            'Fast': int((~df['SlowQuery']).sum())
        }

        df_temp = df.copy()
        df_temp.set_index('Timestamp', inplace=True)
        five_min_counts_series = df_temp['Query'].resample('5min').count().sort_index()
        five_min_counts = {
            timestamp.strftime('%Y-%m-%d %H:%M'): int(count)
            for timestamp, count in five_min_counts_series.items()
        }

        connected_clients = []
        if os.path.exists(CONNECTED_FILE):
            with open(CONNECTED_FILE, 'r', encoding='utf-8') as f:
                try:
                    connected_clients = json.load(f)
                except:
                    connected_clients = []

        return jsonify({
            "total_clients": total_unique_clients,
            "current_connected": len(connected_clients),
            "connected_clients": connected_clients,
            "client_counts": client_counts,
            "slow_counts": slow_counts,
            "five_min_counts": five_min_counts
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/api/client/<client_id>')
# def client_details(client_id):
#     if not os.path.exists(LOG_FILE_PATH):
#         return jsonify({"error": "Log file not found."}), 404

#     try:
#         df = pd.read_csv(LOG_FILE_PATH, dtype=str)
#         if 'Result' not in df.columns:
#             df['Result'] = ""

#         if 'Timestamp' in df.columns:
#             df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#             df = df.dropna(subset=['Timestamp'])
#         else:
#             df['Timestamp'] = pd.to_datetime('now')

#         if 'SlowQuery' in df.columns:
#             df['SlowQuery'] = df['SlowQuery'].astype(str).str.lower().isin(['true','1','yes'])
#         else:
#             df['SlowQuery'] = False

#         df_client = df[df['ClientID'].astype(str) == str(client_id)].sort_values(by='Timestamp', ascending=False)

#         if df_client.empty:
#             return jsonify({"error": f"No data for client {client_id}"}), 404

#         total_queries = len(df_client)
#         slow_queries = int(df_client['SlowQuery'].sum())

#         change_mask = ~df_client['Query'].str.lower().str.strip().str.startswith(('select','show','desc','describe'))
#         change_count = int(change_mask.sum())
#         view_count = int((~change_mask).sum())

#         records = []
#         for _, row in df_client.iterrows():
#             records.append({
#                 'Timestamp': row.get('Timestamp').strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(row.get('Timestamp')) else '',
#                 'Query': row.get('Query', ''),
#                 'Result': row.get('Result', '')
#             })

#         return jsonify({
#             "client_id": client_id,
#             "total_queries": total_queries,
#             "slow_queries": slow_queries,
#             "change_count": change_count,
#             "view_count": view_count,
#             "queries": records
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
# @app.route('/api/client/<client_id>')
# def client_details(client_id):
#     if not os.path.exists(LOG_FILE_PATH):
#         return jsonify({"error": "Log file not found."}), 404

#     try:
#         df = pd.read_csv(LOG_FILE_PATH, dtype=str)
#         if 'Result' not in df.columns:
#             df['Result'] = ""

#         if 'Timestamp' in df.columns:
#             df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#             df = df.dropna(subset=['Timestamp'])
#         else:
#             df['Timestamp'] = pd.to_datetime('now')

#         if 'SlowQuery' in df.columns:
#             df['SlowQuery'] = df['SlowQuery'].astype(str).str.lower().isin(['true','1','yes'])
#         else:
#             df['SlowQuery'] = False

#         df_client = df[df['ClientID'].astype(str) == str(client_id)].sort_values(by='Timestamp', ascending=False)

#         if df_client.empty:
#             return jsonify({"error": f"No data for client {client_id}"}), 404

#         total_queries = len(df_client)
#         slow_queries = int(df_client['SlowQuery'].sum())

#         change_mask = ~df_client['Query'].str.lower().str.strip().str.startswith(('select','show','desc','describe'))
#         change_count = int(change_mask.sum())
#         view_count = int((~change_mask).sum())

#         records = []
#         for _, row in df_client.iterrows():
#             query_text = row.get('Query', '')
#             # 🔹 Added ML root cause ranking
#             ranking = analyze_root_causes(query_text)  # returns list of causes with scores

#             records.append({
#                 'Timestamp': row.get('Timestamp').strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(row.get('Timestamp')) else '',
#                 'Query': query_text,
#                 'Result': row.get('Result', ''),
#                 'Ranking': ranking  # <-- new field added
#             })

#         return jsonify({
#             "client_id": client_id,
#             "total_queries": total_queries,
#             "slow_queries": slow_queries,
#             "change_count": change_count,
#             "view_count": view_count,
#             "queries": records
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route('/api/client/<client_id>')
def client_details(client_id):
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"error": "Log file not found."}), 404

    try:
        df = pd.read_csv(LOG_FILE_PATH, dtype=str)

        # Ensure required columns exist
        if 'Result' not in df.columns:
            df['Result'] = ""

        # Parse and clean timestamp
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df = df.dropna(subset=['Timestamp'])
        else:
            df['Timestamp'] = pd.to_datetime('now')

        # Normalize slow query column
        if 'SlowQuery' in df.columns:
            df['SlowQuery'] = df['SlowQuery'].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['SlowQuery'] = False

        # Filter this client's data
        df_client = df[df['ClientID'].astype(str) == str(client_id)].sort_values(by='Timestamp', ascending=False)
        if df_client.empty:
            return jsonify({"error": f"No data for client {client_id}"}), 404

        # Metrics
        total_queries = len(df_client)
        slow_queries = int(df_client['SlowQuery'].sum())

        # Identify data-changing vs view-only queries
        change_mask = ~df_client['Query'].str.lower().str.strip().str.startswith(('select', 'show', 'desc', 'describe'))
        change_count = int(change_mask.sum())
        view_count = int((~change_mask).sum())

        # Query-level analysis
        records = []
        alerts = []

        for _, row in df_client.iterrows():
            query_text = row.get('Query', '')
            
            # 🔹 Use ML model for analysis
            analysis = analyze_root_causes(query_text)
            status = analysis.get("status", "")
            alert = analysis.get("alert", "")
            causes = analysis.get("root_causes", [])
            score = analysis.get("score", 0.0)

            # Collect alerts (only if meaningful)
            if status in ["Slow", "Near Slow"]:
                alerts.append(alert)

            records.append({
                "Timestamp": row.get('Timestamp').strftime('%Y-%m-%d %H:%M:%S') if not pd.isnull(row.get('Timestamp')) else '',
                "Query": query_text,
                "Result": row.get('Result', ''),
                "Status": status,
                "Score": score,
                "RootCauses": causes,   # list of {"cause":..., "score":...}
                "Alert": alert
            })

        # Prepare final structured response
        response = {
            "client_id": client_id,
            "total_queries": total_queries,
            "slow_queries": slow_queries,
            "change_count": change_count,
            "view_count": view_count,
            "queries": records,
            "alerts": list(set(alerts))  # unique alerts
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs')
def logs():
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"error": "Log file not found."}), 404

    try:
        df = pd.read_csv(LOG_FILE_PATH, dtype=str)
        if 'Result' not in df.columns:
            df['Result'] = ""
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download')
def download():
    if not os.path.exists(LOG_FILE_PATH):
        return jsonify({"error": "Log file not found."}), 404
    return send_file(LOG_FILE_PATH, as_attachment=True)

app.register_blueprint(client_bp, url_prefix="/api")
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
