# monitor/activity_logger.py
import os
import csv
from datetime import datetime

LOG_FILE = os.path.join('logs', 'activity_log.csv')
SLOW_QUERY_THRESHOLD = 10.0  # seconds

def init_log_file():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # If file doesn't exist or is empty, write header including Result
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'ClientID', 'QueryType', 'Query',
                'ExecutionTime', 'SlowQuery', 'Result'
            ])

def log_activity(client_id, query, query_type, exec_time, result):
    """
    Logs an activity row. 'result' should be the exact text sent to the client.
    """
    init_log_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slow_query = exec_time > SLOW_QUERY_THRESHOLD

    # Ensure result is a string (so CSV won't fail)
    if result is None:
        result = ""
    else:
        result = str(result)

    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            client_id,
            query_type,
            query.strip(),
            round(exec_time, 2),
            slow_query,
            result
        ])
