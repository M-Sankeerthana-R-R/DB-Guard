// src/components/Logs.jsx
import React, { useEffect, useState } from "react";
import "../styles/Logs.css";

const Logs = () => {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    fetch("/api/logs")
      .then((res) => res.json())
      .then((data) => setLogs(data || []))
      .catch((err) => console.error("Error fetching logs:", err));
  }, []);

  return (
    <div className="logs-container">
      <h1>Activity Logs</h1>
      <p>
        <a href="/">Back to Dashboard</a> |{" "}
        <a href="/download">Download CSV</a>
      </p>

      <table className="logs-table">
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>ClientID</th>
            <th>QueryType</th>
            <th>Query</th>
            <th>ExecutionTime</th>
            <th>SlowQuery</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {logs && logs.length > 0 ? (
            logs.map((row, index) => (
              <tr key={index}>
                <td>{row.Timestamp}</td>
                <td>{row.ClientID}</td>
                <td>{row.QueryType}</td>
                <td><pre>{row.Query}</pre></td>
                <td>{row.ExecutionTime}</td>
                <td>{row.SlowQuery ? "Yes" : "No"}</td>
                <td><pre>{row.Result}</pre></td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="7">No logs available</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default Logs;
