// src/components/ClientDetails.jsx
import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import Plot from "react-plotly.js";
import "../styles/ClientDetails.css";

const ClientDetails = () => {
  const { clientId } = useParams();
  const [clientData, setClientData] = useState(null);

  useEffect(() => {
    fetch(`/api/client/${clientId}`)
      .then((res) => res.json())
      .then((data) => setClientData(data))
      .catch((err) => console.error("Error fetching client details:", err));
  }, [clientId]);

  if (!clientData) {
    return <p>Loading client details...</p>;
  }

  return (
    <div className="client-details">
      <h1>Client {clientId} Details</h1>

      <div className="card">
        <p><strong>Total Queries:</strong> {clientData.total_queries}</p>
        <p><strong>Slow Queries:</strong> {clientData.slow_queries}</p>
      </div>

      <div className="card chart-container">
        <Plot
          data={[
            {
              labels: ["DB Changes", "View Only"],
              values: [clientData.change_count, clientData.view_count],
              type: "pie",
            },
          ]}
          layout={{
            title: "Change vs View Queries",
            paper_bgcolor: "#1e1e1e",
            font: { color: "#e0e0e0" },
          }}
          style={{ width: "400px", height: "350px" }}
        />
      </div>

      <h2>Executed Queries (most recent first)</h2>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Query</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {clientData.queries.map((q, index) => (
              <tr key={index}>
                <td>{q.Timestamp}</td>
                <td>
                  <pre style={{ whiteSpace: "pre-wrap", color: "#e0e0e0" }}>
                    {q.Query}
                  </pre>
                </td>
                <td>
                  <pre style={{ whiteSpace: "pre-wrap", color: "#e0e0e0" }}>
                    {q.Result}
                  </pre>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p>
        <Link to="/">Back to Dashboard</Link>
      </p>
    </div>
  );
};

export default ClientDetails;
