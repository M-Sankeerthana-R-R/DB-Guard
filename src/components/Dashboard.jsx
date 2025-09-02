import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "../styles/Dashboard.css"; // extracted styles

export default function Dashboard() {
  const [clientData, setClientData] = useState({});
  const [slowData, setSlowData] = useState({});
  const [hourlyData, setHourlyData] = useState({});
  const [totalClients, setTotalClients] = useState(0);

  useEffect(() => {
    // Fetch from Flask API (instead of Jinja template variables)
    fetch("/api/dashboard")
      .then((res) => res.json())
      .then((data) => {
        setClientData(data.client_counts || {});
        setSlowData(data.slow_counts || {});
        setHourlyData(data.five_min_counts || {});
        setTotalClients(data.total_clients || 0);
      });
  }, []);

  const darkLayout = {
    paper_bgcolor: "#1e1e1e",
    plot_bgcolor: "#1e1e1e",
    font: { color: "#e0e0e0" },
  };

  return (
    <div className="dashboard">
      <h1>Database Activity Dashboard</h1>

      <div className="top-metrics">
        <div className="metric-card">
          <h3>Total Unique Clients Ever</h3>
          <p style={{ fontSize: "28px", margin: "8px 0" }}>{totalClients}</p>
        </div>
      </div>

      <div className="chart-row">
        <div className="chart-box">
          <Plot
            data={[
              {
                labels: Object.keys(clientData),
                values: Object.values(clientData),
                type: "pie",
                marker: {
                  colors: ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350", "#ab47bc"],
                },
              },
            ]}
            layout={{ title: "Client Query Distribution", ...darkLayout }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>

        <div className="chart-box">
          <Plot
            data={[
              {
                labels: Object.keys(slowData),
                values: Object.values(slowData),
                type: "pie",
                marker: { colors: ["#ff7043", "#26c6da"] },
              },
            ]}
            layout={{ title: "Slow vs Fast Queries", ...darkLayout }}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
      </div>

      <div className="time-chart">
        <Plot
          data={[
            {
              x: Object.keys(hourlyData),
              y: Object.values(hourlyData),
              type: "scatter",
              mode: "lines+markers",
              line: { color: "#90caf9" },
            },
          ]}
          layout={{ title: "Query Activity Over Time", ...darkLayout }}
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      <div className="links">
        <a href="/logs">View Logs</a> | <a href="/download">Download CSV</a>
      </div>
    </div>
  );
}
