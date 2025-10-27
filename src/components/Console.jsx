// import React, { useState, useEffect, useRef } from "react";

// function ClientConsole() {
//   const [clientId, setClientId] = useState("");
//   const [connected, setConnected] = useState(false);
//   const [query, setQuery] = useState("");
//   const [messages, setMessages] = useState([]);
//   const wsRef = useRef(null);

//   const SERVER_HOST = "wss://10.208.18.195:5050"; // placeholder, will handle via backend proxy

//   const appendMessage = (msg) => {
//     setMessages((prev) => [...prev, msg]);
//   };

//   const handleConnect = async () => {
//     if (!clientId) return alert("Enter Client ID");

//     try {
//       // Connect via your backend proxy API
//       const res = await fetch("/api/connect", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ clientId }),
//       });
//       const data = await res.json();

//       if (data.status === "connected") {
//         setConnected(true);
//         appendMessage("Connected to server: " + data.message);
//       } else {
//         appendMessage("Connection failed: " + data.message);
//       }
//     } catch (err) {
//       appendMessage("Error connecting: " + err.message);
//     }
//   };

//   const handleExecute = async () => {
//     if (!query) return;
//     try {
//       const res = await fetch("/api/execute_query", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ clientId, query }),
//       });
//       const data = await res.json();

//       appendMessage("Query: " + query);
//       appendMessage("Response: " + data.response);
//       if (data.ranking) appendMessage("Ranking: " + JSON.stringify(data.ranking));
//     } catch (err) {
//       appendMessage("Error executing query: " + err.message);
//     }
//     setQuery("");
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>DB Client Console</h2>
//       {!connected ? (
//         <>
//           <input
//             type="text"
//             placeholder="Enter Client ID"
//             value={clientId}
//             onChange={(e) => setClientId(e.target.value)}
//           />
//           <button onClick={handleConnect}>Connect</button>
//         </>
//       ) : (
//         <>
//           <div>
//             <input
//               type="text"
//               placeholder="Enter SQL query"
//               value={query}
//               onChange={(e) => setQuery(e.target.value)}
//               style={{ width: "60%" }}
//             />
//             <button onClick={handleExecute}>Execute</button>
//           </div>
//           <div
//             style={{
//               marginTop: "20px",
//               background: "#000",
//               color: "#0f0",
//               height: "400px",
//               overflowY: "scroll",
//               padding: "10px",
//               fontFamily: "monospace",
//             }}
//           >
//             {messages.map((msg, idx) => (
//               <div key={idx}>{msg}</div>
//             ))}
//           </div>
//         </>
//       )}
//     </div>
//   );
// }

// export default ClientConsole;

import React, { useState } from "react";

function ClientConsole() {
  const [clientId, setClientId] = useState("");
  const [connected, setConnected] = useState(false);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);

  const appendMessage = (msg) => {
    setMessages((prev) => [...prev, msg]);
  };

  //   const handleConnect = async () => {
  //     if (!clientId) return alert("Enter Client ID");

  //     try {
  //       const res = await fetch("/api/connect", {
  //         method: "POST",
  //         headers: { "Content-Type": "application/json" },
  //         body: JSON.stringify({ clientId }),
  //       });

  //       const data = await res.json();

  //       if (data.status === "connected") {
  //         setConnected(true);
  //         appendMessage("Connected to server: " + data.message);
  //       } else {
  //         appendMessage("Connection failed: " + data.message);
  //       }
  //     } catch (err) {
  //       appendMessage("Error connecting: " + err.message);
  //     }
  //   };
  const handleConnect = async () => {
    if (!clientId) return alert("Enter Client ID");

    try {
      const res = await fetch("/api/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clientId }),
      });

      const data = await res.json();

      if (data.status === "connected") {
        setConnected(true);
        appendMessage("Connected to server: " + data.message);
      } else {
        appendMessage("Connection failed: " + data.message);
      }
    } catch (err) {
      appendMessage("Error connecting: " + err.message);
    }
  };

  const handleExecute = async () => {
    if (!query) return;
    try {
      const res = await fetch("/api/execute_query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clientId, query }),
      });
      const data = await res.json();

      // Show query and response
      appendMessage("Query: " + query);
      appendMessage("Response: " + data.response);

      // Show root cause ranking
      if (
        data.ranking &&
        Array.isArray(data.ranking) &&
        data.ranking.length > 0
      ) {
        appendMessage("----- Root Cause Ranking -----");
        data.ranking.forEach((item, idx) => {
          appendMessage(
            `${idx + 1}. Cause: ${item.cause}, Score: ${item.score.toFixed(2)}`
          );
        });
        appendMessage("-----------------------------");
      }
    } catch (err) {
      appendMessage("Error executing query: " + err.message);
    }
    setQuery("");
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>DB Client Console</h2>
      {!connected ? (
        <>
          <input
            type="text"
            placeholder="Enter Client ID"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
          />
          <button onClick={handleConnect}>Connect</button>
        </>
      ) : (
        <>
          <div>
            <input
              type="text"
              placeholder="Enter SQL query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{ width: "60%" }}
            />
            <button onClick={handleExecute}>Execute</button>
          </div>
          <div
            style={{
              marginTop: "20px",
              background: "#000",
              color: "#0f0",
              height: "400px",
              overflowY: "scroll",
              padding: "10px",
              fontFamily: "monospace",
            }}
          >
            {messages.map((msg, idx) => (
              <div key={idx}>{msg}</div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default ClientConsole;
