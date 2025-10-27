// src/App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import ClientDetails from "./components/ClientDetails";
import Logs from "./components/Logs";
import "./App.css";
import ClientConsole from "./components/Console"
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/client/:clientId" element={<ClientDetails />} />
        <Route path="/logs" element={<Logs />} />
        <Route path="/console" element={<ClientConsole />} />
      </Routes>
    </Router>
  );
}

export default App;
// import { BrowserRouter, Routes, Route } from "react-router-dom";

// function Dashboard() {
//   return <h1>Hello Dashboard</h1>;
// }

// export default function App() {
//   return (
//     <BrowserRouter>
//       <Routes>
//         <Route path="/" element={<Dashboard />} />
//       </Routes>
//     </BrowserRouter>
//   );
// }
