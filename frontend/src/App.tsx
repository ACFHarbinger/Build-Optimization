import { HashRouter, Route, Routes } from "react-router-dom";
import { Sidebar } from "./components/Sidebar";
import { BuildExplorer } from "./pages/BuildExplorer";
import { SolverComparison } from "./pages/SolverComparison";
import { TrainingMonitor } from "./pages/TrainingMonitor";
import { ItemDatabase } from "./pages/ItemDatabase";
import { Advisor } from "./pages/Advisor";

function App() {
  return (
    <HashRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<BuildExplorer />} />
            <Route path="/advisor" element={<Advisor />} />
            <Route path="/solver-comparison" element={<SolverComparison />} />
            <Route path="/training-monitor" element={<TrainingMonitor />} />
            <Route path="/item-database" element={<ItemDatabase />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

export default App;
