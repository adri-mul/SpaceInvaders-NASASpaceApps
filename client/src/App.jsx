import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [log, setLog] = useState("");

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleRunPipeline = async () => {
    if (!file) return alert("Select a SH file first");

    const form = new FormData();
    form.append("shFile", file);

    setLog("Starting pipeline...\n");

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        body: form,
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        setLog((prev) => prev + decoder.decode(value));
      }
    } catch (err) {
      console.error(err);
      setLog((prev) => prev + "\nError: " + err.message);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <h1>Space Invaders — SH → Predictions</h1>
        <button onClick={handleRunPipeline}>Run Pipeline</button>
      </div>

      <input type="file" accept=".sh" onChange={handleFileChange} />

      <div
        style={{
          marginTop: 20,
          height: "500px",
          background: "#000",
          color: "#0f0",
          fontFamily: "monospace",
          padding: 10,
          overflow: "auto",
        }}
      >
        <pre>{log}</pre>
      </div>
    </div>
  );
}

export default App;
