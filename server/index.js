const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");

const app = express();
app.use(cors()); // allow React frontend to call backend

const upload = multer({ dest: "uploads/" });

app.post("/api/run", upload.single("shFile"), (req, res) => {
  if (!req.file) return res.status(400).send("No SH file uploaded");

  const shPath = path.resolve(req.file.path);
  console.log(`Received SH file: ${shPath}`);

  // Run your Python script
  const py = spawn("python", ["Code/run_sh_to_predictions.py", shPath]);

  py.stdout.on("data", (data) => {
    console.log(data.toString());
  });

  py.stderr.on("data", (data) => {
    console.error(data.toString());
  });

  py.on("close", (code) => {
    console.log(`Python process exited with code ${code}`);
    res.send("Pipeline finished! Check server console for outputs.");
  });
});

app.listen(3001, () => console.log("Server listening on http://localhost:3001"));
