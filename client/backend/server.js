const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const port = 3001;

// Enable CORS
app.use(cors());

// Multer setup for file uploads
const uploadDir = path.join(__dirname, "uploads");
fs.mkdirSync(uploadDir, { recursive: true });
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage });

// POST endpoint to run pipeline
app.post("/api/run", upload.single("shFile"), (req, res) => {
  if (!req.file) return res.status(400).send("No SH file uploaded");

  const shFilePath = req.file.path;
  const pyScriptPath = path.join(__dirname, "../client/CODE/run_sh_to_predictions.py");

  if (!fs.existsSync(pyScriptPath))
    return res.status(500).send("Python pipeline script not found");

  // Spawn Python process
  const py = spawn("python", [pyScriptPath, "--sh", shFilePath]);

  // Stream stdout/stderr
  py.stdout.on("data", (data) => res.write(data.toString()));
  py.stderr.on("data", (data) => res.write(data.toString()));

  py.on("close", (code) => {
    res.end(`\nPython process exited with code ${code}`);
  });
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
});
