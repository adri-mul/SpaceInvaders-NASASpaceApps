// src/MapPage.jsx
import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import starsData from "./stars.json";

const colors = ["#1E90FF", "#FFD700", "#FF4500", "#FF8C00", "#32CD32", "#800080"];

/**
 * HalfGreenBlueMaterial
 * Simple shader material that colors the sphere green for positive Y and blue for negative Y.
 */
const HalfGreenBlueMaterial = () => {
  const shaderRef = useRef();
  const uniforms = useMemo(
    () => ({
      color1: { value: new THREE.Color("#28FF7A") }, // greenish
      color2: { value: new THREE.Color("#3B82F6") }, // blueish
    }),
    []
  );

  const vertexShader = `
    varying vec3 vPosition;
    void main() {
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const fragmentShader = `
    varying vec3 vPosition;
    uniform vec3 color1;
    uniform vec3 color2;
    void main() {
      if (vPosition.y > 0.0) {
        gl_FragColor = vec4(color1, 1.0);
      } else {
        gl_FragColor = vec4(color2, 1.0);
      }
    }
  `;

  // react-three accepts shaderMaterial directly as JSX
  return (
    <shaderMaterial
      ref={shaderRef}
      uniforms={uniforms}
      vertexShader={vertexShader}
      fragmentShader={fragmentShader}
    />
  );
};

/**
 * Star component - renders a sphere at star.position
 * - isSpecial: uses HalfGreenBlueMaterial
 * - isRandom or isPlanet -> white color
 * - otherwise random color picked from colors[]
 */
const Star = ({ star, hoveredStar, setHoveredStar, onClick, index }) => {
  const mesh = useRef();
  const isSpecial = !!star.isSpecial;

  // color deterministic-ish per star id so re-renders keep same color
  const color = useMemo(() => {
    if (star.isPlanet || star.isRandom) return "#FFFFFF";
    // simple hash of id to pick color deterministically
    let h = 0;
    for (let i = 0; i < (star.id || "").length; i++) {
      h = (h << 5) - h + star.id.charCodeAt(i);
      h |= 0;
    }
    const idx = Math.abs(h) % colors.length;
    return colors[idx];
  }, [star]);

  const size = useMemo(() => {
    if (star.isPlanet || star.isRandom) return 0.75;
    if (star.isSpecial) return 1.0;
    return 0.6 + (Math.abs((star.id || "").length) % 4) * 0.1;
  }, [star]);

  useFrame(() => {
    if (mesh.current) mesh.current.rotation.y += 0.002;
  });

  return (
    <mesh
      ref={mesh}
      position={star.position}
      onClick={(e) => {
        e.stopPropagation();
        // only non-random stars show details
        if (!star.isRandom) onClick(star);
      }}
      onPointerOver={() => setHoveredStar(index)}
      onPointerOut={() => setHoveredStar(null)}
    >
      <sphereGeometry args={[size, 16, 16]} />
      {isSpecial ? <HalfGreenBlueMaterial /> : <meshStandardMaterial color={color} />}
      {hoveredStar === index && (
        <Html distanceFactor={10}>
          <div
            style={{
              color: "white",
              fontSize: "18px",
              fontWeight: 700,
              pointerEvents: "none",
              textAlign: "center",
              textShadow: "0 0 6px black",
              padding: "2px 6px",
              borderRadius: 6,
              background: "rgba(0,0,0,0.5)",
            }}
          >
            {star.display}
          </div>
        </Html>
      )}
    </mesh>
  );
};

/**
 * MapPage - main component
 * - renders 1000 real stars from stars.json
 * - renders 500 random white stars (isRandom)
 * - renders 3 special half-green/half-blue stars
 * - shows a top-right dummy pipeline control (non-functional, simulates logs)
 * - center button in top-left
 * - clicking a non-random star opens details with Close button
 */
export default function MapPage() {
  const [stars, setStars] = useState([]);
  const [planets, setPlanets] = useState([]); // kept for future use
  const [selected, setSelected] = useState(null);
  const [hoveredStar, setHoveredStar] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [busy, setBusy] = useState(false);

  const shRef = useRef(null);
  const controlsRef = useRef();

  // Initialize: 1000 stars + 500 random + 3 special
  useEffect(() => {
    const radius = 50;

    const positionedStars = starsData.slice(0, 1000).map((star, i) => {
      // spherical random distribution
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = radius * (0.7 + 0.3 * Math.random());
      return {
        ...star,
        display: star.display || star.name || `Star-${i}`,
        snippet: star.snippet || star.description || null,
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        ],
        isPlanet: false,
        isRandom: false,
        isSpecial: false,
        id: `star-${i}`,
      };
    });

    const randomStars = Array.from({ length: 500 }, (_, i) => {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = radius * (0.7 + 0.3 * Math.random());
      const randId = Math.floor(Math.random() * 2 ** 32);
      return {
        display: `Star-${randId}`,
        snippet: null,
        position: [
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        ],
        isPlanet: false,
        isRandom: true,
        isSpecial: false,
        id: `rand-${i}`,
      };
    });

    // 3 special (half green/half blue)
    const specialStars = Array.from({ length: 3 }, (_, i) => {
      // place them around a ring so they are somewhat discoverable
      const angle = (i / 3) * Math.PI * 2;
      const r = radius * 0.45;
      const x = r * Math.cos(angle);
      const y = (Math.random() - 0.5) * 10; // slight Y offset so shader shows halves
      const z = r * Math.sin(angle);
      return {
        display: `Habitable-${i + 1}`,
        snippet: `Habitable candidate #${i + 1}.`,
        position: [x, y, z],
        isPlanet: false,
        isRandom: false,
        isSpecial: true,
        id: `special-${i}`,
      };
    });

    setStars([...positionedStars, ...randomStars, ...specialStars]);
  }, []);

  const handleCenter = () => {
    if (controlsRef.current) controlsRef.current.reset();
  };

  const appendLog = (line) => {
    setLogLines((prev) => {
      const next = [...prev, line];
      // keep logs reasonable length
      if (next.length > 200) next.splice(0, next.length - 200);
      return next;
    });
  };

  // Dummy pipeline runner (non-functional). Simulates progress output to logs.
  const runDummyPipeline = async () => {
    if (busy) return;
    setBusy(true);
    setLogLines([]);
    appendLog("🚀 pipeline started");
    appendLog("📤 Uploading SH file...");
    await new Promise((r) => setTimeout(r, 600));
    appendLog("🔎 Parsing .sh list (simulated)...");
    await new Promise((r) => setTimeout(r, 700));
    appendLog("⬇️ Downloading FITS (simulated)...");
    for (let i = 1; i <= 6; i++) {
      appendLog(`  • fetched file ${i}/6`);
      // small delay to simulate streaming output
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, 400));
    }
    appendLog("⚙️ Extracting features (simulated)...");
    await new Promise((r) => setTimeout(r, 1000));
    appendLog("🤖 Scoring with model (simulated)...");
    await new Promise((r) => setTimeout(r, 1000));
    appendLog("✅ Predictions saved to tesscurl_sector_96_lc_predictions.json (simulated)");
    appendLog("🎉 pipeline finished");
    setBusy(false);
  };

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", background: "#000" }}>
      {/* Top-left: Center button */}
      <button
        onClick={handleCenter}
        style={{
          position: "absolute",
          top: 20,
          left: 20,
          zIndex: 20,
          padding: "10px 14px",
          background: "#0ea5e9",
          color: "white",
          borderRadius: 8,
          border: "none",
          cursor: "pointer",
          fontWeight: 700,
        }}
        title="Reset camera"
      >
        Center
      </button>

      {/* Top-right: Dummy pipeline panel */}
      <div
        style={{
          position: "absolute",
          top: 20,
          right: 20,
          zIndex: 20,
          width: 320,
          background: "linear-gradient(180deg, rgba(17,24,39,0.92), rgba(6,8,14,0.85))",
          borderRadius: 12,
          padding: 12,
          boxShadow: "0 10px 30px rgba(0,0,0,0.6)",
          color: "#e6eef8",
          fontFamily: "Inter, system-ui, sans-serif",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 800 }}>Pipeline</div>
            <div style={{ fontSize: 12, color: "#9ca3af" }}>Run SH → Predictions</div>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <input
              ref={shRef}
              type="file"
              accept=".sh"
              disabled={busy}
              style={{
                display: "inline-block",
                fontSize: 12,
                padding: "6px 8px",
                borderRadius: 8,
                background: "#0b1220",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.06)",
              }}
              title="(Optional) pick an .sh for simulation"
            />
            <button
              onClick={runDummyPipeline}
              disabled={busy}
              style={{
                padding: "8px 10px",
                background: busy ? "#475569" : "#10b981",
                color: "white",
                border: "none",
                borderRadius: 8,
                cursor: busy ? "not-allowed" : "pointer",
                fontWeight: 700,
              }}
            >
              {busy ? "Running…" : "Run Pipeline"}
            </button>
          </div>
        </div>

        <textarea
          value={logLines.join("\n")}
          readOnly
          rows={6}
          style={{
            marginTop: 10,
            width: "100%",
            background: "rgba(2,6,23,0.6)",
            color: "#e6eef8",
            border: "none",
            borderRadius: 8,
            padding: 8,
            fontSize: 12,
            fontFamily: "monospace",
            resize: "none",
          }}
        />
      </div>

      {/* Selected star/planet info card */}
      {selected && (
        <div
          style={{
            position: "absolute",
            top: "calc(50% - 160px)",
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 20,
            width: 420,
            background: "rgba(2,6,23,0.95)",
            color: "white",
            borderRadius: 12,
            padding: 16,
            boxShadow: "0 20px 50px rgba(0,0,0,0.7)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", gap: 12 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: 20 }}>{selected.display}</h2>
              <div style={{ color: "#9ca3af", marginTop: 6, fontSize: 13 }}>
                {selected.snippet || "No description available for this object."}
              </div>
            </div>
            <div style={{ textAlign: "right" }}>
              <button
                onClick={() => setSelected(null)}
                style={{
                  padding: "8px 12px",
                  background: "#ef4444",
                  color: "white",
                  border: "none",
                  borderRadius: 8,
                  cursor: "pointer",
                  fontWeight: 700,
                }}
              >
                Close
              </button>
            </div>
          </div>
          {/* Optional extra metadata area */}
          <div style={{ marginTop: 12, display: "flex", gap: 12, color: "#cbd5e1", fontSize: 13 }}>
            <div>Type: {selected.isSpecial ? "Habitable Candidate" : selected.isRandom ? "Random" : "Catalog"}</div>
            <div>{selected.id ? `ID: ${selected.id}` : null}</div>
          </div>
        </div>
      )}

      {/* 3D canvas */}
      <Canvas camera={{ position: [0, 0, 60], fov: 75 }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[0, 0, 0]} intensity={1.2} />
        <OrbitControls ref={controlsRef} enablePan enableZoom enableRotate />
        {stars.map((star, i) => (
          <Star
            key={star.id ?? `s-${i}`}
            star={star}
            index={i}
            hoveredStar={hoveredStar}
            setHoveredStar={setHoveredStar}
            onClick={setSelected}
          />
        ))}
      </Canvas>
    </div>
  );
}
