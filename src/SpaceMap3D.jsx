// src/SpaceMap3D.jsx
import React, { useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars as DreiStars } from "@react-three/drei";
import * as THREE from "three";
import starsData from "./stars.json";

function ForegroundStar({ s, onClick }) {
  return (
    <mesh position={s.position} onClick={(e) => { e.stopPropagation(); onClick(s); }}>
      <sphereGeometry args={[s.size, 16, 16]} />
      <meshStandardMaterial color={s.color} emissive={s.color} emissiveIntensity={0.25} />
    </mesh>
  );
}

export default function SpaceMap3D({ maxStars = 400 }) {
  const [selected, setSelected] = useState(null);
  const controlsRef = useRef();

  // generate star positions/colors once
  const foregroundStars = useMemo(() => {
    const arr = [...starsData];
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    const chosen = arr.slice(0, Math.min(maxStars, arr.length));

    const radiusMin = 60;
    const radiusMax = 160;
    return chosen.map((star, idx) => {
      const u = Math.random();
      const v = Math.random();
      const theta = 2 * Math.PI * u;
      const phi = Math.acos(2 * v - 1);
      const r = radiusMin + Math.random() * (radiusMax - radiusMin);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);

      const hue = (idx * 47) % 360;
      const color = new THREE.Color(`hsl(${hue}, 80%, ${60 - (idx % 8)}%)`).getStyle();
      const size = 0.8 + (idx % 6) * 0.18;

      return { ...star, position: [x, y, z], color, size, id: star.id || `s-${idx}` };
    });
  }, [maxStars]);

  const handleCenter = () => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    const cam = controls.object;
    cam.position.set(0, 0, 220);
    controls.target.set(0, 0, 0);
    controls.update();
    setSelected(null);
  };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#000" }}>
      {/* Center button */}
      <div style={{ position: "absolute", zIndex: 20, left: 16, top: 16 }}>
        <button
          onClick={handleCenter}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            background: "linear-gradient(90deg,#0369a1,#0891b2)",
            color: "white",
            border: "none",
            cursor: "pointer",
            fontWeight: 700,
            boxShadow: "0 6px 18px rgba(2,6,23,0.6)",
          }}
        >
          Center
        </button>
      </div>

      {/* Info box overlay (fixed on screen) */}
      {selected && (
        <div
          style={{
            position: "absolute",
            top: "60px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(0,0,0,0.85)",
            color: "white",
            padding: "16px",
            borderRadius: "12px",
            maxWidth: "400px",
            zIndex: 20,
          }}
        >
          <h2 style={{ margin: 0, marginBottom: "8px" }}>{selected.display}</h2>
          <p style={{ margin: 0 }}>{selected.snippet || "No description available."}</p>
          {selected.url && (
            <a
              href={selected.url}
              target="_blank"
              rel="noreferrer"
              style={{ color: "#38bdf8", textDecoration: "underline" }}
            >
              Learn more
            </a>
          )}
          <div
            onClick={() => setSelected(null)}
            style={{
              marginTop: "8px",
              textAlign: "right",
              cursor: "pointer",
              color: "#f87171",
              fontWeight: "bold",
            }}
          >
            Close
          </div>
        </div>
      )}

      {/* 3D canvas */}
      <Canvas camera={{ position: [0, 0, 220], fov: 60 }}>
        <DreiStars radius={400} depth={200} count={12000} factor={4} fade />
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 80, 50]} intensity={0.8} />
        <group>
          {foregroundStars.map((s) => (
            <ForegroundStar key={s.id} s={s} onClick={setSelected} />
          ))}
        </group>
        <OrbitControls ref={controlsRef} enablePan enableZoom enableRotate />
      </Canvas>
    </div>
  );
}
