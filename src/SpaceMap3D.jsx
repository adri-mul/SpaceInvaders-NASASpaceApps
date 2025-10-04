// src/SpaceMap3D.jsx
import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import starsData from "./stars.json";

const Star = ({ star, onClick }) => {
  const mesh = useRef();
  const color = useMemo(() => new THREE.Color(Math.random(), Math.random(), Math.random()), []);

  return (
    <mesh
      ref={mesh}
      position={star.position}
      onClick={(e) => {
        e.stopPropagation();
        onClick(star);
      }}
    >
      <sphereGeometry args={[0.5, 16, 16]} />
      <meshStandardMaterial color={color} />
    </mesh>
  );
};

export default function SpaceMap3D() {
  const [stars, setStars] = useState([]);
  const [selected, setSelected] = useState(null);
  const controlsRef = useRef();

  useEffect(() => {
    // Distribute stars in a sphere around origin
    const radius = 50;
    const positionedStars = starsData.map((star) => {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = radius * (0.7 + 0.3 * Math.random()); // slightly random distance
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      return { ...star, position: [x, y, z] };
    });
    setStars(positionedStars);
  }, []);

  const handleCenter = () => {
    if (controlsRef.current) {
      controlsRef.current.reset();
    }
  };

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative" }}>
      <button
        onClick={handleCenter}
        style={{
          position: "absolute",
          top: 20,
          left: 20,
          zIndex: 10,
          padding: "10px 20px",
          background: "#1e40af",
          color: "white",
          borderRadius: "8px",
          border: "none",
          cursor: "pointer",
          fontWeight: "bold",
        }}
      >
        Center
      </button>

      {selected && (
        <div
          style={{
            position: "absolute",
            top: "50px",
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(0,0,0,0.85)",
            color: "white",
            padding: "16px",
            borderRadius: "12px",
            maxWidth: "400px",
            zIndex: 10,
          }}
        >
          <h2 style={{ margin: 0, marginBottom: "8px" }}>{selected.display}</h2>
          <p style={{ margin: 0 }}>{selected.snippet}</p>
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

      <Canvas camera={{ position: [0, 0, 0], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 0, 0]} intensity={1} />
        <OrbitControls ref={controlsRef} enablePan enableZoom enableRotate />
        {stars.map((star) => (
          <Star key={star.id} star={star} onClick={setSelected} />
        ))}
      </Canvas>
    </div>
  );
}
