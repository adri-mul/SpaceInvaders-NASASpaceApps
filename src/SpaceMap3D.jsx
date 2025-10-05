// src/SpaceMap3D.jsx
import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import starsData from "./stars.json";
import planetsData from "./planets_for_ui.json";

const colors = ["#1E90FF", "#FFD700", "#FF4500", "#FF8C00", "#32CD32", "#800080"]; // star colors

const Star = ({ star, onClick, hoveredStar, setHoveredStar }) => {
  const mesh = useRef();

  const color = useMemo(
    () => (star.isPlanet ? "#FFFFFF" : colors[Math.floor(Math.random() * colors.length)]),
    [star.isPlanet]
  );

  const size = useMemo(() => (star.isPlanet ? 0.8 : 0.5 + Math.random() * 0.5), [star]);

  // Spin for 3D effect
  useFrame(() => {
    if (mesh.current) mesh.current.rotation.y += 0.002;
  });

  const isHovered = hoveredStar === star.id;

  return (
    <mesh
      ref={mesh}
      position={star.position}
      onClick={(e) => {
        e.stopPropagation();
        if (!star.isPlanet) onClick(star);
      }}
      onPointerOver={() => setHoveredStar(star.id)}
      onPointerOut={() => setHoveredStar(null)}
    >
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial color={color} />
      {isHovered && (
        <Html center distanceFactor={10}>
          <div
            style={{
              color: "white",
              fontSize: "40px",
              fontWeight: "bold",
              pointerEvents: "none",
              textShadow: "0 0 5px black",
              textAlign: "center",
            }}
          >
            {star.display}
          </div>
        </Html>
      )}
    </mesh>
  );
};

export default function SpaceMap3D() {
  const [stars, setStars] = useState([]);
  const [selected, setSelected] = useState(null);
  const [hoveredStar, setHoveredStar] = useState(null);
  const controlsRef = useRef();

  useEffect(() => {
    const radius = 50;

    const positionedStars = starsData
      .slice(0, 1000)
      .map((star) => {
        const phi = Math.acos(2 * Math.random() - 1);
        const theta = 2 * Math.PI * Math.random();
        const r = radius * (0.7 + 0.3 * Math.random());
        return {
          ...star,
          position: [r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi)],
          isPlanet: false,
        };
      });

    const positionedPlanets = planetsData.map((planet) => {
      const phi = Math.acos(2 * Math.random() - 1);
      const theta = 2 * Math.PI * Math.random();
      const r = radius * (0.7 + 0.3 * Math.random());
      return {
        ...planet,
        position: [r * Math.sin(phi) * Math.cos(theta), r * Math.sin(phi) * Math.sin(theta), r * Math.cos(phi)],
        isPlanet: true,
      };
    });

    setStars([...positionedStars, ...positionedPlanets]);
  }, []);

  const handleCenter = () => {
    if (controlsRef.current) controlsRef.current.reset();
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

      <Canvas camera={{ position: [0, 0, 60], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 0, 0]} intensity={1} />
        <OrbitControls ref={controlsRef} enablePan enableZoom enableRotate />
        {stars.map((star) => (
          <Star
            key={star.id}
            star={star}
            onClick={setSelected}
            hoveredStar={hoveredStar}
            setHoveredStar={setHoveredStar}
          />
        ))}
      </Canvas>
    </div>
  );
}
