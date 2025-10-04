import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import starsDataRaw from "./stars.json";

export default function SpaceMap3D() {
  const [selectedStar, setSelectedStar] = useState(null);
  const [starInfoCache, setStarInfoCache] = useState({});
  const [loadingInfo, setLoadingInfo] = useState(null);
  const [cameraZoom, setCameraZoom] = useState(12);

  // Convert starsDataRaw to include positions and a "brightness" proxy
  const starsData = useMemo(() => {
    return starsDataRaw.map((s, i) => ({
      ...s,
      id: `${s.source}-${s.id}`,
      position: [
        (Math.random() - 0.5) * 200,
        (Math.random() - 0.5) * 200,
        (Math.random() - 0.5) * 200,
      ],
      brightness: Math.random(), // simple proxy; could be radius or magnitude
    }));
  }, []);

  // Compute which stars are currently "in view" and pick top 1000 brightest
  const visibleStars = useMemo(() => {
    const { camera } = { camera: { position: [0, 0, cameraZoom] } }; // placeholder
    // For simplicity, pick top 1000 brightest globally
    return [...starsData]
      .sort((a, b) => b.brightness - a.brightness)
      .slice(0, 1000);
  }, [starsData, cameraZoom]);

  // SerpAPI fetch for star description
  const fetchStarInfo = async (star) => {
    if (starInfoCache[star.id]?.description) return;
    setLoadingInfo(star.id);
    try {
      const res = await fetch(`/api/starinfo?q=${encodeURIComponent(star.search_query)}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      const next = { ...starInfoCache, [star.id]: data };
      setStarInfoCache(next);
      localStorage.setItem("starinfo-cache", JSON.stringify(next));
    } catch {
      const next = {
        ...starInfoCache,
        [star.id]: { title: star.display, description: "No description found", url: null },
      };
      setStarInfoCache(next);
      localStorage.setItem("starinfo-cache", JSON.stringify(next));
    } finally {
      setLoadingInfo(null);
    }
  };

  return (
    <div className="w-full h-screen bg-black relative">
      <Canvas camera={{ position: [0, 0, cameraZoom], fov: 55 }}>
        <color attach="background" args={["#000"]} />
        <BasicControls cameraZoom={cameraZoom} setCameraZoom={setCameraZoom} />

        <group>
          {visibleStars.map((star) => (
            <Planet
              key={star.id}
              star={star}
              onClick={() => {
                setSelectedStar(star);
                fetchStarInfo(star);
              }}
            />
          ))}
        </group>
      </Canvas>

      {selectedStar && (
        <div className="absolute bottom-4 left-4 bg-white/10 text-black p-4 rounded-xl max-w-sm">
          <div className="font-semibold">{selectedStar.display}</div>
          <div className="text-sm mt-1">
            {starInfoCache[selectedStar.id]?.description ||
              "Loading description…"}
          </div>
          {starInfoCache[selectedStar.id]?.url && (
            <a
              href={starInfoCache[selectedStar.id].url}
              target="_blank"
              rel="noreferrer"
              className="text-blue-500 underline text-xs"
            >
              Learn more
            </a>
          )}
          <button
            className="mt-2 px-2 py-1 bg-white/20 rounded"
            onClick={() => setSelectedStar(null)}
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}

function Planet({ star, onClick }) {
  const ref = useRef();
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.1;
  });
  const color = `hsl(${star.brightness * 60}, 100%, 50%)`; // color based on brightness/heat
  const radius = 0.5 + star.brightness * 1.5;

  return (
    <mesh ref={ref} position={star.position} onClick={onClick}>
      <sphereGeometry args={[radius, 16, 16]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

// Simple orbit-style controls
function BasicControls({ cameraZoom, setCameraZoom }) {
  const { camera, gl } = useThree();
  const spherical = useRef(new THREE.Spherical(cameraZoom, Math.PI / 3, 0));
  const isDragging = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  useFrame(() => {
    const pos = new THREE.Vector3().setFromSpherical(spherical.current);
    camera.position.lerp(pos, 0.1);
    camera.lookAt(0, 0, 0);
  });

  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e) => {
      isDragging.current = true;
      last.current = { x: e.clientX, y: e.clientY };
    };
    const onMove = (e) => {
      if (!isDragging.current) return;
      const dx = e.clientX - last.current.x;
      const dy = e.clientY - last.current.y;
      last.current = { x: e.clientX, y: e.clientY };
      spherical.current.theta -= dx * 0.005;
      spherical.current.phi = THREE.MathUtils.clamp(
        spherical.current.phi + dy * 0.005,
        0.1,
        Math.PI - 0.1
      );
    };
    const onUp = () => {
      isDragging.current = false;
    };
    const onWheel = (e) => {
      spherical.current.radius = THREE.MathUtils.clamp(
        spherical.current.radius + e.deltaY * 0.01,
        5,
        60
      );
      setCameraZoom(spherical.current.radius);
    };

    el.addEventListener("pointerdown", onDown);
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    el.addEventListener("wheel", onWheel, { passive: true });

    return () => {
      el.removeEventListener("pointerdown", onDown);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      el.removeEventListener("wheel", onWheel);
    };
  }, [gl, setCameraZoom]);

  return null;
}
