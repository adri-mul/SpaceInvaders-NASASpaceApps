import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import starsData from "./stars.json";

function sphericalToXYZ(radius, theta, phi) {
  return [
    radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  ];
}

function Star({ star, onClick }) {
  const ref = useRef();
  useFrame(() => {
    if (ref.current) ref.current.rotation.y += 0.001;
  });
  return (
    <mesh ref={ref} position={star.position} onClick={() => onClick(star)} scale={[2, 2, 2]}>
      <sphereGeometry args={[0.8, 24, 24]} />
      <meshBasicMaterial color={star.color} />
    </mesh>
  );
}

function Controls({ radiusRef, defaultSpherical }) {
  const { camera, gl } = useThree();
  const spherical = useRef(defaultSpherical);
  const dragging = useRef(false);
  const last = useRef({ x: 0, y: 0 });

  useFrame(() => {
    const pos = new THREE.Vector3().setFromSpherical(spherical.current);
    camera.position.lerp(pos, 0.2);
    camera.lookAt(0, 0, 0);
  });

  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e) => { dragging.current = true; last.current = { x: e.clientX, y: e.clientY }; };
    const onMove = (e) => {
      if (!dragging.current) return;
      const dx = e.clientX - last.current.x;
      const dy = e.clientY - last.current.y;
      last.current = { x: e.clientX, y: e.clientY };
      spherical.current.theta -= dx * 0.005;
      spherical.current.phi = THREE.MathUtils.clamp(spherical.current.phi + dy * 0.005, 0.1, Math.PI - 0.1);
    };
    const onUp = () => (dragging.current = false);
    const onWheel = (e) => {
      spherical.current.radius = THREE.MathUtils.clamp(spherical.current.radius + e.deltaY * 0.1, 15, 200);
      radiusRef.current = spherical.current.radius;
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
  }, [gl, radiusRef]);

  return null;
}

export default function SpaceMap3D() {
  const [activeStar, setActiveStar] = useState(null);
  const [starInfo, setStarInfo] = useState({});
  const [loading, setLoading] = useState(null);
  const radiusRef = useRef(50);
  const defaultSpherical = new THREE.Spherical(radiusRef.current, Math.PI / 2, 0);

  const stars = useMemo(() => {
    return starsData.slice(0, 1000).map((s) => {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const radius = 40 + Math.random() * 60;
      const position = sphericalToXYZ(radius, theta, phi);
      const color = `hsl(${Math.random() * 360}, 80%, 60%)`;
      return { ...s, position, color };
    });
  }, []);

  const fetchStarInfo = async (star) => {
    if (starInfo[star.display]?.description) return;
    setLoading(star.display);
    try {
      const r = await fetch(`/api/starinfo?q=${encodeURIComponent(star.search_query)}`);
      const data = await r.json();
      const description = data.description || `Search for "${star.display}" online.`;
      setStarInfo((prev) => ({ ...prev, [star.display]: { ...data, description } }));
    } catch {
      setStarInfo((prev) => ({
        ...prev,
        [star.display]: { title: star.display, description: `Search for "${star.display}" online.`, url: null },
      }));
    } finally {
      setLoading(null);
    }
  };

  const centerView = () => {
    defaultSpherical.radius = 50;
    defaultSpherical.theta = Math.PI / 2;
    defaultSpherical.phi = 0;
    radiusRef.current = defaultSpherical.radius;
  };

  return (
    <div className="w-full h-screen relative bg-black text-white">
      <Canvas camera={{ position: [0, 0, radiusRef.current], fov: 70 }}>
        <color attach="background" args={["#000"]} />
        <Controls radiusRef={radiusRef} defaultSpherical={defaultSpherical} />
        {stars.map((s, i) => (
          <Star key={i} star={s} onClick={(star) => { setActiveStar(star); fetchStarInfo(star); }} />
        ))}
      </Canvas>

      {/* Center Button */}
      <button
        onClick={centerView}
        className="absolute top-4 left-4 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold shadow-lg z-20"
      >
        Center
      </button>

      {/* Star Info Popup (always top-left HUD) */}
      {activeStar && (
        <div className="absolute top-16 left-4 w-96 p-4 bg-black/90 text-white rounded-xl border border-white/20 shadow-lg z-20">
          <div className="font-bold text-lg">{activeStar.display}</div>
          <div className="mt-2 text-sm">
            {starInfo[activeStar.display]?.description || (loading === activeStar.display ? "Loading…" : "Fetching info…")}
          </div>
          {starInfo[activeStar.display]?.url && (
            <a
              href={starInfo[activeStar.display].url}
              target="_blank"
              rel="noreferrer"
              className="text-sky-300 underline text-xs mt-1 inline-block"
            >
              Learn more
            </a>
          )}
          <button
            onClick={() => setActiveStar(null)}
            className="mt-3 px-3 py-1 text-xs bg-white/10 rounded hover:bg-white/20"
          >
            Close
          </button>
        </div>
      )}
    </div>
  );
}
