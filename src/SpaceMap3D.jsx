import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

/**
 * Hybrid: Pure 2D Exoplanet Map + Minimal 3D View (no textures, no drei)
 * - Toggle between 2D SVG map and 3D scene.
 * - 3D uses only three + @react-three/fiber, MeshBasicMaterial (no texture.source usage).
 */

export default function ExoplanetMapWith3D() {
  const [mode, setMode] = useState("map"); // "map" | "3d"

  return (
    <div className="w-full h-[85vh] bg-black text-white grid grid-cols-12 gap-0">
      <div className="col-span-12 lg:col-span-9 relative">
        {/* Top-left HUD */}
        <div className="absolute z-10 left-3 top-3 flex items-center gap-2 bg-white/5 backdrop-blur px-3 py-2 rounded-2xl shadow">
          <button
            onClick={() => setMode(mode === "map" ? "3d" : "map")}
            className="text-xs px-2 py-1 rounded-xl bg-white/10 hover:bg-white/20"
          >
            {mode === "map" ? "Switch to 3D" : "Switch to Map"}
          </button>
        </div>

        {mode === "map" ? <ExoBlankMap /> : <ThreeView3D />}
      </div>

      <aside className="col-span-12 lg:col-span-3 border-l border-white/10 bg-[#0b1020] p-4 space-y-4">
        <h2 className="text-xl font-semibold tracking-tight">{mode === "map" ? "Exoplanet Map (2D)" : "3D Space View"}</h2>
        {mode === "map" ? (
          <div className="text-sm text-white/80 leading-relaxed">
            Pan/zoom with mouse, click a star system. Data from NASA Exoplanet Archive (RA/Dec). Fallback to a tiny sample if API is blocked.
          </div>
        ) : (
          <div className="text-sm text-white/80 leading-relaxed">
            Drag to orbit, wheel to zoom. 100% texture-free (MeshBasicMaterial), so no WebGL <code>texture.source</code> issues.
          </div>
        )}
        <div className="text-xs text-white/50">Tested with <code>three@0.158.x</code> + <code>@react-three/fiber@8.15.x</code></div>
      </aside>
    </div>
  );
}

/* =========================
 * 2D EXOPLANET MAP (SVG)
 * ========================= */
function raDecToXY(raDeg, decDeg, width, height) {
  const x = (((raDeg % 360) + 360) % 360) / 360 * width;
  const y = (1 - (decDeg + 90) / 180) * height;
  return [x, y];
}

function ExoBlankMap() {
  const [w, h] = [2000, 1000];
  const [scale, setScale] = useState(0.45);
  const [tx, setTx] = useState(-200);
  const [ty, setTy] = useState(-120);
  const [active, setActive] = useState(null);
  const [systems, setSystems] = useState([]);
  const drag = useRef(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const url =
          "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" +
          encodeURIComponent(
            "select pl_name,hostname,ra,dec from pscomppars where ra is not null and dec is not null"
          ) +
          "&format=json";
        const res = await fetch(url);
        const rows = await res.json();
        const byHost = new Map();
        for (const r of rows) {
          if (!byHost.has(r.hostname)) {
            byHost.set(r.hostname, {
              name: r.hostname,
              ra: Number(r.ra),
              dec: Number(r.dec),
              planets: [],
            });
          }
          byHost.get(r.hostname).planets.push(r.pl_name);
        }
        if (mounted) setSystems(Array.from(byHost.values()));
      } catch (e) {
        if (mounted)
          setSystems([
            { name: "Kepler-62", ra: 267.0, dec: 45.3, planets: ["Kepler-62e", "Kepler-62f"] },
            { name: "TRAPPIST-1", ra: 346.62, dec: -5.04, planets: ["b","c","d","e","f","g","h"] },
            { name: "Kepler-452", ra: 283.6, dec: 44.5, planets: ["Kepler-452b"] },
          ]);
      }
    })();
    return () => (mounted = false);
  }, []);

  const points = useMemo(
    () => systems.map((s) => ({ ...s, xy: raDecToXY(s.ra, s.dec, w, h) })),
    [systems, w, h]
  );

  useEffect(() => {
    const onWheel = (e) => {
      e.preventDefault();
      const k = e.deltaY > 0 ? 0.9 : 1.1;
      setScale((prev) => Math.max(0.1, Math.min(5, prev * k)));
    };
    window.addEventListener("wheel", onWheel, { passive: false });
    return () => window.removeEventListener("wheel", onWheel);
  }, []);

  const onDown = (e) => {
    drag.current = { x: e.clientX, y: e.clientY, tx, ty };
  };
  const onMove = (e) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    setTx(drag.current.tx + dx);
    setTy(drag.current.ty + dy);
  };
  const onUp = () => (drag.current = null);

  return (
    <div
      style={{ width: "100%", height: "100%", background: "#03060f", color: "#fff", position: "relative", userSelect: "none" }}
      onMouseDown={onDown}
      onMouseMove={onMove}
      onMouseUp={onUp}
      onMouseLeave={onUp}
    >
      <svg width="100%" height="100%" viewBox={`0 0 ${w} ${h}`}>
        <rect x="0" y="0" width={w} height={h} fill="#03060f" />
        <g transform={`translate(${tx},${ty}) scale(${scale})`}>
          <g opacity="0.15" stroke="#6ea8ff" strokeWidth="0.5">
            {Array.from({ length: 13 }, (_, i) => {
              const y = (i / 12) * h;
              return <line key={`lat-${i}`} x1="0" y1={y} x2={w} y2={y} />;
            })}
            {Array.from({ length: 13 }, (_, i) => {
              const x = (i / 12) * w;
              return <line key={`lon-${i}`} x1={x} y1="0" x2={x} y2={h} />;
            })}
          </g>

          {points.map((s, i) => (
            <g key={i} transform={`translate(${s.xy[0]}, ${s.xy[1]})`} onClick={() => setActive(s)}>
              <circle r="3.2" fill="#ffd06e" stroke="#fff4" strokeWidth="0.6" />
            </g>
          ))}
        </g>
      </svg>

      {active && (
        <div style={{ position: "absolute", bottom: 12, left: 12, background: "rgba(10,16,32,0.85)", padding: 12, borderRadius: 10, border: "1px solid #1e2a4a" }}>
          <div style={{ fontWeight: 600 }}>{active.name}</div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>RA {active.ra}°, Dec {active.dec}°</div>
          {active.planets?.length ? (
            <div style={{ marginTop: 6, fontSize: 13 }}>Planets: {active.planets.join(", ")}</div>
          ) : null}
          <button onClick={() => setActive(null)} style={{ marginTop: 8, fontSize: 12, background: "#17223f", padding: "4px 8px", borderRadius: 6, border: "1px solid #263a74", color: "#cfe3ff" }}>Close</button>
        </div>
      )}
    </div>
  );
}

/* =========================
 * 3D VIEW (no drei; texture-free)
 * ========================= */
function ThreeView3D() {
  const [resetKey, setResetKey] = useState(0);
  return (
    <div className="w-full h-full">
      <Canvas key={resetKey} camera={{ position: [0, 0, 12], fov: 55 }} gl={{ antialias: true }} dpr={[1, 2]}>
        <color attach="background" args={["#02040a"]} />
        <BasicControls />
        <group>
          {/* Decorative ring */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[8, 0.4, 16, 64]} />
            <meshBasicMaterial color="#ff8f00" transparent opacity={0.08} />
          </mesh>
          {/* Planets (basic materials: no lights needed) */}
          <Planet r={1.2} color="#4fa6ff" position={[-3, 0, 0]} />
          <Planet r={0.9} color="#d56b43" position={[0, 0, 0]} />
          <Planet r={1.6} color="#e3b792" position={[3.5, 0, 0]} />
        </group>
      </Canvas>
      <div className="absolute right-3 top-3">
        <button onClick={() => setResetKey((k) => k + 1)} className="text-xs px-2 py-1 rounded-xl bg-white/10 hover:bg-white/20">
          Reset 3D View
        </button>
      </div>
    </div>
  );
}

function Planet({ r = 1, color = "#4fa6ff", position = [0, 0, 0] }) {
  const ref = useRef(null);
  useFrame((_, dt) => {
    if (ref.current) ref.current.rotation.y += dt * 0.3;
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[r, 32, 32]} />
      {/* MeshBasicMaterial -> NO lights, NO textures, NO env maps */}
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function BasicControls() {
  const { camera, gl } = useThree();
  const isDragging = useRef(false);
  const last = useRef({ x: 0, y: 0 });
  const spherical = useRef(new THREE.Spherical(12, Math.PI / 3, 0));

  useFrame(() => {
    const s = spherical.current;
    const target = new THREE.Vector3(0, 0, 0);
    const pos = new THREE.Vector3().setFromSpherical(s).add(target);
    camera.position.lerp(pos, 0.15);
    camera.lookAt(target);
  });

  React.useEffect(() => {
    const el = gl.domElement;
    const onDown = (e) => { isDragging.current = true; last.current = { x: e.clientX, y: e.clientY }; };
    const onMove = (e) => {
      if (!isDragging.current) return;
      const dx = e.clientX - last.current.x;
      const dy = e.clientY - last.current.y;
      last.current = { x: e.clientX, y: e.clientY };
      const s = spherical.current;
      s.theta -= dx * 0.005;
      s.phi = THREE.MathUtils.clamp(s.phi + dy * 0.005, 0.1, Math.PI - 0.1);
    };
    const onUp = () => { isDragging.current = false; };
    const onWheel = (e) => {
      const s = spherical.current;
      s.radius = THREE.MathUtils.clamp(s.radius + e.deltaY * 0.01, 4, 60);
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
  }, [gl, camera]);

  return null;
}