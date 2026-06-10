"use client";

import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const EARTH_RADIUS = 1.8;

// Precise continent polygon outlines [lon, lat][] — detailed coastline tracing
const continents: [number, number][][] = [
  // ---- North America (full coastline) ----
  [
    [-132,54],[-130,50],[-125,48],[-124,46],[-124,44],[-124,42],[-123,41],[-122,39],[-122,37],
    [-121,36],[-120,35],[-119,34],[-118,34],[-117,33],[-117,32],[-116,30],[-115,29],[-114,27],
    [-112,26],[-110,24],[-107,22],[-104,20],[-102,19],[-100,19],[-98,18],[-96,18],[-94,18],
    [-92,18],[-90,19],[-89,20],[-88,22],[-87,24],[-85,25],[-83,24],[-82,23],[-80,22],
    [-78,22],[-77,23],[-76,25],[-77,27],[-78,29],[-80,30],[-82,30],[-81,32],[-80,33],
    [-78,34],[-77,36],[-75,38],[-74,40],[-73,42],[-71,44],[-69,46],[-67,48],[-65,48],
    [-63,48],[-61,48],[-59,50],[-57,52],[-56,54],[-57,56],[-59,58],[-62,58],[-64,60],
    [-66,62],[-70,62],[-74,62],[-78,64],[-82,66],[-86,68],[-90,68],[-94,68],[-98,66],
    [-102,64],[-106,62],[-110,60],[-114,58],[-118,56],[-122,54],[-126,52],[-130,52],
    [-134,50],[-138,48],[-142,46],[-146,44],[-150,42],[-154,40],[-158,38],[-162,36],
    [-166,34],[-170,32],[-174,30],[-178,28],[180,26],[176,24],[172,22],[168,20],[164,22],
    [162,24],[160,26],[158,28],[156,30],[154,32],[152,34],[150,36],[148,38],[146,40],
    [-144,42],[-142,44],[-140,46],[-138,48],[-136,50],[-134,52],[-132,54],
  ],
  // ---- Greenland ----
  [
    [-55,76],[-52,78],[-48,80],[-44,82],[-40,84],[-36,84],[-32,84],[-28,83],[-24,82],
    [-20,80],[-18,78],[-18,76],[-20,74],[-22,72],[-24,70],[-26,68],[-30,66],[-34,64],
    [-38,64],[-42,64],[-46,66],[-50,68],[-54,70],[-56,72],[-56,74],[-55,76],
  ],
  // ---- South America ---- 
  [
    [-80,8],[-78,10],[-76,12],[-74,12],[-72,12],[-70,12],[-68,10],[-66,10],[-64,10],
    [-62,10],[-60,8],[-58,7],[-56,6],[-54,4],[-50,2],[-46,0],[-42,-2],[-38,-4],
    [-36,-8],[-36,-12],[-38,-16],[-40,-20],[-42,-24],[-44,-28],[-46,-30],[-48,-32],
    [-50,-34],[-52,-36],[-54,-38],[-56,-40],[-58,-42],[-60,-44],[-62,-46],[-64,-48],
    [-66,-50],[-68,-52],[-70,-54],[-72,-56],[-74,-58],[-76,-58],[-75,-56],[-74,-54],
    [-72,-52],[-70,-50],[-68,-48],[-66,-44],[-64,-40],[-62,-36],[-60,-32],[-58,-28],
    [-58,-24],[-58,-20],[-60,-16],[-62,-12],[-64,-8],[-66,-4],[-68,0],[-70,2],
    [-72,4],[-74,6],[-76,8],[-78,8],[-80,8],
  ],
  // ---- Europe (detailed) ----
  [
    [-10,52],[-10,54],[-8,56],[-6,58],[-4,60],[-2,60],[0,60],[2,60],[4,60],[6,60],
    [8,60],[10,58],[12,56],[14,54],[16,54],[18,56],[20,56],[22,56],[24,56],[26,58],
    [28,60],[30,62],[32,64],[34,66],[36,68],[38,70],[40,68],[42,66],[44,64],[46,62],
    [48,60],[50,58],[48,56],[46,54],[44,52],[42,50],[40,48],[38,46],[36,44],[34,42],
    [32,40],[30,38],[28,36],[26,34],[24,32],[22,30],[20,28],[18,28],[16,28],[14,28],
    [12,30],[10,32],[8,34],[6,36],[4,38],[2,40],[0,42],[-2,44],[-4,46],[-6,48],
    [-8,50],[-10,52],
  ],
  // ---- Africa (full coastline) ----
  [
    [-16,35],[-14,36],[-12,35],[-10,34],[-6,34],[-2,34],[2,34],[6,34],[10,33],[12,32],
    [14,30],[16,28],[18,26],[20,24],[22,22],[24,20],[26,18],[28,16],[30,14],[32,12],
    [34,10],[36,8],[38,6],[40,4],[42,2],[44,0],[46,-2],[48,-4],[50,-6],[52,-8],
    [54,-10],[52,-12],[50,-14],[48,-16],[46,-18],[44,-20],[42,-22],[40,-24],[38,-26],
    [36,-28],[34,-30],[32,-32],[30,-34],[28,-34],[26,-33],[24,-30],[22,-28],[20,-26],
    [18,-24],[16,-22],[14,-20],[12,-18],[10,-16],[8,-14],[6,-12],[4,-10],[2,-8],
    [0,-6],[-2,-4],[-4,-2],[-6,0],[-8,2],[-10,4],[-12,6],[-14,8],[-16,10],
    [-16,12],[-14,14],[-12,16],[-10,18],[-8,20],[-6,22],[-4,24],[-2,26],[0,28],
    [2,30],[4,32],[2,34],[0,35],[-2,36],[-6,36],[-10,36],[-14,36],[-16,35],
  ],
  // ---- Asia (Russia + Central Asia + China) ----
  [
    [42,64],[46,66],[50,68],[54,70],[58,72],[62,74],[66,76],[70,76],[74,74],[78,72],
    [82,72],[86,72],[90,70],[94,70],[98,68],[102,66],[106,64],[110,62],[114,60],[118,58],
    [122,56],[126,54],[130,52],[132,50],[134,48],[136,46],[138,44],[140,42],[142,40],
    [144,38],[146,36],[148,34],[148,32],[146,30],[144,28],[142,26],[140,24],[138,22],
    [136,20],[134,18],[132,16],[130,14],[128,12],[126,10],[124,8],[122,6],[120,4],
    [118,2],[116,0],[114,-2],[112,-4],[110,-6],[108,-8],[106,-8],[104,-6],[102,-4],
    [100,-2],[98,0],[96,2],[94,4],[92,6],[90,8],[88,10],[86,12],[84,14],[82,16],
    [80,18],[78,20],[76,22],[74,24],[72,26],[70,28],[68,30],[66,32],[64,34],[62,36],
    [60,38],[58,40],[56,42],[54,44],[52,46],[50,48],[48,50],[46,52],[44,54],[42,56],
    [40,58],[38,60],[36,62],[34,64],[36,66],[38,68],[40,68],[42,64],
  ],
  // ---- India ----
  [
    [68,32],[70,34],[72,34],[74,30],[76,26],[78,22],[80,18],[82,14],[84,12],[86,10],
    [88,8],[90,8],[92,10],[94,12],[96,14],[96,16],[94,18],[92,20],[90,22],[88,24],
    [86,26],[84,28],[82,30],[80,32],[78,34],[76,36],[74,36],[72,34],[70,32],[68,32],
  ],
  // ---- Southeast Asia ----
  [
    [98,10],[100,8],[102,6],[104,4],[106,2],[108,0],[110,-2],[112,-4],[114,-6],
    [116,-8],[118,-8],[120,-6],[122,-4],[124,-2],[126,0],[128,2],[130,4],[132,6],
    [134,8],[134,10],[132,12],[130,14],[128,14],[126,12],[124,10],[122,8],[120,8],
    [118,10],[116,12],[114,14],[112,12],[110,10],[108,8],[106,8],[104,10],[102,12],
    [100,12],[98,10],
  ],
  // ---- Indonesia ----
  [
    [96,6],[98,4],[100,2],[102,0],[104,-2],[106,-4],[108,-6],[110,-8],[112,-8],
    [114,-6],[116,-4],[118,-2],[120,0],[122,2],[124,4],[126,6],[128,8],[130,8],
    [132,6],[132,4],[130,2],[128,0],[126,-2],[124,-4],[122,-6],[120,-8],[118,-10],
    [116,-10],[114,-8],[112,-6],[110,-4],[108,-2],[106,0],[104,2],[102,4],[100,6],
    [98,8],[96,8],[96,6],
  ],
  // ---- Japan ----
  [
    [140,44],[142,44],[144,42],[146,40],[146,38],[144,36],[142,34],[140,32],[138,30],
    [136,30],[134,32],[134,34],[136,36],[138,38],[140,40],[140,42],[140,44],
  ],
  // ---- Korea ----
  [
    [126,34],[128,34],[130,36],[130,38],[130,40],[128,42],[126,42],[124,40],[124,38],
    [124,36],[126,34],
  ],
  // ---- UK / Ireland ----
  [
    [-6,52],[-5,54],[-3,56],[-1,56],[1,54],[2,52],[2,50],[1,50],[0,50],[-2,50],
    [-4,50],[-6,52],
  ],
  // ---- Australia ----
  [
    [130,-12],[132,-12],[134,-12],[136,-12],[138,-12],[140,-12],[142,-14],[144,-16],[146,-18],
    [148,-20],[150,-22],[152,-24],[154,-26],[154,-28],[152,-30],[150,-32],[148,-34],[146,-36],
    [144,-36],[142,-34],[140,-32],[138,-30],[136,-28],[134,-26],[132,-24],[130,-22],[128,-20],
    [126,-18],[124,-16],[122,-14],[124,-12],[126,-10],[128,-10],[130,-12],
  ],
  // ---- New Zealand ----
  [
    [172,-36],[174,-36],[176,-38],[176,-40],[174,-42],[172,-44],[170,-42],[170,-40],[170,-38],[172,-36],
  ],
  // ---- Madagascar ----
  [
    [46,-14],[48,-14],[50,-16],[50,-18],[48,-20],[46,-22],[44,-24],[44,-22],[44,-20],[44,-18],
    [44,-16],[46,-14],
  ],
  // ---- Papua New Guinea ----
  [
    [140,-6],[142,-6],[144,-6],[146,-8],[148,-8],[150,-10],[152,-10],[150,-8],[148,-4],[146,-2],
    [144,0],[142,0],[140,-2],[140,-4],[140,-6],
  ],
  // ---- Philippines ----
  [
    [120,18],[122,18],[124,16],[126,14],[126,12],[124,10],[122,8],[120,8],[118,10],[118,12],
    [116,14],[118,16],[120,18],
  ],
  // ---- Cuba ----
  [
    [-86,22],[-84,22],[-82,22],[-80,22],[-78,22],[-76,22],[-74,22],[-74,20],[-76,20],
    [-78,20],[-80,20],[-82,20],[-84,20],[-86,22],
  ],
  // ---- Iceland ----
  [
    [-24,66],[-22,66],[-20,66],[-18,66],[-16,66],[-14,66],[-14,64],[-16,64],[-18,64],
    [-20,64],[-22,64],[-24,66],
  ],
];

function pointInPolygon(px: number, py: number, polygon: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function isLand(lon: number, lat: number): boolean {
  for (const poly of continents) {
    if (pointInPolygon(lon, lat, poly)) return true;
  }
  return false;
}

function lonLatToXY(lon: number, lat: number, W: number, H: number): [number, number] {
  return [((lon + 180) / 360) * W, ((90 - lat) / 180) * H];
}

function generateDotMapTexture(): THREE.CanvasTexture {
  const W = 4096, H = 2048;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;

  // Dark background
  ctx.fillStyle = "#05080F";
  ctx.fillRect(0, 0, W, H);

  const islandDots: [number, number][] = [];

  // Step 1: scatter random dots inside continents
  const targetDots = 800;
  let attempts = 0;
  while (islandDots.length < targetDots && attempts < targetDots * 30) {
    attempts++;
    const x = Math.random() * W;
    const y = Math.random() * H;
    const lon = (x / W) * 360 - 180;
    const lat = 90 - (y / H) * 180;
    if (isLand(lon, lat)) {
      islandDots.push([x, y]);
    }
  }

  // Step 2: Draw dots for continents (blue-white glowing dots)
  for (const [dx, dy] of islandDots) {
    const alpha = 0.3 + Math.random() * 0.7;
    const size = 1.5 + Math.random() * 2.5;

    // Glow around dot
    const glow = ctx.createRadialGradient(dx, dy, 0, dx, dy, size * 3);
    glow.addColorStop(0, `rgba(102, 227, 255, ${alpha * 0.3})`);
    glow.addColorStop(1, "rgba(102, 227, 255, 0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(dx, dy, size * 3, 0, Math.PI * 2);
    ctx.fill();

    // Core dot
    ctx.fillStyle = `rgba(180, 240, 255, ${alpha})`;
    ctx.beginPath();
    ctx.arc(dx, dy, size * 0.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Step 3: Draw connection lines between nearby dots (wireframe effect)
  ctx.strokeStyle = "rgba(77, 124, 254, 0.12)";
  ctx.lineWidth = 0.5;
  const connectionDist = 40;
  for (let i = 0; i < islandDots.length; i++) {
    const [x1, y1] = islandDots[i];
    let connections = 0;
    for (let j = i + 1; j < islandDots.length && connections < 2; j++) {
      const [x2, y2] = islandDots[j];
      const dist = Math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2);
      if (dist < connectionDist) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        connections++;
      }
    }
  }

  // Step 4: Add subtle coastline outline dots (brighter)
  // Sample the perimeter of each continent
  for (const poly of continents) {
    for (let i = 0; i < poly.length; i++) {
      const [lon, lat] = poly[i];
      const [x, y] = lonLatToXY(lon, lat, W, H);
      const size = 2 + Math.random() * 2;
      ctx.fillStyle = "rgba(150, 220, 255, 0.6)";
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();

      // Draw line segments between coastline points
      const next = poly[(i + 1) % poly.length];
      const [nx, ny] = lonLatToXY(next[0], next[1], W, H);
      ctx.strokeStyle = "rgba(77, 124, 254, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(nx, ny);
      ctx.stroke();
    }
  }

  // Step 5: Add a few ocean dots (very sparse, dim)
  for (let i = 0; i < 300; i++) {
    const x = Math.random() * W;
    const y = Math.random() * H;
    const lon = (x / W) * 360 - 180;
    const lat = 90 - (y / H) * 180;
    if (!isLand(lon, lat)) {
      ctx.fillStyle = `rgba(77, 124, 254, ${0.05 + Math.random() * 0.1})`;
      ctx.beginPath();
      ctx.arc(x, y, 0.8 + Math.random(), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

function generateOceanDots(count: number): Float32Array {
  const pos: number[] = [];
  let attempts = 0;
  while (pos.length < count * 3 && attempts < count * 30) {
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const lon = theta * (180 / Math.PI);
    const lat = phi * (180 / Math.PI) - 90;
    attempts++;
    if (isLand(lon, lat)) continue;
    pos.push(
      EARTH_RADIUS * 1.005 * Math.sin(phi) * Math.cos(theta),
      EARTH_RADIUS * 1.005 * Math.cos(phi),
      EARTH_RADIUS * 1.005 * Math.sin(phi) * Math.sin(theta)
    );
  }
  return new Float32Array(pos);
}

function EarthBody() {
  const tex = useMemo(() => generateDotMapTexture(), []);
  return (
    <mesh>
      <sphereGeometry args={[EARTH_RADIUS, 64, 64]} />
      <meshBasicMaterial map={tex} />
    </mesh>
  );
}

function Atmosphere() {
  return (
    <mesh>
      <sphereGeometry args={[EARTH_RADIUS * 1.015, 48, 48]} />
      <meshBasicMaterial color="#4D7CFE" transparent opacity={0.06} side={THREE.BackSide} />
    </mesh>
  );
}

function OuterGlow() {
  return (
    <mesh>
      <sphereGeometry args={[EARTH_RADIUS * 1.15, 32, 32]} />
      <meshBasicMaterial color="#4D7CFE" transparent opacity={0.03} side={THREE.BackSide} />
    </mesh>
  );
}

function OceanLightDots() {
  const geo = useMemo(() => {
    const positions = generateOceanDots(400);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, []);

  return (
    <points geometry={geo}>
      <pointsMaterial size={0.015} color="#4D7CFE" transparent opacity={0.3} sizeAttenuation />
    </points>
  );
}

function BrightOceanDots() {
  const geo = useMemo(() => {
    const positions = generateOceanDots(100);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    return g;
  }, []);

  return (
    <points geometry={geo}>
      <pointsMaterial size={0.03} color="#66E3FF" transparent opacity={0.5} sizeAttenuation />
    </points>
  );
}

function GlowRing() {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, delta) => {
    ref.current.rotation.z += delta * 0.15;
    ref.current.rotation.x += delta * 0.05;
  });
  return (
    <mesh ref={ref}>
      <ringGeometry args={[EARTH_RADIUS * 1.4, EARTH_RADIUS * 1.44, 80]} />
      <meshBasicMaterial color="#4D7CFE" transparent opacity={0.1} side={THREE.DoubleSide} />
    </mesh>
  );
}

function Scene() {
  const groupRef = useRef<THREE.Group>(null!);
  useFrame((_, delta) => {
    if (groupRef.current) groupRef.current.rotation.y += delta * 0.12;
  });

  return (
    <group ref={groupRef}>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 5, 5]} intensity={0.5} />
      <pointLight position={[-4, 3, 5]} intensity={1.5} color="#4D7CFE" />
      <pointLight position={[4, -2, -4]} intensity={1} color="#66E3FF" />
      <EarthBody />
      <Atmosphere />
      <OuterGlow />
      <OceanLightDots />
      <BrightOceanDots />
      <GlowRing />
    </group>
  );
}

export default function Hero3D() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0.5, 5], fov: 40 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <Scene />
      </Canvas>
    </div>
  );
}
