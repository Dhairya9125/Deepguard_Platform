"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

const shapes = [
  { geom: (s: number) => new THREE.TetrahedronGeometry(s), color: "#4D7CFE" },
  { geom: (s: number) => new THREE.OctahedronGeometry(s), color: "#66E3FF" },
  { geom: (s: number) => new THREE.TorusKnotGeometry(s * 0.6, s * 0.25, 32, 16), color: "#4D7CFE" },
  { geom: (s: number) => new THREE.BoxGeometry(s, s, s), color: "#66E3FF" },
  { geom: (s: number) => new THREE.IcosahedronGeometry(s), color: "#3B82F6" },
];

function Shape({
  type,
  position,
  color,
}: {
  type: number;
  position: [number, number, number];
  color: string;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const s = 0.12 + Math.random() * 0.18;

  useFrame((_, delta) => {
    meshRef.current.rotation.x += delta * (0.1 + Math.random() * 0.2);
    meshRef.current.rotation.y += delta * (0.15 + Math.random() * 0.25);
    meshRef.current.position.y += Math.sin(Date.now() * 0.001 + position[0]) * delta * 0.1;
  });

  const geometry = useMemo(() => shapes[type % shapes.length].geom(s), [type, s]);

  return (
    <mesh ref={meshRef} position={position} geometry={geometry}>
      <meshBasicMaterial color={color} transparent opacity={0.2} />
    </mesh>
  );
}

function Scene({ count }: { count: number }) {
  const items = useMemo(() => {
    return Array.from({ length: count }, (_, i) => ({
      type: i % shapes.length,
      position: [
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 8,
        (Math.random() - 0.5) * 6 - 2,
      ] as [number, number, number],
      color: shapes[i % shapes.length].color,
    }));
  }, [count]);

  return (
    <>
      {items.map((item, i) => (
        <Shape key={i} type={item.type} position={item.position} color={item.color} />
      ))}
    </>
  );
}

export default function FloatingGeometry({ count = 3 }: { count?: number }) {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 60 }}
      dpr={[1, 1.5]}
      gl={{ antialias: false, alpha: true }}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}
    >
      <Scene count={count} />
    </Canvas>
  );
}
