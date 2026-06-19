"use client";

import { useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Float, PointMaterial, Points } from "@react-three/drei";
import * as THREE from "three";

function StarField() {
  const count = 2000;
  const [positions] = useState(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 50;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 50;
    }
    return pos;
  });

  const ref = useRef<THREE.Points>(null!);

  useFrame((_, delta) => {
    ref.current.rotation.y += delta * 0.015;
    ref.current.rotation.x += delta * 0.005;
  });

  return (
    <Points
      ref={ref}
      positions={positions}
      stride={3}
      frustumCulled={false}
    >
      <PointMaterial
        transparent
        color="#4D7CFE"
        size={0.05}
        sizeAttenuation
        depthWrite={false}
        opacity={0.6}
      />
    </Points>
  );
}

function FloatingOrbs() {
  const [orbs] = useState(() => {
    return Array.from({ length: 5 }, (_, i) => ({
      position: [
        (Math.random() - 0.5) * 12,
        (Math.random() - 0.5) * 8,
        (Math.random() - 0.5) * 6 - 4,
      ] as [number, number, number],
      scale: 0.3 + Math.random() * 0.6,
      color: i % 2 === 0 ? "#4D7CFE" : "#66E3FF",
      speed: 0.3 + Math.random() * 0.4,
      opacity: 0.06 + Math.random() * 0.04,
    }));
  });

  return (
    <>
      {orbs.map((orb, i) => (
        <Float key={i} speed={orb.speed} rotationIntensity={0.2} floatIntensity={0.5}>
          <mesh position={orb.position}>
            <sphereGeometry args={[orb.scale, 32, 32]} />
            <meshBasicMaterial
              color={orb.color}
              transparent
              opacity={orb.opacity}
            />
          </mesh>
        </Float>
      ))}
    </>
  );
}

function DepthFog() {
  return (
    <mesh scale={[30, 30, 30]}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshBasicMaterial color="#050816" side={THREE.BackSide} />
    </mesh>
  );
}

export default function BackgroundCanvas() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0">
      <Canvas
        camera={{ position: [0, 0, 8], fov: 60 }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: false }}
        style={{ background: "#050816" }}
      >
        <StarField />
        <FloatingOrbs />
        <DepthFog />
      </Canvas>
    </div>
  );
}
