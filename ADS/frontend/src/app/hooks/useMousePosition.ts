"use client";

import { useEffect } from "react";
import { useTruxStore } from "../lib/store";

export function useMousePosition() {
  const setMouse = useTruxStore((s) => s.setMouse);

  useEffect(() => {
    const handle = (e: MouseEvent) => {
      setMouse(e.clientX, e.clientY);
    };
    window.addEventListener("mousemove", handle, { passive: true });
    return () => window.removeEventListener("mousemove", handle);
  }, [setMouse]);
}
