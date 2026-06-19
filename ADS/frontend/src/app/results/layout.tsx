"use client";

import AuthGuard from "../components/AuthGuard";

export default function ResultsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AuthGuard>{children}</AuthGuard>;
}
