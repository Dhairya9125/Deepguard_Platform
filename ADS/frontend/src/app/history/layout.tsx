"use client";

import AuthGuard from "../components/AuthGuard";

export default function HistoryLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AuthGuard>{children}</AuthGuard>;
}
