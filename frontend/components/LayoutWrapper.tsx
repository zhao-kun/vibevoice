"use client";

import { usePathname } from "next/navigation";
import Navigation from "@/components/Navigation";

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isHomePage = pathname === "/";

  if (isHomePage) {
    // Home page (project selector) - no navigation
    return <>{children}</>;
  }

  // Other pages - show navigation
  return (
    <div className="flex h-screen">
      <Navigation />
      <main className="flex-1 ml-64 overflow-auto">
        {children}
      </main>
    </div>
  );
}
