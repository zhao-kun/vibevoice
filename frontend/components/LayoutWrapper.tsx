"use client";

import { usePathname } from "next/navigation";
import Navigation from "@/components/Navigation";
import { useEffect, useState } from "react";

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    console.log('[LayoutWrapper] Component mounted on client, pathname:', pathname);
    setMounted(true);
  }, [pathname]);

  const isHomePage = pathname === "/";
  const showNavigation = mounted && !isHomePage;

  console.log('[LayoutWrapper] Render:', { pathname, mounted, isHomePage, showNavigation });
  console.log('[LayoutWrapper] BUILD_TIMESTAMP: 2025-10-31-01:30:00');
  console.log('[LayoutWrapper] children:', typeof children, children);
  console.log('[LayoutWrapper] About to render children in <main>');

  // Always return consistent wrapper structure
  return (
    <div className={showNavigation ? "flex h-screen" : ""}>
      {showNavigation && <Navigation />}
      <main className={showNavigation ? "flex-1 ml-64 overflow-auto" : ""}>
        {children}
      </main>
    </div>
  );
}
