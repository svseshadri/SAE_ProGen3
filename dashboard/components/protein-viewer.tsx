"use client";

import { useEffect, useRef } from "react";

export function ProteinViewer() {
  const viewerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!viewerRef.current) return;

    let cancelled = false;
    let viewer: any = null;

    async function initViewer() {
      const $3Dmol = await import("3dmol");
      const mol = $3Dmol.default ?? $3Dmol;

      if (cancelled || !viewerRef.current) return;

      viewer = mol.createViewer(viewerRef.current, {
        antialias: true,
        backgroundColor: "#000000",
        backgroundAlpha: 0,
      });

      viewer.setBackgroundColor(0x000000, 0);

      mol.download("pdb:1CRN", viewer, {}, () => {
        viewer.setStyle({}, {
          cartoon: {
            color: "spectrum",
            ribbon: true,
            thickness: 0.8,
            style: "trace",
          },
        });

        viewer.addSurface(mol.SurfaceType.VDW, {
          opacity: 0.18,
          color: "#f9a8d4",
        });

        viewer.zoomTo();
        viewer.zoom(0.82);
        viewer.setProjection("orthographic");
        viewer.render();
      });
    }

    initViewer();

    return () => {
      cancelled = true;
      if (viewer) viewer.clear();
    };
  }, []);

  return (
    <div
      ref={viewerRef}
      className="protein-viewer h-full w-full"
      style={{ background: "transparent" }}
    />
  );
}
