import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SLAP | Semantic Latent-Algebraic Protocol Dashboard",
  description: "Continuous Latent S^{d-1} Hypersphere, INT8 Residual Quantization & Latent WAF Control Plane",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#060913] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
