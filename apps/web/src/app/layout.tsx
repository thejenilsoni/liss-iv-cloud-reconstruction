import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LISS-IV Reconstruction Console",
  description:
    "Cloud detection, multispectral reconstruction, and uncertainty analysis for LISS-IV imagery.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

