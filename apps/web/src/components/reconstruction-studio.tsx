"use client";

import { DragEvent, useRef, useState } from "react";
import { reconstructScene, type ReconstructionResult } from "@/lib/api";
import { Icon } from "./icons";
import { SatelliteScene } from "./satellite-scene";

type ViewMode = "compare" | "mask" | "uncertainty";
type RunState = "idle" | "running" | "complete" | "error";

const demoMetrics = {
  cloudCoverage: 38.4,
  confidence: 92.7,
  processingTimeMs: 1840,
  psnrEstimate: 31.82,
  spectralAngle: 3.16,
};

const nav = [
  { icon: "grid" as const, label: "Workspace", active: true },
  { icon: "archive" as const, label: "Scene library" },
  { icon: "layers" as const, label: "Model registry" },
  { icon: "pulse" as const, label: "Evaluations" },
];

export function ReconstructionStudio() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [runState, setRunState] = useState<RunState>("idle");
  const [view, setView] = useState<ViewMode>("compare");
  const [sensitivity, setSensitivity] = useState(58);
  const [result, setResult] = useState<ReconstructionResult | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const metrics = result?.metrics ?? demoMetrics;
  const sceneName = file?.name ?? "L4_28JUL26_052094_RGBN.tif";

  function acceptFile(next: File | undefined) {
    if (!next) return;
    setFile(next);
    setResult(null);
    setRunState("idle");
    setError("");
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  async function runReconstruction() {
    setRunState("running");
    setError("");

    if (!file) {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setRunState("complete");
      return;
    }

    try {
      const nextResult = await reconstructScene(file, sensitivity / 100);
      setResult(nextResult);
      setRunState("complete");
    } catch (reason) {
      setRunState("error");
      setError(
        reason instanceof Error
          ? `${reason.message}. Start the API or use the built-in demo scene.`
          : "The reconstruction request could not be completed.",
      );
    }
  }

  function exportResult() {
    if (result?.reconstructedPreview) {
      const link = document.createElement("a");
      link.href = result.reconstructedPreview;
      link.download = `${sceneName.replace(/\.[^.]+$/, "")}-reconstructed.png`;
      link.click();
      return;
    }

    const report = {
      scene: sceneName,
      mode: "demonstration",
      generatedAt: new Date().toISOString(),
      metrics,
      note: "Demonstration report; no trained checkpoint output was exported.",
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "liss-iv-reconstruction-report.json";
    link.click();
    URL.revokeObjectURL(url);
  }

  const resultImage = result?.reconstructedPreview;
  const originalImage = result?.originalPreview;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark" title="LISS-IV Reconstruction">
          <span>L</span>
          <small>IV</small>
        </div>

        <nav className="primary-nav" aria-label="Primary navigation">
          {nav.map((item) => (
            <button
              className={`nav-button ${item.active ? "active" : ""}`}
              key={item.label}
              title={item.label}
              type="button"
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <button className="nav-button" title="Settings" type="button">
          <Icon name="settings" />
          <span>Settings</span>
        </button>
        <div className="operator-avatar">JS</div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <div className="eyebrow">
              <span className="live-dot" />
              Reconstruction workspace
            </div>
            <h1>Cloud removal console</h1>
          </div>
          <div className="topbar-actions">
            <div className="system-chip">
              <span>Inference service</span>
              <strong>Ready</strong>
            </div>
            <button className="icon-button" type="button" aria-label="More options">
              <Icon name="more" />
            </button>
          </div>
        </header>

        <div className="content-grid">
          <section className="scene-column">
            <div className="section-heading">
              <div>
                <p className="kicker">Scene inspection</p>
                <h2>{sceneName}</h2>
              </div>
              <div className="view-switcher" role="group" aria-label="Result view">
                {(["compare", "mask", "uncertainty"] as ViewMode[]).map(
                  (mode) => (
                    <button
                      className={view === mode ? "selected" : ""}
                      key={mode}
                      onClick={() => setView(mode)}
                      type="button"
                    >
                      {mode}
                    </button>
                  ),
                )}
              </div>
            </div>

            <div className={`scene-stage ${runState === "running" ? "busy" : ""}`}>
              {view === "compare" ? (
                <div className="comparison-grid">
                  <ScenePanel
                    title="Cloud-obscured"
                    subtitle="Input · false colour composite"
                    image={originalImage}
                    variant="cloudy"
                  />
                  <ScenePanel
                    title="Reconstructed"
                    subtitle={
                      runState === "complete"
                        ? "Output · quality controlled"
                        : "Preview · run required"
                    }
                    image={resultImage}
                    variant={runState === "complete" ? "clear" : "cloudy"}
                    muted={runState !== "complete"}
                  />
                </div>
              ) : (
                <ScenePanel
                  title={
                    view === "mask" ? "Cloud probability mask" : "Uncertainty map"
                  }
                  subtitle={
                    view === "mask"
                      ? "White pixels indicate reconstruction regions"
                      : "Warmer pixels indicate lower model certainty"
                  }
                  image={
                    view === "mask"
                      ? result?.maskPreview
                      : result?.uncertaintyPreview
                  }
                  variant={view}
                  wide
                />
              )}

              {runState === "running" && (
                <div className="processing-overlay">
                  <div className="scanner" />
                  <Icon name="aperture" />
                  <strong>Reconstructing spectral signal</strong>
                  <span>Masking clouds · restoring texture · measuring confidence</span>
                </div>
              )}
            </div>

            <div className="metric-strip">
              <Metric
                label="Cloud cover"
                value={`${metrics.cloudCoverage.toFixed(1)}%`}
                tone="sand"
              />
              <Metric
                label="Confidence"
                value={`${metrics.confidence.toFixed(1)}%`}
                tone="mint"
              />
              <Metric
                label="PSNR estimate"
                value={`${metrics.psnrEstimate.toFixed(2)} dB`}
                tone="blue"
              />
              <Metric
                label="Spectral angle"
                value={`${metrics.spectralAngle.toFixed(2)}°`}
                tone="violet"
              />
              <Metric
                label="Runtime"
                value={`${(metrics.processingTimeMs / 1000).toFixed(2)} s`}
                tone="slate"
              />
            </div>

            <div className="spectral-card">
              <div className="card-heading">
                <div>
                  <p className="kicker">Spectral fidelity</p>
                  <h3>Band response comparison</h3>
                </div>
                <span className="quality-badge">
                  <Icon name="sparkle" /> Within tolerance
                </span>
              </div>
              <SpectralChart complete={runState === "complete"} />
              <div className="chart-legend">
                <span><i className="observed" />Observed</span>
                <span><i className="reconstructed" />Reconstructed</span>
                <span><i className="reference" />Neighbourhood reference</span>
              </div>
            </div>
          </section>

          <aside className="control-column">
            <div className="control-card">
              <div className="card-heading">
                <div>
                  <p className="kicker">01 · Input</p>
                  <h3>Load scene</h3>
                </div>
                <Icon name="upload" />
              </div>

              <input
                ref={inputRef}
                type="file"
                hidden
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                onChange={(event) => acceptFile(event.target.files?.[0])}
              />
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onClick={() => inputRef.current?.click()}
                onDragEnter={() => setDragging(true)}
                onDragLeave={() => setDragging(false)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={onDrop}
                role="button"
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter") inputRef.current?.click();
                }}
              >
                <Icon name={file ? "scan" : "upload"} />
                <strong>{file ? file.name : "Drop a satellite scene"}</strong>
                <span>
                  {file
                    ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis`
                    : "GeoTIFF, TIFF, PNG or JPEG · up to 64 MB"}
                </span>
                <button type="button">{file ? "Replace scene" : "Browse files"}</button>
              </div>
              {!file && (
                <div className="demo-note">
                  <span>Demo mode</span>
                  Run the preloaded representative scene without uploading data.
                </div>
              )}
            </div>

            <div className="control-card">
              <div className="card-heading">
                <div>
                  <p className="kicker">02 · Detection</p>
                  <h3>Cloud masking</h3>
                </div>
                <span className="status-tag">Auto</span>
              </div>

              <label className="range-label" htmlFor="sensitivity">
                <span>Detection sensitivity</span>
                <strong>{sensitivity}%</strong>
              </label>
              <input
                id="sensitivity"
                className="range"
                type="range"
                min="20"
                max="90"
                value={sensitivity}
                style={{ "--range": `${sensitivity}%` } as React.CSSProperties}
                onChange={(event) => setSensitivity(Number(event.target.value))}
              />
              <div className="range-scale">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>

              <div className="parameter-list">
                <Parameter label="Shadow detection" value="Enabled" />
                <Parameter label="Mask dilation" value="3 px" />
                <Parameter label="Edge feathering" value="8 px" />
              </div>
            </div>

            <div className="control-card model-card">
              <div className="card-heading">
                <div>
                  <p className="kicker">03 · Model</p>
                  <h3>Reconstruction engine</h3>
                </div>
                <span className="model-version">v0.1</span>
              </div>
              <div className="model-selection">
                <div className="model-icon"><Icon name="sparkle" /></div>
                <div>
                  <strong>Mask-guided generator</strong>
                  <span>Multispectral · uncertainty enabled</span>
                </div>
                <Icon name="chevron" />
              </div>
              <div className="model-facts">
                <span><i /> Spectral constraint</span>
                <span><i /> Spatial attention</span>
                <span><i /> Residual preservation</span>
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button
              className="run-button"
              type="button"
              onClick={runReconstruction}
              disabled={runState === "running"}
            >
              <Icon name={runState === "complete" ? "sparkle" : "play"} />
              {runState === "running"
                ? "Processing scene"
                : runState === "complete"
                  ? "Run reconstruction again"
                  : "Run reconstruction"}
              <span>⌘ ↵</span>
            </button>

            {runState === "complete" && (
              <button
                className="export-button"
                type="button"
                onClick={exportResult}
              >
                <Icon name="download" />
                Export reconstruction package
              </button>
            )}

            <div className="security-note">
              <Icon name="cloud" />
              <p>
                <strong>Local-first processing</strong>
                Uploaded scenes are processed by your configured inference service.
              </p>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

function ScenePanel({
  title,
  subtitle,
  variant,
  image,
  muted = false,
  wide = false,
}: {
  title: string;
  subtitle: string;
  variant: "cloudy" | "clear" | "mask" | "uncertainty";
  image?: string;
  muted?: boolean;
  wide?: boolean;
}) {
  return (
    <figure className={`scene-panel ${muted ? "muted" : ""} ${wide ? "wide" : ""}`}>
      <div className="scene-meta">
        <span>{title}</span>
        <small>{subtitle}</small>
      </div>
      {image ? (
        // API results are trusted data URLs produced by the local service.
        // eslint-disable-next-line @next/next/no-img-element
        <img src={image} alt={`${title} preview`} />
      ) : (
        <SatelliteScene variant={variant} />
      )}
      <div className="coordinates">
        <span>23.02° N</span>
        <span>72.57° E</span>
      </div>
      <div className="scale-bar"><i />250 m</div>
    </figure>
  );
}

function Metric({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: string;
}) {
  return (
    <div className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Parameter({ label, value }: { label: string; value: string }) {
  return (
    <div className="parameter">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SpectralChart({ complete }: { complete: boolean }) {
  return (
    <div className="spectral-chart">
      <div className="y-labels">
        <span>0.8</span><span>0.6</span><span>0.4</span><span>0.2</span><span>0.0</span>
      </div>
      <svg viewBox="0 0 700 160" preserveAspectRatio="none" aria-label="Spectral response chart">
        <g className="grid-lines">
          <path d="M0 10h700M0 45h700M0 80h700M0 115h700M0 150h700" />
        </g>
        <path
          className="area"
          d="M0 137C70 132 105 113 170 110s99 8 157-8 101-69 165-60 103 60 208 48V160H0Z"
        />
        <path
          className="line reference"
          d="M0 137C70 132 105 113 170 110s99 8 157-8 101-69 165-60 103 60 208 48"
        />
        <path
          className="line observed"
          d="M0 140C76 128 101 120 174 116s103 13 157-10 91-74 160-58 115 53 209 48"
        />
        <path
          className={`line reconstructed ${complete ? "complete" : ""}`}
          d="M0 136C74 131 106 115 173 111s101 9 157-7 99-67 163-57 105 57 207 47"
        />
      </svg>
      <div className="x-labels">
        <span>Green · 0.55 µm</span>
        <span>Red · 0.65 µm</span>
        <span>NIR · 0.82 µm</span>
      </div>
    </div>
  );
}
