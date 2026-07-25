export type ReconstructionMetrics = {
  cloudCoverage: number;
  confidence: number;
  processingTimeMs: number;
  psnrEstimate: number;
  spectralAngle: number;
};

export type ReconstructionResult = {
  requestId: string;
  originalPreview: string;
  reconstructedPreview: string;
  maskPreview: string;
  uncertaintyPreview: string;
  metrics: ReconstructionMetrics;
  model: string;
  mode: "learned" | "baseline";
};

const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

export async function reconstructScene(
  file: File,
  sensitivity: number,
): Promise<ReconstructionResult> {
  const form = new FormData();
  form.append("scene", file);
  form.append("cloud_sensitivity", sensitivity.toString());

  const response = await fetch(`${API_URL}/v1/reconstruct`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Inference failed with status ${response.status}`);
  }

  return (await response.json()) as ReconstructionResult;
}

