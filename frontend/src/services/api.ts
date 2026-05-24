import type { UploadResponse, ExtractionSettings, GpuInfo } from '../types';

const API_BASE = '/api';

export async function uploadVideo(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Upload failed');
  }

  return res.json();
}

export async function downloadUrl(url: string): Promise<UploadResponse> {
  const res = await fetch(`${API_BASE}/download-url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Download failed');
  }

  return res.json();
}

export async function startExtraction(
  jobId: string,
  settings: ExtractionSettings
): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/extract/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to start extraction');
  }

  return res.json();
}

export async function getGpuInfo(): Promise<GpuInfo> {
  const res = await fetch(`${API_BASE}/gpu-info`);
  if (!res.ok) {
    return { backend: 'cpu', device: 'Unknown', provider: 'CPUExecutionProvider' };
  }
  return res.json();
}

export const getDownloadUrl = (jobId: string) => {
  return `${API_BASE}/jobs/${jobId}/download`;
};

export const getVideoUrl = (jobId: string) => {
  return `${API_BASE}/video/${jobId}`;
};

export const getVideoDownloadUrl = (jobId: string) => {
  return `${API_BASE}/video/${jobId}/download`;
};

export function getWsUrl(jobId: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  return `${protocol}//${host}/ws/${jobId}`;
}
