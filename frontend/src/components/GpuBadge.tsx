import { useEffect, useState } from 'react';
import { Cpu, Zap } from 'lucide-react';
import { getGpuInfo } from '../services/api';
import type { GpuInfo } from '../types';

export function GpuBadge() {
  const [gpu, setGpu] = useState<GpuInfo | null>(null);

  useEffect(() => {
    getGpuInfo().then(setGpu).catch(() => {});
  }, []);

  if (!gpu) return null;

  const isGpu = gpu.backend !== 'cpu';
  const label = gpu.backend === 'cuda' ? 'CUDA' : gpu.backend === 'coreml' ? 'Metal' : 'CPU';

  return (
    <div className="gpu-badge" title={gpu.device}>
      <span className={`gpu-dot ${isGpu ? 'gpu-dot--active' : ''}`} />
      {isGpu ? <Zap size={13} /> : <Cpu size={13} />}
      <span className="gpu-label">{label}</span>
    </div>
  );
}
