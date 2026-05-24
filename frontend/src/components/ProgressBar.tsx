import { motion } from 'motion/react';

interface Props {
  progress: number;
  stage: string;
}

const STAGE_LABELS: Record<string, string> = {
  starting: 'Initializing pipeline...',
  sampling: 'Sampling frames...',
  detecting: 'Detecting faces...',
  extracting: 'Extracting crops...',
  processing: 'Processing...',
  complete: 'Complete!',
};

export function ProgressBar({ progress, stage }: Props) {
  const pct = Math.round(progress * 100);
  const label = STAGE_LABELS[stage] || stage;

  return (
    <div className="progress-container">
      <div className="progress-header">
        <span className="progress-stage">{label}</span>
        <span className="progress-pct">{pct}%</span>
      </div>
      <div className="progress-track">
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3, ease: 'easeOut' }}
        />
        <div
          className="progress-glow"
          style={{ left: `${pct}%` }}
        />
      </div>
    </div>
  );
}
