import { motion } from 'motion/react';
import { Images, Minus, Eye, Copy, ImageOff, UserX } from 'lucide-react';
import type { ExtractionStats } from '../types';

interface Props {
  stats: ExtractionStats;
}

export function StatsCard({ stats }: Props) {
  const items = [
    { icon: <Images size={15} />, label: 'Extracted', value: stats.extracted, color: 'var(--success)' },
    { icon: <Minus size={15} />, label: 'Blurry', value: stats.blurry_discarded, color: 'var(--warning)' },
    { icon: <ImageOff size={15} />, label: 'Low res', value: stats.low_resolution_discarded ?? 0, color: 'var(--warning)' },
    { icon: <Eye size={15} />, label: 'No face', value: stats.no_face_discarded, color: 'var(--error)' },
    { icon: <UserX size={15} />, label: 'Occluded', value: stats.occluded_discarded ?? 0, color: 'var(--error)' },
    { icon: <Copy size={15} />, label: 'Duplicates', value: stats.duplicate_discarded, color: 'var(--accent)' },
  ];

  return (
    <motion.div
      className="stats-card"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="stats-row">
        {items.map((item, i) => (
          <motion.div
            key={item.label}
            className="stat-item"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
          >
            <span className="stat-icon" style={{ color: item.color }}>{item.icon}</span>
            <span className="stat-value" style={{ color: item.color }}>{item.value}</span>
            <span className="stat-label">{item.label}</span>
          </motion.div>
        ))}
      </div>
      <div className="stats-summary">
        {stats.total_sampled} frames sampled
        {stats.gpu_backend && stats.gpu_backend !== 'cpu' && (
          <span className="stats-gpu"> · GPU accelerated ({stats.gpu_backend})</span>
        )}
      </div>
    </motion.div>
  );
}
