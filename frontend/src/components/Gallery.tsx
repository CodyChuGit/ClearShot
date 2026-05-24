import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, ImageOff } from 'lucide-react';

interface Props {
  results: string[];
}

export function Gallery({ results }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (results.length === 0) {
    return (
      <div className="gallery-empty">
        <ImageOff size={40} className="gallery-empty__icon" />
        <p>No frames extracted yet</p>
      </div>
    );
  }

  return (
    <>
      <div className="gallery-grid">
        {results.map((src, i) => (
          <motion.div
            key={src}
            className="gallery-item"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.03, duration: 0.25 }}
            onClick={() => setLightbox(src)}
          >
            <img
              src={src}
              alt={`Extracted frame ${i + 1}`}
              loading="lazy"
              className="gallery-img"
            />
            <div className="gallery-item__overlay">
              <span className="gallery-item__index">#{i + 1}</span>
            </div>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {lightbox && (
          <motion.div
            className="lightbox"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLightbox(null)}
          >
            <button className="lightbox__close" onClick={() => setLightbox(null)}>
              <X size={20} />
            </button>
            <motion.img
              src={lightbox}
              alt="Full size"
              className="lightbox__img"
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ duration: 0.2 }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
