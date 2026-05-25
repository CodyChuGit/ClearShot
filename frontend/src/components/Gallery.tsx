import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, ImageOff, ChevronLeft, ChevronRight } from 'lucide-react';

interface Props {
  results: string[];
}

export function Gallery({ results }: Props) {
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const handlePrevious = useCallback((e?: React.MouseEvent) => {
    e?.stopPropagation();
    setLightboxIndex((prev) => (prev !== null && prev > 0 ? prev - 1 : prev));
  }, []);

  const handleNext = useCallback((e?: React.MouseEvent) => {
    e?.stopPropagation();
    setLightboxIndex((prev) => (prev !== null && prev < results.length - 1 ? prev + 1 : prev));
  }, [results.length]);

  const handleClose = useCallback((e?: React.MouseEvent) => {
    e?.stopPropagation();
    setLightboxIndex(null);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (lightboxIndex === null) return;
      if (e.key === 'ArrowLeft') handlePrevious();
      if (e.key === 'ArrowRight') handleNext();
      if (e.key === 'Escape') handleClose();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [lightboxIndex, handlePrevious, handleNext, handleClose]);

  if (results.length === 0) {
    return (
      <div className="gallery-empty">
        <ImageOff size={40} className="gallery-empty__icon" />
        <p>No frames extracted yet</p>
      </div>
    );
  }

  const hasPrevious = lightboxIndex !== null && lightboxIndex > 0;
  const hasNext = lightboxIndex !== null && lightboxIndex < results.length - 1;

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
            onClick={() => setLightboxIndex(i)}
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
        {lightboxIndex !== null && (
          <motion.div
            className="lightbox"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          >
            <button className="lightbox__close" onClick={handleClose}>
              <X size={20} />
            </button>
            
            {hasPrevious && (
              <button className="lightbox__nav lightbox__nav--prev" onClick={handlePrevious}>
                <ChevronLeft size={32} />
              </button>
            )}

            <motion.img
              key={lightboxIndex}
              src={results[lightboxIndex]}
              alt={`Full size ${lightboxIndex + 1}`}
              className="lightbox__img"
              initial={{ scale: 0.85, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={(e) => e.stopPropagation()}
            />

            {hasNext && (
              <button className="lightbox__nav lightbox__nav--next" onClick={handleNext}>
                <ChevronRight size={32} />
              </button>
            )}
            
            <div className="lightbox__counter">
              {lightboxIndex + 1} / {results.length}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
