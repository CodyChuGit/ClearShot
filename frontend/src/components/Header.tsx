import { Crosshair } from 'lucide-react';
import { GpuBadge } from './GpuBadge';

export function Header() {
  return (
    <header className="header">
      <div className="header__brand">
        <Crosshair size={22} className="header__icon" />
        <h1 className="header__title">ClearShot</h1>
      </div>
      <div className="header__right">
        <GpuBadge />
      </div>
    </header>
  );
}
