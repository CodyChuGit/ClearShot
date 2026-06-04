# ClearShot Frontend Design System

This document outlines the visual language, design principles, and CSS architecture for the ClearShot web application. By adhering to this guide, future contributors can maintain a cohesive, "AI UI/UX Pro Max" aesthetic that feels native, premium, and distraction-free.

## 1. Design Philosophy

ClearShot is an AI utility, not a marketing website. The design prioritizes:
- **Minimalism**: Remove all unnecessary visual noise. Focus purely on the content (video and images) and the controls.
- **Density**: Keep controls compact. Data preparation tools require high information density.
- **Fluidity**: Transitions between states (uploading -> extracting -> complete) must be seamless, utilizing micro-animations that don't slow down the user.
- **System Integration**: The app should respect the host OS's theme preferences by default, while offering manual overrides.

## 2. Color System & Theming

The application uses a semantic CSS variable system (`var(--name)`) to support robust Light and Dark modes. The default base is Dark Mode, which is ideal for working with media.

### CSS Variables
- **`--bg-primary`**: The deep base background of the app (Dark: `#0f172a`, Light: `#f9fafb`)
- **`--bg-secondary`**: The sidebar/control panel background (Dark: `#1e293b`, Light: `#ffffff`)
- **`--bg-card`**: Background for grouped content (Dark: `#1e293b`, Light: `#ffffff`)
- **`--bg-card-hover`**: Interactive hover state for cards (Dark: `#334155`, Light: `#f3f4f6`)
- **`--border-subtle`**: Very faint borders to separate sections (Dark: `#334155`, Light: `#e5e7eb`)
- **`--border-accent`**: Stronger borders for active/focused elements (Dark: `#4338ca`, Light: `#c7d2fe`)
- **`--text-primary`**: Main body text (Dark: `#f8fafc`, Light: `#111827`)
- **`--text-secondary`**: De-emphasized text (Dark: `#94a3b8`, Light: `#4b5563`)
- **`--text-muted`**: Placeholder or extremely low-priority text (Dark: `#64748b`, Light: `#9ca3af`)
- **`--accent`**: Primary brand/action color (Dark: `#6366f1`, Light: `#4f46e5`)

### Theme Toggling
The app listens to standard `@media (prefers-color-scheme: light)` for automatic system-level theme switching. 
Additionally, users can press the **`M`** key to manually toggle `.theme-light` or `.theme-dark` classes on the `<html>` root, bypassing the system preference.

## 3. Typography

ClearShot uses the system sans-serif stack to blend seamlessly with macOS and Windows. 
- **Font Family**: `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Headings**: Bold (`600`/`700` weight), tightly tracked.
- **Body**: Regular (`400`), highly legible sizing (base `14px` or `0.875rem` for UI controls).
- **Numbers**: Monospaced tabular figures where appropriate (e.g., progress percentages and stats).

## 4. Layout Architecture

The application uses a **Two-Column Split Layout**:

1. **Left Panel (Controls)**
   - Fixed width (`350px`).
   - Sticky positioned elements.
   - Contains the Video Upload component and scrollable Extraction Settings.
   - Distinct background color (`--bg-secondary`) to separate it from the results.

2. **Right Panel (Results & Gallery)**
   - Fluid width (`flex-1`).
   - Contains the Video Preview, Status Tracker, and the responsive Grid gallery.
   - Uses `--bg-primary` to let the extracted frames pop visually.

3. **Status Footer**
   - Minimalist, floating sticky footer or inline status indicator.
   - Replaces traditional heavy progress bars with sleek text-based metrics and subtle loaders.

## 5. Components & UI Elements

- **Buttons**:
  - `Primary`: Solid `--accent` background, bold text. Used for main actions ("Extract Frames").
  - `Secondary`: Transparent background, subtle border, subtle hover state. Used for resets or alternative actions.
- **Cards**: Flat design with zero drop-shadow in Dark Mode, subtle 1px borders (`--border-subtle`). Rounded corners (`--radius-lg` or `12px`).
- **Sliders & Inputs**: Native HTML inputs styled heavily to match the premium theme. Minimalist range sliders without bulky thumbs.
- **Icons**: Sourced from `lucide-react`. Stroke width `1.5` or `2.0`, sized relative to text (`16px` to `24px`). Never use solid/filled icons unless representing an active state.

## 6. Animation & Motion

Animations are handled by `motion/react`.
- **Transitions**: Keep them incredibly fast (`< 0.2s` duration).
- **Entrance**: Fade-in and slight slide-up (`y: 10`) for new elements appearing (like the gallery).
- **Hover**: Scale effects should be negligible (`1.02` max) to avoid a cheap "bouncy" feel.
- **Progress**: The extraction progress bar should use smooth CSS transitions for its `width` property.

## 7. Responsive Design

While ClearShot is primarily a desktop tool, the interface should cleanly stack or scale.
- Below `900px` width: The two-column layout should flex to a stacked single-column layout.
- The gallery uses `GridItem(.adaptive(minimum: 120))` (or CSS Grid `auto-fill, minmax(120px, 1fr)`) to automatically reflow images.
