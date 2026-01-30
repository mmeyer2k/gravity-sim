# 🌌 Gravity Simulator

A real-time 3D N-body gravity simulator built with Rust/WebAssembly and HTML5 Canvas.

## Quick Start

```bash
# Start the dev server
cd www
python3 -m http.server 8000

# Open http://localhost:8000
```

That's it! The WASM is pre-built. Just serve the `www` folder.

## Building from Source

Only needed if you modify the Rust code.

### Prerequisites

- [Rust](https://rustup.rs/)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)

```bash
# Install wasm-pack
cargo install wasm-pack

# Build
wasm-pack build --target web --out-dir www/pkg
```

## Controls

| Action | Effect |
|--------|--------|
| **Click** | Add body at rest |
| **Drag** | Add body with velocity (direction + speed) |
| **Right-click** | Remove nearest body |
| **Scroll** | Zoom in/out |
| **Shift + drag** | Pan view |
| **Ctrl + Shift + drag** | Rotate 3D view |
| **R** | Reset view rotation (top-down) |
| **Space** | Pause/Resume |

## Features

- **3D visualization** - Rotate the view to see orbital planes from any angle
- **N-body physics** - Full gravitational interaction between all bodies
- **Gravitational wave drag** - Neutron star mergers include energy loss from GW radiation (Peters formula)
- **Point masses** - Option for zero collision radius (realistic close encounters)
- **Collision effects** - Visual flash when bodies merge
- **Presets** - Solar system, binary stars, neutron star merger, figure-8 three-body, chaotic systems
- **Trails & vectors** - Orbital paths and velocity arrows
- **Energy tracking** - Monitor kinetic and potential energy

## Presets

| Preset | Description |
|--------|-------------|
| **Solar System** | Sun with inner planets |
| **Binary Star** | Two stars in mutual orbit with a circumbinary planet |
| **Neutron Star Merger** | Two neutron stars spiraling inward via GW radiation |
| **Figure-8** | Famous stable 3-body choreography (Chenciner-Montgomery) |
| **Chaotic System** | 3 bodies in unstable configuration |
| **Empty Space** | Blank canvas - add your own bodies |

## Physics

- **Newton's Law**: F = G × m₁ × m₂ / r²
- **Symplectic Euler integration** - Stable for orbital mechanics
- **Gravitational wave energy loss** - dE/dt ∝ (m₁m₂)² (m₁+m₂) / r⁵
- **Momentum-conserving collisions** - Bodies merge at center of mass

## Tech Stack

- **Rust** → WebAssembly (physics engine)
- **Vanilla JS** + Canvas (rendering)
- **No dependencies** - just serve static files
