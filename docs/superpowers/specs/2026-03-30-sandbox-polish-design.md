# Sandbox Polish: Visual Sizing, Orbit Preview, Trail Fade, Collision Effects

**Date:** 2026-03-30
**Goal:** Make the gravity sim more fun and visually satisfying as a sandbox/playground while maintaining physical accuracy.

---

## 1. Smart Visual Sizing

### Problem
Visual radii are exaggerated to make bodies visible at solar-system scale, causing bodies to appear to orbit inside each other.

### Design
Replace the current flat-multiplier body size system with zoom-aware min/max pixel radius clamping. All changes in JS only — no Rust changes.

**Rendering logic:**
- Compute each body's physical radius in screen pixels using current zoom level.
- Clamp to a **minimum screen radius** (default ~3-4px). Bodies are always visible.
- Clamp to a **maximum screen radius** of ~15% of the smaller screen dimension. Zooming into the Sun doesn't fill the canvas.
- Within min/max bounds, bodies scale proportionally to their physical_radius. A gas giant is visibly bigger than a rocky planet.

**Body size slider:**
- Reframed: controls the minimum-pixel floor (range: 2px to 10px).
- At minimum: tiny dots, most physically accurate proportions.
- At maximum: chunky spheres, easy to see and click.
- Default: 3-4px.

**Why this fixes the overlap:** At solar system scale, orbital gaps are enormous in screen pixels even when zoomed out. With bodies clamped to a few pixels, there's always visible separation. The "inside each other" effect was caused by exaggerated radii exceeding orbital gaps — clamping removes that.

**Black hole rendering:** Black holes keep their accretion disk glow effect but the glow radius is also computed from screen-pixel size rather than an arbitrary multiplier, ensuring consistent appearance across zoom levels.

---

## 2. Orbit Preview on Drag

### Problem
When dragging to launch a body, there's no indication of where it will go. Placement is trial and error.

### Design
While dragging, compute and draw a predicted Keplerian orbit as a dashed line. All computation in JS — no Rust changes.

**Dominant body selection:**
- For each existing body, compute gravitational influence: `F_influence = G * m / r^2` where r is distance from the drag start point.
- Use the body with the highest influence as the "primary" for the two-body approximation.
- If no bodies exist or influence is negligible, don't draw a preview.

**Orbital computation (standard Keplerian mechanics):**
1. Compute position relative to primary: `r_vec = pos_new - pos_primary`
2. Compute relative velocity: `v_vec = drag_velocity - vel_primary`
3. Specific orbital energy: `E = v^2/2 - G*M/r`
4. Specific angular momentum: `h = r_vec x v_vec`
5. Eccentricity vector: `e_vec = (v x h)/(G*M) - r_hat`
6. Semi-major axis: `a = -G*M / (2*E)` (elliptical) or infinite (parabolic)
7. For ellipses (E < 0): compute semi-minor axis `b = a * sqrt(1 - e^2)`, orientation from eccentricity vector, draw full ellipse.
8. For hyperbolas (E > 0): compute asymptotes, draw partial arc (~180-240 degrees of the near branch).
9. For near-parabolic (|E| ~ 0): treat as hyperbolic with large semi-major axis.

**Visual treatment:**
- Dashed line, same color as the body being placed, ~50% opacity.
- Ellipses: draw full closed loop.
- Hyperbolas: draw arc from approach to departure.
- Updates every frame as drag direction/length changes.
- Applies 3D camera rotation so the preview is consistent with the current view.

**Limitations (acceptable):**
- Two-body approximation — actual N-body trajectory will diverge. This is expected and fine for a placement aid.
- Very close to the primary (inside softening distance), the preview may be inaccurate. Acceptable.

---

## 3. Trail Fade

### Problem
Trails are uniform opacity, looking flat. Hard to distinguish recent from old positions.

### Design
Apply a linear opacity gradient across each body's trail. JS-only change to the existing trail drawing loop.

- **Newest point:** full trail opacity (~0.7 alpha).
- **Oldest point:** 0 alpha (fully transparent).
- **Linear interpolation** between them based on index in the trail array.
- Works with all trail length settings (50 to 2000 to infinite).
- For infinite trails, the fade spans the entire stored history — very old points become invisible, effectively self-cleaning visually even though the data persists.
- Each trail segment gets its own `strokeStyle` with computed alpha. Since we're already iterating over segments, the overhead is just changing the alpha per segment — negligible.

---

## 4. Better Collision Effects

### Problem
Collision flash is identical regardless of collision energy. Two asteroids look the same as two stars merging.

### Design
Scale collision visual effects with the kinetic energy of the collision. JS-side changes — Rust already provides mass and velocity data at merge time.

**Energy computation at merge time:**
- Relative velocity: `v_rel = |v1 - v2|`
- Collision energy proxy: `E_collision = 0.5 * (m1 * m2 / (m1 + m2)) * v_rel^2` (reduced mass formulation)
- Map energy to visual parameters on a log scale (collision energies span many orders of magnitude).

**Visual parameters scaled by energy:**

| Parameter | Small collision (asteroids) | Medium (planets) | Large (stars/BHs) |
|---|---|---|---|
| Flash radius | 1.5x body radius | 3x body radius | 6x body radius |
| Duration | 100ms | 250ms | 500ms |
| Ring effect | None | None | Expanding ring |

**Flash rendering (enhanced):**
- Bright white core (inner 30%) fading to body color at edge.
- Radial gradient as currently implemented, but sized by energy.
- Alpha fades over the duration.

**Expanding ring (large collisions only):**
- Triggered when collision energy exceeds a threshold (roughly neutron star mergers and above).
- Thin ring that expands outward from collision point at ~2x the flash expansion rate.
- Starts at flash radius, expands to 3x flash radius, fading as it goes.
- White/light blue color, ~40% starting opacity.
- Duration: ~400ms.

**Threshold for ring:** `E_collision > 1e40 J` (approximate). This means it only appears for star-mass collisions at significant relative velocities, keeping it special.

---

## Out of Scope (Future)

- **Roche limit tidal disruption** — designed but deferred. Bodies inside the Roche limit of a much more massive body break into 5-15 fragments. Emergent ring formation from fragment orbits.
- **Save/load simulation states** — discussed, user declined for now.
- **Screen shake** — discussed, user declined.

---

## Technical Notes

- All four features are JS-side changes to `www/index.html` except: none require Rust changes.
- No new dependencies.
- No changes to the WASM interface.
- Features are independent of each other and can be implemented/tested in any order.
