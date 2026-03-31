# Sandbox Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gravity sim more visually honest and satisfying as a sandbox — fix body overlap, add orbit preview on drag, fade trails, and scale collision effects with energy.

**Architecture:** All four features are JS-only changes in `www/index.html`. No Rust/WASM changes needed. The existing `Simulation` API provides all necessary data (positions, velocities, masses, physical radii). Features are independent and can be implemented in any order.

**Tech Stack:** Vanilla JS, Canvas 2D API, existing wasm-bindgen interface.

---

### Task 1: Smart Visual Sizing

Replace the flat exaggeration multiplier with zoom-aware min/max pixel radius clamping so bodies never visually overlap during orbits.

**Files:**
- Modify: `www/index.html:566-604` (body drawing code)
- Modify: `www/index.html:189-193` (body size slider HTML + label)
- Modify: `www/index.html:284` (sizeScale state variable)
- Modify: `www/index.html:990-994` (size slider event handler)
- Modify: `www/index.html:696-712` (drag indicator uses preset.radius)
- Modify: `www/index.html:1109-1139` (hover detection uses old radius)
- Modify: `www/index.html:1253-1280` (follow click uses old radius)

- [ ] **Step 1: Replace sizeScale with minPixelRadius**

In the state variables section (~line 284), change:

```js
// Old
let sizeScale = 1.0;  // 0 = realistic, 1 = exaggerated

// New
let minPixelRadius = 4;  // Minimum screen radius in pixels (controlled by slider)
```

- [ ] **Step 2: Update body size slider HTML**

Change the slider label and range at ~line 189-193:

```html
<div class="control-group">
    <label>Min Body Size: <span id="size-value">4 px</span></label>
    <input type="range" id="size-slider" min="2" max="10" step="1" value="4">
</div>
```

- [ ] **Step 3: Update size slider event handler**

Replace the handler at ~line 990-994:

```js
document.getElementById('size-slider').addEventListener('input', (e) => {
    minPixelRadius = parseInt(e.target.value);
    document.getElementById('size-value').textContent = minPixelRadius + ' px';
});
```

- [ ] **Step 4: Create computeScreenRadius helper function**

Add this function right after `screenVelocityToWorld` (~after line 395):

```js
// Compute screen radius for a body: physical radius scaled to screen,
// clamped to min/max pixel bounds. Ensures bodies are visible but
// never so large they overlap orbital partners.
function computeScreenRadius(physicalRadius, mass) {
    const physicalScreenR = physicalRadius * zoom;
    const maxScreenR = Math.min(canvas.width, canvas.height) * 0.15;
    return Math.max(minPixelRadius, Math.min(physicalScreenR, maxScreenR));
}
```

- [ ] **Step 5: Update body drawing code to use computeScreenRadius**

Replace the radius computation in the body drawing loop at ~line 567-577. Remove all references to `exaggeratedR`, `realisticR`, and `sizeScale`. The new body drawing section:

```js
for (let i = 0; i < sim.body_count(); i++) {
    const x = sim.get_body_x(i);
    const y = sim.get_body_y(i);
    const z = sim.get_body_z(i);
    const physicalR = sim.get_body_physical_radius(i);
    const mass = sim.get_body_mass(i);
    const r = computeScreenRadius(physicalR, mass);
    const screen = worldToScreen(x, y, z);
    const color = bodyMeta[i]?.color || '#ffffff';
```

- [ ] **Step 6: Update black hole glow rendering**

In the black hole drawing section (~line 581-598), update the glow radius to use the new `r` (which is already screen-pixel-based):

```js
if (mass > 1e31) {
    // Black hole: draw black filled circle
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#000000';
    ctx.fill();

    // Draw glowing accretion ring scaled to screen radius
    const glowR = r * 2.5;
    const grad = ctx.createRadialGradient(screen.x, screen.y, r * 0.8, screen.x, screen.y, glowR);
    grad.addColorStop(0, 'rgba(255, 140, 0, 0)');
    grad.addColorStop(0.3, 'rgba(255, 140, 0, 0.7)');
    grad.addColorStop(0.6, 'rgba(255, 80, 0, 0.3)');
    grad.addColorStop(1, 'rgba(255, 50, 0, 0)');
    ctx.beginPath();
    ctx.arc(screen.x, screen.y, glowR, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
} else {
```

- [ ] **Step 7: Update drag indicator to use computeScreenRadius**

In the drag indicator drawing code (~line 696-712), replace the preset.radius usage:

```js
// Draw drag indicator
if (isDragging && dragStart) {
    const mouse = dragStart.current;
    ctx.beginPath();
    ctx.moveTo(dragStart.x, dragStart.y);
    ctx.lineTo(mouse.x, mouse.y);
    ctx.strokeStyle = '#ffff0088';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    const preset = massPresets[selectedMassIndex];
    const dragR = computeScreenRadius(0, preset.mass);
    ctx.beginPath();
    ctx.arc(dragStart.x, dragStart.y, dragR, 0, Math.PI * 2);
    ctx.fillStyle = selectedColor + '88';
    ctx.fill();
}
```

- [ ] **Step 8: Update hover detection to use computeScreenRadius**

In the hover detection code (~line 1114-1139), replace the radius lookup:

```js
for (let i = 0; i < sim.body_count(); i++) {
    const bx = sim.get_body_x(i);
    const by = sim.get_body_y(i);
    const bz = sim.get_body_z(i);
    const screen = worldToScreen(bx, by, bz);
    const physicalR = sim.get_body_physical_radius(i);
    const mass = sim.get_body_mass(i);
    const r = computeScreenRadius(physicalR, mass);
    const dx = e.clientX - screen.x;
    const dy = e.clientY - screen.y;
    const dist = Math.sqrt(dx*dx + dy*dy);

    if (dist < r + 10) { // 10px margin
```

- [ ] **Step 9: Update follow-body click detection to use computeScreenRadius**

In `tryFollowBodyAt` (~line 1254-1280), replace the radius lookup:

```js
function tryFollowBodyAt(x, y) {
    for (let i = 0; i < sim.body_count(); i++) {
        const bx = sim.get_body_x(i);
        const by = sim.get_body_y(i);
        const bz = sim.get_body_z(i);
        const screen = worldToScreen(bx, by, bz);
        const physicalR = sim.get_body_physical_radius(i);
        const mass = sim.get_body_mass(i);
        const r = computeScreenRadius(physicalR, mass);
        const dx = x - screen.x;
        const dy = y - screen.y;
        const dist = Math.sqrt(dx*dx + dy*dy);

        if (dist < r + 15) {  // 15px margin for easier selection
```

- [ ] **Step 10: Commit**

```bash
git add www/index.html
git commit -m "Replace exaggerated body sizing with zoom-aware min/max pixel clamping

Bodies now render at their physical radius scaled to screen pixels, clamped
to a configurable minimum (2-10px) so they're always visible but never
overlap orbital partners. Fixes the visual issue of bodies orbiting inside
each other."
```

---

### Task 2: Trail Fade

Apply a linear opacity gradient across trail segments so newest points are bright and oldest fade to transparent.

**Files:**
- Modify: `www/index.html:545-564` (trail drawing loop)

- [ ] **Step 1: Replace the uniform-opacity trail drawing with per-segment gradient**

Replace the trail drawing block at ~line 545-564:

```js
// Draw trails (3D) with fade from old (transparent) to new (opaque)
if (maxTrailLength > 0) {
    for (let i = 0; i < bodyMeta.length; i++) {
        const meta = bodyMeta[i];
        if (meta.trail.length < 2) continue;

        const totalPts = meta.trail.length;
        // Parse the hex color once for this trail
        const r = parseInt(meta.color.slice(1, 3), 16);
        const g = parseInt(meta.color.slice(3, 5), 16);
        const b = parseInt(meta.color.slice(5, 7), 16);

        for (let j = 1; j < totalPts; j++) {
            const alpha = (j / totalPts) * 0.7;  // 0 at oldest, 0.7 at newest
            const prev = worldToScreen(meta.trail[j-1].x, meta.trail[j-1].y, meta.trail[j-1].z || 0);
            const pt = worldToScreen(meta.trail[j].x, meta.trail[j].y, meta.trail[j].z || 0);

            ctx.beginPath();
            ctx.moveTo(prev.x, prev.y);
            ctx.lineTo(pt.x, pt.y);
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add www/index.html
git commit -m "Add trail opacity fade from old (transparent) to new (bright)

Trail segments now have per-segment alpha that fades linearly from 0 at the
oldest point to 0.7 at the newest, replacing the previous uniform opacity."
```

---

### Task 3: Orbit Preview on Drag

When dragging to set velocity, compute and draw a predicted Keplerian orbit (ellipse or hyperbola) as a dashed line.

**Files:**
- Modify: `www/index.html:696-712` (drag indicator drawing, add orbit preview after it)

- [ ] **Step 1: Add the computeOrbitPreview function**

Add this function right after `computeScreenRadius` (after the function added in Task 1):

```js
// Compute predicted Keplerian orbit points for a two-body approximation.
// Returns array of {x, y, z} world-space points, or empty if no dominant body.
function computeOrbitPreview(posX, posY, velX, velY, velZ, newMass) {
    const G = 6.674e-11;

    // Find the most gravitationally influential body
    let bestInfluence = 0;
    let primaryIdx = -1;
    for (let i = 0; i < sim.body_count(); i++) {
        const bx = sim.get_body_x(i);
        const by = sim.get_body_y(i);
        const bm = sim.get_body_mass(i);
        const dx = posX - bx;
        const dy = posY - by;
        const r2 = dx * dx + dy * dy;
        if (r2 < 1) continue;
        const influence = G * bm / r2;
        if (influence > bestInfluence) {
            bestInfluence = influence;
            primaryIdx = i;
        }
    }

    if (primaryIdx < 0) return [];

    const px = sim.get_body_x(primaryIdx);
    const py = sim.get_body_y(primaryIdx);
    const pz = sim.get_body_z(primaryIdx);
    const pvx = sim.get_body_vx(primaryIdx);
    const pvy = sim.get_body_vy(primaryIdx);
    const pvz = sim.get_body_vz(primaryIdx);
    const M = sim.get_body_mass(primaryIdx);
    const mu = G * (M + newMass);

    // Relative position and velocity
    const rx = posX - px;
    const ry = posY - py;
    const rz = 0 - pz;
    const rvx = velX - pvx;
    const rvy = velY - pvy;
    const rvz = velZ - pvz;

    const r = Math.sqrt(rx * rx + ry * ry + rz * rz);
    const v2 = rvx * rvx + rvy * rvy + rvz * rvz;

    if (r < 1) return [];

    // Specific orbital energy
    const energy = v2 / 2 - mu / r;

    // Specific angular momentum vector (h = r x v)
    const hx = ry * rvz - rz * rvy;
    const hy = rz * rvx - rx * rvz;
    const hz = rx * rvy - ry * rvx;
    const h = Math.sqrt(hx * hx + hy * hy + hz * hz);

    if (h < 1e-10) return []; // Radial trajectory, skip

    // Eccentricity vector: e = (v x h) / mu - r_hat
    // v x h
    const vxh_x = rvy * hz - rvz * hy;
    const vxh_y = rvz * hx - rvx * hz;
    const vxh_z = rvx * hy - rvy * hx;

    const ex = vxh_x / mu - rx / r;
    const ey = vxh_y / mu - ry / r;
    const ez = vxh_z / mu - rz / r;
    const e = Math.sqrt(ex * ex + ey * ey + ez * ez);

    // Semi-latus rectum
    const p = h * h / mu;

    // Orbit orientation: angle of periapsis in the orbital plane
    // For 2D-ish orbits, use atan2 of eccentricity vector
    const omega = Math.atan2(ey, ex);

    // Generate points along the orbit using the conic equation: r = p / (1 + e*cos(theta))
    const points = [];
    const numPts = 120;

    // Determine range of true anomaly to draw
    let thetaMin, thetaMax;
    if (e < 1.0) {
        // Ellipse: draw full orbit
        thetaMin = 0;
        thetaMax = 2 * Math.PI;
    } else {
        // Hyperbola: draw from -maxTheta to +maxTheta
        // Asymptote at cos(theta) = -1/e
        const maxTheta = Math.acos(-1 / Math.min(e, 100)) - 0.05;
        thetaMin = -maxTheta;
        thetaMax = maxTheta;
    }

    for (let i = 0; i <= numPts; i++) {
        const theta = thetaMin + (thetaMax - thetaMin) * (i / numPts);
        const denom = 1 + e * Math.cos(theta);
        if (denom <= 0.01) continue; // Skip near-asymptote points

        const orbitR = p / denom;

        // Cap very large orbits to avoid drawing off-screen noise
        if (orbitR > r * 200) continue;

        // Convert from orbital coordinates to world coordinates
        // The orbit lies in the plane defined by angular momentum,
        // but for simplicity we project into the XY-ish plane using omega
        const angle = theta + omega;
        const worldX = px + orbitR * Math.cos(angle);
        const worldY = py + orbitR * Math.sin(angle);
        const worldZ = pz; // Approximate: keep in primary's Z plane

        points.push({ x: worldX, y: worldY, z: worldZ });
    }

    return points;
}
```

- [ ] **Step 2: Draw the orbit preview during drag**

In the drag indicator block (~line 696), add orbit preview drawing after the existing drag line and body preview. Insert right before the closing `}` of the `if (isDragging && dragStart)` block:

```js
    // Draw orbit preview (reuses `world`, `dx`, `dy` from drag indicator above)
    const worldDrag = screenToWorld(dragStart.x, dragStart.y);
    const dragDx = dragStart.current.x - dragStart.x;
    const dragDy = dragStart.current.y - dragStart.y;
    const vScale = 150;
    const worldVel = screenVelocityToWorld(dragDx * vScale, dragDy * vScale);
    const orbitPts = computeOrbitPreview(worldDrag.x, worldDrag.y, worldVel.vx, worldVel.vy, worldVel.vz, preset.mass);

    if (orbitPts.length > 1) {
        ctx.beginPath();
        const first = worldToScreen(orbitPts[0].x, orbitPts[0].y, orbitPts[0].z);
        ctx.moveTo(first.x, first.y);
        for (let i = 1; i < orbitPts.length; i++) {
            const pt = worldToScreen(orbitPts[i].x, orbitPts[i].y, orbitPts[i].z);
            ctx.lineTo(pt.x, pt.y);
        }
        ctx.strokeStyle = selectedColor + '80';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    }
```

- [ ] **Step 3: Commit**

```bash
git add www/index.html
git commit -m "Add Keplerian orbit preview when dragging to place bodies

While dragging, a dashed line shows the predicted orbit (ellipse or
hyperbola) around the most gravitationally influential body. Uses
standard two-body orbital mechanics (conic section from energy and
angular momentum)."
```

---

### Task 4: Better Collision Effects

Scale collision flash size, duration, and visual intensity with the kinetic energy of the collision. Add an expanding ring for very high-energy events.

**Files:**
- Modify: `www/index.html:299` (collisionFlashes array definition — no change needed, just context)
- Modify: `www/index.html:874-905` (merger detection in update loop)
- Modify: `www/index.html:678-693` (collision flash drawing)
- Modify: `www/index.html:930-938` (flash animation/decay)

- [ ] **Step 1: Store pre-step velocities alongside masses for energy computation**

In the update function, expand the pre-step tracking at ~line 874-880 to also capture velocities:

```js
// Track body states before step to detect mergers
const prevCount = sim.body_count();
const prevBodies = [];
for (let i = 0; i < prevCount; i++) {
    prevBodies.push({
        mass: sim.get_body_mass(i),
        vx: sim.get_body_vx(i),
        vy: sim.get_body_vy(i),
        vz: sim.get_body_vz(i)
    });
}
```

- [ ] **Step 2: Compute collision energy and create scaled flash**

Replace the merger detection at ~line 886-905:

```js
// Check if bodies merged (count decreased) - create scaled flash
const newCount = sim.body_count();
if (newCount < prevCount) {
    const mergedCount = prevCount - newCount;
    for (let i = 0; i < newCount; i++) {
        const newMass = sim.get_body_mass(i);
        const oldMass = prevBodies[i]?.mass || 0;
        if (newMass > oldMass * 1.01) {
            // Estimate collision energy using reduced mass and relative velocity
            const absorbedMass = newMass - oldMass;
            const reducedMass = (oldMass * absorbedMass) / (oldMass + absorbedMass);

            // Estimate relative velocity from the velocity change
            const dvx = sim.get_body_vx(i) - (prevBodies[i]?.vx || 0);
            const dvy = sim.get_body_vy(i) - (prevBodies[i]?.vy || 0);
            const dvz = sim.get_body_vz(i) - (prevBodies[i]?.vz || 0);
            const dv = Math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
            // Scale up: the actual relative velocity was larger than the delta-v
            const vRel = dv * (oldMass + absorbedMass) / absorbedMass;
            const energy = 0.5 * reducedMass * vRel * vRel;

            // Map energy to visual scale (log scale, energies span many magnitudes)
            // Asteroid collision ~1e20 J, planet ~1e30 J, star ~1e44 J
            const logE = Math.max(0, Math.log10(energy + 1) - 18); // 0 for tiny, ~26 for star-star
            const energyFrac = Math.min(logE / 26, 1); // 0 to 1

            const screen = worldToScreen(sim.get_body_x(i), sim.get_body_y(i), sim.get_body_z(i));
            const bodyR = computeScreenRadius(sim.get_body_physical_radius(i), newMass);

            // Flash radius: 1.5x to 6x body radius
            const flashRadius = bodyR * (1.5 + 4.5 * energyFrac);
            // Duration controlled by decay rate: smaller = slower fade
            const decayRate = 0.06 - 0.04 * energyFrac; // 0.06 (fast) to 0.02 (slow)

            collisionFlashes.push({
                x: screen.x,
                y: screen.y,
                radius: flashRadius * 0.3,
                maxRadius: flashRadius,
                alpha: 1.0,
                decayRate: decayRate,
                color: bodyMeta[i]?.color || '#ffffff'
            });

            // Expanding ring for very high-energy collisions (E > ~1e40 J)
            if (energy > 1e40) {
                collisionFlashes.push({
                    x: screen.x,
                    y: screen.y,
                    radius: flashRadius * 0.5,
                    maxRadius: flashRadius * 3,
                    alpha: 0.4,
                    decayRate: 0.015,
                    color: '#aaddff',
                    isRing: true,
                    ringWidth: 3
                });
            }
        }
    }
}
```

- [ ] **Step 3: Update flash animation to use per-flash decay rate**

Replace the flash animation code at ~line 930-938:

```js
// Animate collision flashes (runs even when paused for smooth fadeout)
for (let i = collisionFlashes.length - 1; i >= 0; i--) {
    const flash = collisionFlashes[i];
    // Expand toward maxRadius
    flash.radius += (flash.maxRadius - flash.radius) * 0.15;
    flash.alpha -= flash.decayRate || 0.03;
    if (flash.alpha <= 0) {
        collisionFlashes.splice(i, 1);
    }
}
```

- [ ] **Step 4: Update flash drawing to support rings and white-core gradient**

Replace the collision flash drawing at ~line 678-693:

```js
// Draw collision flashes
for (const flash of collisionFlashes) {
    if (flash.isRing) {
        // Expanding ring effect
        ctx.beginPath();
        ctx.arc(flash.x, flash.y, flash.radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(170, 221, 255, ${flash.alpha})`;
        ctx.lineWidth = flash.ringWidth || 3;
        ctx.stroke();
    } else {
        // Radial flash with bright white core
        const gradient = ctx.createRadialGradient(
            flash.x, flash.y, 0,
            flash.x, flash.y, flash.radius
        );
        gradient.addColorStop(0, `rgba(255, 255, 255, ${flash.alpha})`);
        gradient.addColorStop(0.3, `rgba(255, 220, 150, ${flash.alpha * 0.8})`);
        gradient.addColorStop(0.6, `rgba(255, 100, 50, ${flash.alpha * 0.4})`);
        gradient.addColorStop(1, `rgba(255, 50, 0, 0)`);

        ctx.beginPath();
        ctx.arc(flash.x, flash.y, flash.radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add www/index.html
git commit -m "Scale collision effects with kinetic energy of merger

Flash radius, duration, and intensity now scale logarithmically with
collision energy. Small asteroid bumps get a small pop; star mergers
get a large flare with an expanding shockwave ring."
```

---

### Task 5: Final Cleanup and Verification

Remove any dead code from the old sizing system, and manually verify all four features work.

**Files:**
- Modify: `www/index.html` (remove unused `get_body_radius` calls if any remain)

- [ ] **Step 1: Search for remaining references to old exaggerated radius**

Search for `get_body_radius` (the visual/exaggerated radius getter) — it should no longer be called anywhere since we replaced all usages with `get_body_physical_radius` + `computeScreenRadius`. If any remain, replace them.

Also search for `sizeScale` — it should no longer exist. If found, remove it.

- [ ] **Step 2: Test all four features**

Start the local server and verify:

```bash
cd www && python3 -m http.server 8000
```

Manual test checklist:
- **Smart sizing:** Load Solar System preset. Bodies should be small dots with visible space between them. Zoom in/out — bodies clamp to min pixels. Adjust body size slider — min size changes from 2px to 10px. Black hole preset shows glow correctly.
- **Trail fade:** Load any preset, let it run. Trails should fade from bright (newest) to transparent (oldest). Check with different trail lengths (50, 500, infinity).
- **Orbit preview:** In Solar System, drag to create a new body near a planet. A dashed orbit line should appear showing the predicted path. Try elliptical (slow drag) and hyperbolic (fast drag). Preview should update as you move the mouse.
- **Collision effects:** Set up bodies to collide. Small bodies should produce small flashes. Load Neutron Star Merger — the final collision should produce a large flash with an expanding ring.

- [ ] **Step 3: Final commit if cleanup was needed**

```bash
git add www/index.html
git commit -m "Clean up dead code from old body sizing system"
```
