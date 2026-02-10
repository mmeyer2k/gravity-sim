use wasm_bindgen::prelude::*;

const G: f64 = 6.674e-11; // Gravitational constant
const C: f64 = 2.998e8;   // Speed of light
const SOFTENING: f64 = 1e6; // Softening parameter (1000 km) - prevents singularities at close range
const MIN_DIST: f64 = 1.0;  // Minimum distance to prevent division by zero (1 meter)

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Body {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    pub mass: f64,
    pub radius: f64,           // Visual radius (pixels)
    pub physical_radius: f64,  // Actual radius (meters)
}

#[wasm_bindgen]
impl Body {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, vx: f64, vy: f64, mass: f64, radius: f64) -> Body {
        let physical_radius = estimate_physical_radius(mass);
        Body { x, y, z: 0.0, vx, vy, vz: 0.0, mass, radius, physical_radius }
    }
    
    pub fn new_3d(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64, radius: f64) -> Body {
        let physical_radius = estimate_physical_radius(mass);
        Body { x, y, z, vx, vy, vz, mass, radius, physical_radius }
    }
    
    /// Create a point mass (zero collision radius) - objects pass through each other
    pub fn new_point_mass(x: f64, y: f64, z: f64, vx: f64, vy: f64, vz: f64, mass: f64, radius: f64) -> Body {
        Body { x, y, z, vx, vy, vz, mass, radius, physical_radius: 0.0 }
    }
}

// Estimate physical radius from mass
fn estimate_physical_radius(mass: f64) -> f64 {
    // Use different density regimes:
    // - Rocky bodies (asteroids, planets): ~5000 kg/m³
    // - Gas giants: ~1300 kg/m³  
    // - Stars: ~1400 kg/m³
    // - Neutron stars: ~4e17 kg/m³ (mass 1.4-3 solar masses, radius ~10-15km)
    
    let density = if mass < 1e20 {
        3000.0  // Asteroid
    } else if mass < 1e25 {
        5000.0  // Rocky planet
    } else if mass < 1e28 {
        1300.0  // Gas giant
    } else if mass < 2.5e30 {
        1400.0  // Star (up to ~1.25 solar masses)
    } else if mass < 6e30 {
        4.0e17  // Neutron star (~1.25-3 solar masses, ultra-dense)
    } else {
        1400.0  // Massive star / supergiant
    };
    
    // V = 4/3 * pi * r³, m = density * V
    // r = (3m / (4 * pi * density))^(1/3)
    let volume = mass / density;
    (3.0 * volume / (4.0 * std::f64::consts::PI)).powf(1.0 / 3.0)
}

#[wasm_bindgen]
pub struct Simulation {
    bodies: Vec<Body>,
    gravity_mult: f64,
    collisions_enabled: bool,
    gw_drag_enabled: bool,  // Gravitational wave radiation energy loss
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        Simulation {
            bodies: Vec::new(),
            gravity_mult: 1.0,
            collisions_enabled: true,
            gw_drag_enabled: false,
        }
    }

    pub fn set_collisions(&mut self, enabled: bool) {
        self.collisions_enabled = enabled;
    }

    pub fn get_collisions(&self) -> bool {
        self.collisions_enabled
    }

    pub fn set_gw_drag(&mut self, enabled: bool) {
        self.gw_drag_enabled = enabled;
    }

    pub fn get_gw_drag(&self) -> bool {
        self.gw_drag_enabled
    }

    pub fn set_gravity_mult(&mut self, mult: f64) {
        self.gravity_mult = mult;
    }

    pub fn get_gravity_mult(&self) -> f64 {
        self.gravity_mult
    }

    pub fn add_body(&mut self, body: Body) {
        self.bodies.push(body);
    }

    pub fn remove_body(&mut self, index: usize) {
        if index < self.bodies.len() {
            self.bodies.remove(index);
        }
    }

    pub fn clear(&mut self) {
        self.bodies.clear();
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    pub fn get_body_x(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.x).unwrap_or(0.0)
    }

    pub fn get_body_y(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.y).unwrap_or(0.0)
    }

    pub fn get_body_z(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.z).unwrap_or(0.0)
    }

    pub fn get_body_vx(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.vx).unwrap_or(0.0)
    }

    pub fn get_body_vy(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.vy).unwrap_or(0.0)
    }

    pub fn get_body_vz(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.vz).unwrap_or(0.0)
    }

    pub fn get_body_mass(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.mass).unwrap_or(0.0)
    }

    pub fn get_body_radius(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.radius).unwrap_or(0.0)
    }

    pub fn get_body_physical_radius(&self, index: usize) -> f64 {
        self.bodies.get(index).map(|b| b.physical_radius).unwrap_or(0.0)
    }

    /// Step simulation using Velocity Verlet integration (2nd order symplectic)
    /// This is the "kick-drift-kick" form of leapfrog, which is time-reversible
    /// and conserves energy much better than simple Euler methods.
    /// dt is in seconds
    pub fn step(&mut self, dt: f64) {
        let n = self.bodies.len();
        if n == 0 {
            return;
        }

        // Step 1: Calculate initial accelerations
        let (mut ax, mut ay, mut az) = self.compute_accelerations();

        // Step 2: Half-kick - update velocities by half timestep
        let half_dt = dt * 0.5;
        for i in 0..n {
            self.bodies[i].vx += ax[i] * half_dt;
            self.bodies[i].vy += ay[i] * half_dt;
            self.bodies[i].vz += az[i] * half_dt;
        }

        // Step 3: Drift - update positions by full timestep
        for i in 0..n {
            self.bodies[i].x += self.bodies[i].vx * dt;
            self.bodies[i].y += self.bodies[i].vy * dt;
            self.bodies[i].z += self.bodies[i].vz * dt;
        }

        // Step 4: Recompute accelerations at new positions
        let (ax_new, ay_new, az_new) = self.compute_accelerations();
        ax = ax_new;
        ay = ay_new;
        az = az_new;

        // Step 5: Half-kick - complete velocity update
        for i in 0..n {
            self.bodies[i].vx += ax[i] * half_dt;
            self.bodies[i].vy += ay[i] * half_dt;
            self.bodies[i].vz += az[i] * half_dt;
        }

        // Gravitational wave radiation drag (causes orbital decay)
        if self.gw_drag_enabled && n >= 2 {
            self.apply_gw_drag(dt);
        }

        // Handle collisions/merging
        if self.collisions_enabled {
            self.handle_collisions();
        }
    }
    
    /// Compute gravitational accelerations for all bodies
    /// Uses Newton's law of universal gravitation: F = G * m1 * m2 / r^2
    fn compute_accelerations(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.bodies.len();
        let mut ax = vec![0.0; n];
        let mut ay = vec![0.0; n];
        let mut az = vec![0.0; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                
                let dist_sq = dx * dx + dy * dy + dz * dz;
                
                // Apply softening to prevent singularity at close range
                let dist_sq_soft = dist_sq + SOFTENING * SOFTENING;
                
                // Prevent division by zero for unit vector
                let dist = dist_sq.sqrt().max(MIN_DIST);
                
                // a = G * m / r^2, using softened distance for magnitude
                let accel_mag = G * self.gravity_mult / dist_sq_soft;
                
                // Unit vector from i to j (using actual distance for direction)
                let inv_dist = 1.0 / dist;
                let ux = dx * inv_dist;
                let uy = dy * inv_dist;
                let uz = dz * inv_dist;
                
                // Apply Newton's 3rd law: equal and opposite accelerations
                // a_i = G * m_j / r^2 (toward j)
                // a_j = G * m_i / r^2 (toward i)
                let ai = accel_mag * self.bodies[j].mass;
                let aj = accel_mag * self.bodies[i].mass;
                
                ax[i] += ai * ux;
                ay[i] += ai * uy;
                az[i] += ai * uz;
                ax[j] -= aj * ux;
                ay[j] -= aj * uy;
                az[j] -= aj * uz;
            }
        }
        
        (ax, ay, az)
    }
    
    /// Apply gravitational wave radiation energy loss (Peters formula approximation)
    /// This causes orbits to decay over time, simulating inspiral
    /// Reference: Peters, P.C. (1964) "Gravitational Radiation and the Motion of Two Point Masses"
    /// 
    /// NOTE: The effect is boosted by 10^15 for visualization purposes.
    /// Real GW inspiral at these separations takes millions of years.
    fn apply_gw_drag(&mut self, dt: f64) {
        let n = self.bodies.len();
        // Pre-factor: 32/5 * G^4 / c^5, boosted for visibility
        let gw_factor = (32.0 / 5.0) * G.powi(4) / C.powi(5) * 1.0e15;
        
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dist = dist_sq.sqrt().max(MIN_DIST);
                
                let m1 = self.bodies[i].mass;
                let m2 = self.bodies[j].mass;
                let total_mass = m1 + m2;
                
                // Power radiated: P = (32/5) * (G^4/c^5) * (m1*m2)^2 * (m1+m2) / r^5
                let power = gw_factor * (m1 * m2).powi(2) * total_mass / dist.powi(5);
                
                // Relative velocity
                let dvx = self.bodies[j].vx - self.bodies[i].vx;
                let dvy = self.bodies[j].vy - self.bodies[i].vy;
                let dvz = self.bodies[j].vz - self.bodies[i].vz;
                let rel_speed_sq = dvx * dvx + dvy * dvy + dvz * dvz;
                
                if rel_speed_sq < 1.0 { continue; }
                
                // Total kinetic energy in center-of-mass frame
                let reduced_mass = (m1 * m2) / total_mass;
                let kinetic_energy = 0.5 * reduced_mass * rel_speed_sq;
                
                if kinetic_energy < 1.0 { continue; }
                
                // Energy loss this timestep
                let energy_loss = power * dt;
                
                // Fraction of energy to remove (capped to prevent instability)
                let fraction = (energy_loss / kinetic_energy).min(0.01);
                
                // To cause inspiral, we need to remove orbital energy by applying
                // tangential drag (opposite to the direction of motion).
                // 
                // The tangential velocity is the component perpendicular to the radial direction.
                // Reducing tangential speed removes angular momentum and causes inspiral.
                
                let drag = fraction * 0.5;  // Split between both bodies
                
                // Unit vector from i to j (radial direction)
                let inv_dist = 1.0 / dist;
                let rx = dx * inv_dist;
                let ry = dy * inv_dist;
                let rz = dz * inv_dist;
                
                // Radial component of relative velocity
                let v_radial = dvx * rx + dvy * ry + dvz * rz;
                
                // Tangential component of relative velocity (what we want to reduce)
                let dvx_tan = dvx - v_radial * rx;
                let dvy_tan = dvy - v_radial * ry;
                let dvz_tan = dvz - v_radial * rz;
                
                // Mass ratios for momentum conservation
                let m2_ratio = m2 / total_mass;
                let m1_ratio = m1 / total_mass;
                
                // Apply drag to tangential velocity only
                // This reduces angular momentum, causing inspiral
                // Body i: moves toward j's tangential velocity (reduces relative tangential v)
                // Body j: moves toward i's tangential velocity
                self.bodies[i].vx += drag * dvx_tan * m2_ratio;
                self.bodies[i].vy += drag * dvy_tan * m2_ratio;
                self.bodies[i].vz += drag * dvz_tan * m2_ratio;
                self.bodies[j].vx -= drag * dvx_tan * m1_ratio;
                self.bodies[j].vy -= drag * dvy_tan * m1_ratio;
                self.bodies[j].vz -= drag * dvz_tan * m1_ratio;
            }
        }
    }

    fn handle_collisions(&mut self) {
        // Check all pairs for collisions, merge if overlapping (3D)
        let mut i = 0;
        while i < self.bodies.len() {
            let mut j = i + 1;
            while j < self.bodies.len() {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                
                // Collision when bodies physically touch
                let collision_dist = self.bodies[i].physical_radius + self.bodies[j].physical_radius;
                
                if dist < collision_dist {
                    // Merge bodies: conserve momentum (3D)
                    let m1 = self.bodies[i].mass;
                    let m2 = self.bodies[j].mass;
                    let total_mass = m1 + m2;
                    
                    // New velocity from momentum conservation
                    let new_vx = (m1 * self.bodies[i].vx + m2 * self.bodies[j].vx) / total_mass;
                    let new_vy = (m1 * self.bodies[i].vy + m2 * self.bodies[j].vy) / total_mass;
                    let new_vz = (m1 * self.bodies[i].vz + m2 * self.bodies[j].vz) / total_mass;
                    
                    // New position at center of mass
                    let new_x = (m1 * self.bodies[i].x + m2 * self.bodies[j].x) / total_mass;
                    let new_y = (m1 * self.bodies[i].y + m2 * self.bodies[j].y) / total_mass;
                    let new_z = (m1 * self.bodies[i].z + m2 * self.bodies[j].z) / total_mass;
                    
                    // New visual radius based on combined volume
                    let r1 = self.bodies[i].radius;
                    let r2 = self.bodies[j].radius;
                    let new_radius = (r1.powi(3) + r2.powi(3)).powf(1.0 / 3.0);
                    
                    // New physical radius from combined mass
                    let new_physical_radius = estimate_physical_radius(total_mass);
                    
                    // Update body i with merged properties
                    self.bodies[i].x = new_x;
                    self.bodies[i].y = new_y;
                    self.bodies[i].z = new_z;
                    self.bodies[i].vx = new_vx;
                    self.bodies[i].vy = new_vy;
                    self.bodies[i].vz = new_vz;
                    self.bodies[i].mass = total_mass;
                    self.bodies[i].radius = new_radius;
                    self.bodies[i].physical_radius = new_physical_radius;
                    
                    // Remove body j
                    self.bodies.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    /// Get total kinetic energy (3D)
    pub fn kinetic_energy(&self) -> f64 {
        self.bodies.iter()
            .map(|b| 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz))
            .sum()
    }

    /// Get total potential energy (3D)
    pub fn potential_energy(&self) -> f64 {
        let mut pe = 0.0;
        let n = self.bodies.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                pe -= G * self.bodies[i].mass * self.bodies[j].mass / dist;
            }
        }
        pe
    }

    pub fn center_of_mass_x(&self) -> f64 {
        let total_mass: f64 = self.bodies.iter().map(|b| b.mass).sum();
        if total_mass == 0.0 { return 0.0; }
        self.bodies.iter().map(|b| b.x * b.mass).sum::<f64>() / total_mass
    }

    pub fn center_of_mass_y(&self) -> f64 {
        let total_mass: f64 = self.bodies.iter().map(|b| b.mass).sum();
        if total_mass == 0.0 { return 0.0; }
        self.bodies.iter().map(|b| b.y * b.mass).sum::<f64>() / total_mass
    }

    pub fn center_of_mass_z(&self) -> f64 {
        let total_mass: f64 = self.bodies.iter().map(|b| b.mass).sum();
        if total_mass == 0.0 { return 0.0; }
        self.bodies.iter().map(|b| b.z * b.mass).sum::<f64>() / total_mass
    }
}

// Preset: Solar system (planets only, no moons)
#[wasm_bindgen]
pub fn create_solar_system_no_moons() -> Simulation {
    let mut sim = Simulation::new();
    
    let incl_mercury = 7.00_f64.to_radians();
    let incl_venus = 3.39_f64.to_radians();
    let incl_mars = 1.85_f64.to_radians();
    let incl_jupiter = 1.31_f64.to_radians();
    let incl_saturn = 2.49_f64.to_radians();
    let incl_uranus = 0.77_f64.to_radians();
    let incl_neptune = 1.77_f64.to_radians();
    
    let sun_mass = 1.989e30;
    sim.add_body(Body::new(0.0, 0.0, 0.0, 0.0, sun_mass, 30.0));
    
    // Mercury
    let mercury_dist = 5.79e10;
    let mercury_v = (G * sun_mass / mercury_dist).sqrt();
    let vy = mercury_v * incl_mercury.cos();
    let vz = mercury_v * incl_mercury.sin();
    sim.add_body(Body::new_3d(mercury_dist, 0.0, 0.0, 0.0, vy, vz, 3.301e23, 4.0));
    
    // Venus
    let venus_dist = 1.082e11;
    let venus_v = (G * sun_mass / venus_dist).sqrt();
    let y = venus_dist * incl_venus.cos();
    let z = venus_dist * incl_venus.sin();
    sim.add_body(Body::new_3d(0.0, y, z, -venus_v, 0.0, 0.0, 4.867e24, 7.0));
    
    // Earth
    let earth_dist = 1.496e11;
    let earth_v = (G * sun_mass / earth_dist).sqrt();
    sim.add_body(Body::new_3d(-earth_dist, 0.0, 0.0, 0.0, -earth_v, 0.0, 5.972e24, 8.0));
    
    // Mars
    let mars_dist = 2.279e11;
    let mars_v = (G * sun_mass / mars_dist).sqrt();
    let mars_y = -mars_dist * incl_mars.cos();
    let mars_z = -mars_dist * incl_mars.sin();
    sim.add_body(Body::new_3d(0.0, mars_y, mars_z, mars_v, 0.0, 0.0, 6.417e23, 5.0));
    
    // Jupiter
    let jupiter_dist = 7.785e11;
    let jupiter_v = (G * sun_mass / jupiter_dist).sqrt();
    let jupiter_vy = jupiter_v * incl_jupiter.cos();
    let jupiter_vz = jupiter_v * incl_jupiter.sin();
    sim.add_body(Body::new_3d(jupiter_dist, 0.0, 0.0, 0.0, jupiter_vy, jupiter_vz, 1.898e27, 18.0));
    
    // Saturn
    let saturn_dist = 1.432e12;
    let saturn_v = (G * sun_mass / saturn_dist).sqrt();
    let saturn_y = saturn_dist * incl_saturn.cos();
    let saturn_z = saturn_dist * incl_saturn.sin();
    sim.add_body(Body::new_3d(0.0, saturn_y, saturn_z, -saturn_v, 0.0, 0.0, 5.683e26, 15.0));
    
    // Uranus
    let uranus_dist = 2.867e12;
    let uranus_v = (G * sun_mass / uranus_dist).sqrt();
    let uranus_vy = -uranus_v * incl_uranus.cos();
    let uranus_vz = -uranus_v * incl_uranus.sin();
    sim.add_body(Body::new_3d(-uranus_dist, 0.0, 0.0, 0.0, uranus_vy, uranus_vz, 8.681e25, 10.0));
    
    // Neptune
    let neptune_dist = 4.515e12;
    let neptune_v = (G * sun_mass / neptune_dist).sqrt();
    let neptune_y = -neptune_dist * incl_neptune.cos();
    let neptune_z = -neptune_dist * incl_neptune.sin();
    sim.add_body(Body::new_3d(0.0, neptune_y, neptune_z, neptune_v, 0.0, 0.0, 1.024e26, 10.0));

    sim
}

// Preset: Solar system with all major moons
#[wasm_bindgen]
pub fn create_solar_system() -> Simulation {
    let mut sim = Simulation::new();
    
    // Orbital inclinations relative to ecliptic (in radians)
    let incl_mercury = 7.00_f64.to_radians();
    let incl_venus = 3.39_f64.to_radians();
    let incl_moon = 5.14_f64.to_radians();  // Moon's orbit inclined to ecliptic
    let incl_mars = 1.85_f64.to_radians();
    let incl_jupiter = 1.31_f64.to_radians();
    let incl_saturn = 2.49_f64.to_radians();
    let incl_uranus = 0.77_f64.to_radians();
    let incl_neptune = 1.77_f64.to_radians();
    
    let sun_mass = 1.989e30;
    sim.add_body(Body::new(0.0, 0.0, 0.0, 0.0, sun_mass, 30.0));
    
    // Mercury: 0.387 AU, mass 3.301e23 kg, incl 7.00°
    let mercury_dist = 5.79e10;
    let mercury_v = (G * sun_mass / mercury_dist).sqrt();
    let vy = mercury_v * incl_mercury.cos();
    let vz = mercury_v * incl_mercury.sin();
    sim.add_body(Body::new_3d(mercury_dist, 0.0, 0.0, 0.0, vy, vz, 3.301e23, 4.0));
    
    // Venus: 0.723 AU, mass 4.867e24 kg, incl 3.39°
    let venus_dist = 1.082e11;
    let venus_v = (G * sun_mass / venus_dist).sqrt();
    let y = venus_dist * incl_venus.cos();
    let z = venus_dist * incl_venus.sin();
    sim.add_body(Body::new_3d(0.0, y, z, -venus_v, 0.0, 0.0, 4.867e24, 7.0));
    
    // ========== EARTH SYSTEM ==========
    let earth_dist = 1.496e11;
    let earth_v = (G * sun_mass / earth_dist).sqrt();
    let earth_mass = 5.972e24;
    let earth_x = -earth_dist;
    let earth_vy = -earth_v;
    sim.add_body(Body::new_3d(earth_x, 0.0, 0.0, 0.0, earth_vy, 0.0, earth_mass, 8.0));
    
    // Luna (Moon): 384,400 km, mass 7.342e22 kg
    let moon_dist = 3.844e8;
    let moon_v = (G * earth_mass / moon_dist).sqrt();
    let moon_y = moon_dist * incl_moon.cos();
    let moon_z = moon_dist * incl_moon.sin();
    sim.add_body(Body::new_3d(earth_x, moon_y, moon_z, -moon_v * incl_moon.cos(), earth_vy, -moon_v * incl_moon.sin(), 7.342e22, 3.0));
    
    // ========== MARS SYSTEM ==========
    let mars_dist = 2.279e11;
    let mars_v = (G * sun_mass / mars_dist).sqrt();
    let mars_mass = 6.417e23;
    let mars_y = -mars_dist * incl_mars.cos();
    let mars_z = -mars_dist * incl_mars.sin();
    let mars_vx = mars_v;
    sim.add_body(Body::new_3d(0.0, mars_y, mars_z, mars_vx, 0.0, 0.0, mars_mass, 5.0));
    
    // Phobos: 9,376 km, mass 1.0659e16 kg
    let phobos_dist = 9.376e6;
    let phobos_v = (G * mars_mass / phobos_dist).sqrt();
    sim.add_body(Body::new_3d(phobos_dist, mars_y, mars_z, mars_vx, phobos_v, 0.0, 1.0659e16, 1.0));
    
    // Deimos: 23,463 km, mass 1.4762e15 kg
    let deimos_dist = 2.3463e7;
    let deimos_v = (G * mars_mass / deimos_dist).sqrt();
    sim.add_body(Body::new_3d(-deimos_dist, mars_y, mars_z, mars_vx, -deimos_v, 0.0, 1.4762e15, 1.0));
    
    // ========== JUPITER SYSTEM ==========
    let jupiter_dist = 7.785e11;
    let jupiter_v = (G * sun_mass / jupiter_dist).sqrt();
    let jupiter_mass = 1.898e27;
    let jupiter_x = jupiter_dist;
    let jupiter_vy = jupiter_v * incl_jupiter.cos();
    let jupiter_vz = jupiter_v * incl_jupiter.sin();
    sim.add_body(Body::new_3d(jupiter_x, 0.0, 0.0, 0.0, jupiter_vy, jupiter_vz, jupiter_mass, 18.0));
    
    // Io: 421,700 km, mass 8.9319e22 kg
    let io_dist = 4.217e8;
    let io_v = (G * jupiter_mass / io_dist).sqrt();
    sim.add_body(Body::new_3d(jupiter_x, io_dist, 0.0, -io_v, jupiter_vy, jupiter_vz, 8.9319e22, 2.5));
    
    // Europa: 671,034 km, mass 4.7998e22 kg
    let europa_dist = 6.71034e8;
    let europa_v = (G * jupiter_mass / europa_dist).sqrt();
    sim.add_body(Body::new_3d(jupiter_x, -europa_dist, 0.0, europa_v, jupiter_vy, jupiter_vz, 4.7998e22, 2.2));
    
    // Ganymede: 1,070,412 km, mass 1.4819e23 kg (largest moon in solar system)
    let ganymede_dist = 1.070412e9;
    let ganymede_v = (G * jupiter_mass / ganymede_dist).sqrt();
    sim.add_body(Body::new_3d(jupiter_x + ganymede_dist, 0.0, 0.0, 0.0, jupiter_vy + ganymede_v, jupiter_vz, 1.4819e23, 3.0));
    
    // Callisto: 1,882,709 km, mass 1.0759e23 kg
    let callisto_dist = 1.882709e9;
    let callisto_v = (G * jupiter_mass / callisto_dist).sqrt();
    sim.add_body(Body::new_3d(jupiter_x - callisto_dist, 0.0, 0.0, 0.0, jupiter_vy - callisto_v, jupiter_vz, 1.0759e23, 2.8));
    
    // ========== SATURN SYSTEM ==========
    let saturn_dist = 1.432e12;
    let saturn_v = (G * sun_mass / saturn_dist).sqrt();
    let saturn_mass = 5.683e26;
    let saturn_y = saturn_dist * incl_saturn.cos();
    let saturn_z = saturn_dist * incl_saturn.sin();
    let saturn_vx = -saturn_v;
    sim.add_body(Body::new_3d(0.0, saturn_y, saturn_z, saturn_vx, 0.0, 0.0, saturn_mass, 15.0));
    
    // Mimas: 185,539 km, mass 3.75e19 kg
    let mimas_dist = 1.85539e8;
    let mimas_v = (G * saturn_mass / mimas_dist).sqrt();
    sim.add_body(Body::new_3d(mimas_dist, saturn_y, saturn_z, saturn_vx, mimas_v, 0.0, 3.75e19, 1.0));
    
    // Enceladus: 237,948 km, mass 1.08e20 kg
    let enceladus_dist = 2.37948e8;
    let enceladus_v = (G * saturn_mass / enceladus_dist).sqrt();
    sim.add_body(Body::new_3d(-enceladus_dist, saturn_y, saturn_z, saturn_vx, -enceladus_v, 0.0, 1.08e20, 1.0));
    
    // Tethys: 294,619 km, mass 6.175e20 kg
    let tethys_dist = 2.94619e8;
    let tethys_v = (G * saturn_mass / tethys_dist).sqrt();
    sim.add_body(Body::new_3d(0.0, saturn_y + tethys_dist, saturn_z, saturn_vx - tethys_v, 0.0, 0.0, 6.175e20, 1.2));
    
    // Dione: 377,396 km, mass 1.095e21 kg
    let dione_dist = 3.77396e8;
    let dione_v = (G * saturn_mass / dione_dist).sqrt();
    sim.add_body(Body::new_3d(0.0, saturn_y - dione_dist, saturn_z, saturn_vx + dione_v, 0.0, 0.0, 1.095e21, 1.3));
    
    // Rhea: 527,108 km, mass 2.3065e21 kg
    let rhea_dist = 5.27108e8;
    let rhea_v = (G * saturn_mass / rhea_dist).sqrt();
    sim.add_body(Body::new_3d(rhea_dist, saturn_y, saturn_z, saturn_vx, rhea_v, 0.0, 2.3065e21, 1.5));
    
    // Titan: 1,221,870 km, mass 1.3452e23 kg (largest Saturn moon, has atmosphere)
    let titan_dist = 1.22187e9;
    let titan_v = (G * saturn_mass / titan_dist).sqrt();
    sim.add_body(Body::new_3d(-titan_dist, saturn_y, saturn_z, saturn_vx, -titan_v, 0.0, 1.3452e23, 3.0));
    
    // Iapetus: 3,560,820 km, mass 1.8056e21 kg
    let iapetus_dist = 3.56082e9;
    let iapetus_v = (G * saturn_mass / iapetus_dist).sqrt();
    sim.add_body(Body::new_3d(0.0, saturn_y + iapetus_dist, saturn_z, saturn_vx - iapetus_v, 0.0, 0.0, 1.8056e21, 1.5));
    
    // ========== URANUS SYSTEM ==========
    let uranus_dist = 2.867e12;
    let uranus_v = (G * sun_mass / uranus_dist).sqrt();
    let uranus_mass = 8.681e25;
    let uranus_x = -uranus_dist;
    let uranus_vy = -uranus_v * incl_uranus.cos();
    let uranus_vz = -uranus_v * incl_uranus.sin();
    sim.add_body(Body::new_3d(uranus_x, 0.0, 0.0, 0.0, uranus_vy, uranus_vz, uranus_mass, 10.0));
    
    // Miranda: 129,390 km, mass 6.6e19 kg
    let miranda_dist = 1.2939e8;
    let miranda_v = (G * uranus_mass / miranda_dist).sqrt();
    sim.add_body(Body::new_3d(uranus_x, miranda_dist, 0.0, -miranda_v, uranus_vy, uranus_vz, 6.6e19, 1.0));
    
    // Ariel: 190,900 km, mass 1.29e21 kg
    let ariel_dist = 1.909e8;
    let ariel_v = (G * uranus_mass / ariel_dist).sqrt();
    sim.add_body(Body::new_3d(uranus_x, -ariel_dist, 0.0, ariel_v, uranus_vy, uranus_vz, 1.29e21, 1.3));
    
    // Umbriel: 266,000 km, mass 1.28e21 kg
    let umbriel_dist = 2.66e8;
    let umbriel_v = (G * uranus_mass / umbriel_dist).sqrt();
    sim.add_body(Body::new_3d(uranus_x + umbriel_dist, 0.0, 0.0, 0.0, uranus_vy + umbriel_v, uranus_vz, 1.28e21, 1.3));
    
    // Titania: 436,300 km, mass 3.42e21 kg (largest Uranus moon)
    let titania_dist = 4.363e8;
    let titania_v = (G * uranus_mass / titania_dist).sqrt();
    sim.add_body(Body::new_3d(uranus_x - titania_dist, 0.0, 0.0, 0.0, uranus_vy - titania_v, uranus_vz, 3.42e21, 1.6));
    
    // Oberon: 583,500 km, mass 3.08e21 kg
    let oberon_dist = 5.835e8;
    let oberon_v = (G * uranus_mass / oberon_dist).sqrt();
    sim.add_body(Body::new_3d(uranus_x, oberon_dist, 0.0, -oberon_v, uranus_vy, uranus_vz, 3.08e21, 1.5));
    
    // ========== NEPTUNE SYSTEM ==========
    let neptune_dist = 4.515e12;
    let neptune_v = (G * sun_mass / neptune_dist).sqrt();
    let neptune_mass = 1.024e26;
    let neptune_y = -neptune_dist * incl_neptune.cos();
    let neptune_z = -neptune_dist * incl_neptune.sin();
    let neptune_vx = neptune_v;
    sim.add_body(Body::new_3d(0.0, neptune_y, neptune_z, neptune_vx, 0.0, 0.0, neptune_mass, 10.0));
    
    // Triton: 354,759 km, mass 2.14e22 kg (RETROGRADE orbit - only large moon with retrograde)
    let triton_dist = 3.54759e8;
    let triton_v = (G * neptune_mass / triton_dist).sqrt();
    // Retrograde: velocity opposite to normal prograde direction
    sim.add_body(Body::new_3d(triton_dist, neptune_y, neptune_z, neptune_vx, triton_v, 0.0, 2.14e22, 2.5));

    sim
}

// Preset: Binary stars
#[wasm_bindgen]
pub fn create_binary_star() -> Simulation {
    let mut sim = Simulation::new();
    
    let star_mass = 1.989e30;
    let separation = 100.0e9;
    let r = separation / 2.0;
    
    // Each star orbits center of mass at distance r = separation/2
    // For equal masses: v = sqrt(G * m / (2 * separation))
    let orbital_v = (G * star_mass / (2.0 * separation)).sqrt();
    
    // Stars on x-axis, velocities in y-direction for circular orbit
    sim.add_body(Body::new(-r, 0.0, 0.0, -orbital_v, star_mass, 25.0));
    sim.add_body(Body::new(r, 0.0, 0.0, orbital_v, star_mass, 25.0));
    
    // Circumbinary planet - needs to orbit total mass of both stars
    let planet_dist = 400.0e9;
    let planet_v = (G * 2.0 * star_mass / planet_dist).sqrt();
    sim.add_body(Body::new(planet_dist, 0.0, 0.0, planet_v, 5.972e24, 8.0));

    sim
}

// Preset: Figure-8 three body
#[wasm_bindgen]
pub fn create_figure_eight() -> Simulation {
    let mut sim = Simulation::new();
    
    // Scaled figure-8 solution (Chenciner-Montgomery)
    let mass = 1.0e30;
    let scale = 100.0e9;
    let v_scale = (G * mass / scale).sqrt();
    
    sim.add_body(Body::new(
        -0.97000436 * scale, 0.24308753 * scale,
        0.4662036850 * v_scale, 0.4323657300 * v_scale,
        mass, 20.0
    ));
    sim.add_body(Body::new(
        0.97000436 * scale, -0.24308753 * scale,
        0.4662036850 * v_scale, 0.4323657300 * v_scale,
        mass, 20.0
    ));
    sim.add_body(Body::new(
        0.0, 0.0,
        -0.93240737 * v_scale, -0.86473146 * v_scale,
        mass, 20.0
    ));

    sim
}

// Preset: Neutron Star Merger
#[wasm_bindgen]
pub fn create_neutron_merger() -> Simulation {
    let mut sim = Simulation::new();
    
    // Enable gravitational wave radiation drag for realistic inspiral
    sim.set_gw_drag(true);
    
    // Two neutron stars as point masses (no collision radius)
    let ns_mass = 2.8e30;  // ~1.4 solar masses each
    let separation = 20.0e9;  // 20 million km apart (closer for faster inspiral)
    
    // Circular orbit velocity - GW drag will cause inspiral
    let v_circular = (G * ns_mass / separation).sqrt();
    
    sim.add_body(Body::new_point_mass(-separation/2.0, 0.0, 0.0, 0.0, -v_circular/2.0, 0.0, ns_mass, 6.0));
    sim.add_body(Body::new_point_mass(separation/2.0, 0.0, 0.0, 0.0, v_circular/2.0, 0.0, ns_mass, 6.0));
    
    sim
}

// Preset: Chaotic 3-body
#[wasm_bindgen]
pub fn create_chaos() -> Simulation {
    let mut sim = Simulation::new();
    
    let mass = 1.0e30;
    let dist = 100.0e9;
    let v_base = (G * mass / dist).sqrt();
    
    sim.add_body(Body::new(dist, 0.0, 0.0, v_base * 0.6, mass, 18.0));
    sim.add_body(Body::new(-dist * 0.5, dist * 0.866, -v_base * 0.5, -v_base * 0.3, mass, 18.0));
    sim.add_body(Body::new(-dist * 0.5, -dist * 0.866, v_base * 0.4, -v_base * 0.2, mass * 1.5, 22.0));

    sim
}

// Preset: Lagrange Points
// Demonstrates the 5 equilibrium points in a Sun-Earth-like system
#[wasm_bindgen]
pub fn create_lagrange() -> Simulation {
    let mut sim = Simulation::new();
    
    let sun_mass = 1.989e30;
    let earth_mass = 5.972e24;
    let earth_dist = 150.0e9;  // 1 AU
    
    // Sun at origin
    sim.add_body(Body::new(0.0, 0.0, 0.0, 0.0, sun_mass, 30.0));
    
    // Earth on positive x-axis, orbiting counterclockwise (velocity in +y)
    let earth_v = (G * sun_mass / earth_dist).sqrt();
    sim.add_body(Body::new(earth_dist, 0.0, 0.0, earth_v, earth_mass, 10.0));
    
    // Mass ratio parameter for L1/L2/L3 calculations
    let mu = earth_mass / (sun_mass + earth_mass);  // ~3e-6
    let alpha = (mu / 3.0_f64).powf(1.0 / 3.0);     // Hill sphere ratio
    
    // Angular velocity of Earth (L-points must co-rotate with Earth)
    let omega = earth_v / earth_dist;
    
    // Test mass for L-point objects (very light, won't perturb system)
    let test_mass = 1000.0;  // 1 ton - essentially massless
    
    // L1: Between Sun and Earth (on Sun-Earth line, sunward of Earth)
    // Distance from Sun: r * (1 - alpha)
    let l1_dist = earth_dist * (1.0 - alpha);
    let l1_v = omega * l1_dist;  // Co-rotate with Earth
    sim.add_body(Body::new(l1_dist, 0.0, 0.0, l1_v, test_mass, 5.0));
    
    // L2: Beyond Earth (on Sun-Earth line, anti-sunward of Earth)
    // Distance from Sun: r * (1 + alpha)
    let l2_dist = earth_dist * (1.0 + alpha);
    let l2_v = omega * l2_dist;  // Co-rotate with Earth
    sim.add_body(Body::new(l2_dist, 0.0, 0.0, l2_v, test_mass, 5.0));
    
    // L3: Opposite side of Sun from Earth
    let l3_dist = earth_dist * (1.0 + 5.0 * mu / 12.0);
    let l3_v = omega * l3_dist;  // Co-rotate with Earth
    sim.add_body(Body::new(-l3_dist, 0.0, 0.0, -l3_v, test_mass, 5.0));
    
    // L4: 60° ahead of Earth (leading Trojan point) - STABLE
    // Forms equilateral triangle with Sun and Earth
    let angle_l4 = std::f64::consts::PI / 3.0;  // 60°
    let l4_x = earth_dist * angle_l4.cos();
    let l4_y = earth_dist * angle_l4.sin();
    // Velocity perpendicular to radius, same speed as Earth
    let l4_vx = -earth_v * angle_l4.sin();
    let l4_vy = earth_v * angle_l4.cos();
    sim.add_body(Body::new(l4_x, l4_y, l4_vx, l4_vy, test_mass, 5.0));
    
    // L5: 60° behind Earth (trailing Trojan point) - STABLE
    let angle_l5 = -std::f64::consts::PI / 3.0;  // -60°
    let l5_x = earth_dist * angle_l5.cos();
    let l5_y = earth_dist * angle_l5.sin();
    let l5_vx = -earth_v * angle_l5.sin();
    let l5_vy = earth_v * angle_l5.cos();
    sim.add_body(Body::new(l5_x, l5_y, l5_vx, l5_vy, test_mass, 5.0));
    
    sim
}

// Preset: Galaxy Collision
// Two "galaxies" each with a massive central body and orbiting stars
// Tuned for a realistic merger where most mass stays bound
#[wasm_bindgen]
pub fn create_galaxy_collision() -> Simulation {
    let mut sim = Simulation::new();
    
    let core_mass = 2.0e32;  // Massive central body (increased for stronger binding)
    let star_mass = 5.0e27;  // Smaller "star" mass (less likely to perturb cores)
    let galaxy_radius = 150.0e9;  // Galaxy radius
    let separation = 500.0e9;  // Initial separation (closer = more bound)
    
    // Calculate a gentle approach velocity (sub-escape, will merge not fly-by)
    // Escape velocity from combined mass at this separation
    let escape_v = (2.0 * G * 2.0 * core_mass / separation).sqrt();
    let approach_v = escape_v * 0.3;  // 30% of escape velocity - gentle approach
    
    // Simple pseudo-random number generator (deterministic for reproducibility)
    let mut seed: u64 = 12345;
    let mut random = || -> f64 {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((seed >> 16) & 0x7fff) as f64 / 32768.0  // Returns 0.0 to 1.0
    };
    
    // Galaxy 1 - slight tangential velocity for spiral-in rather than head-on
    let g1_x = -separation / 2.0;
    let tangent_v = approach_v * 0.5;  // Some tangential motion
    sim.add_body(Body::new_3d(g1_x, 0.0, 0.0, approach_v, tangent_v, 0.0, core_mass, 20.0));
    
    // Stars orbiting galaxy 1
    let stars_per_galaxy = 30;
    for i in 0..stars_per_galaxy {
        // Base angle with randomness
        let base_angle = 2.0 * std::f64::consts::PI * (i as f64) / (stars_per_galaxy as f64);
        let angle = base_angle + (random() - 0.5) * 0.5;  // ±0.25 radians jitter
        
        // Radius with randomness - keep stars closer to core
        let base_r = galaxy_radius * (0.2 + 0.6 * ((i % 3) as f64 + 1.0) / 3.0);
        let r = base_r * (0.85 + 0.3 * random());  // 85% to 115% of base radius
        
        // Position with Z offset for 3D thickness
        let z = galaxy_radius * 0.12 * (random() - 0.5);  // ±6% of radius in Z
        let x = g1_x + r * angle.cos();
        let y = r * angle.sin();
        
        // Orbital velocity with slight inclination
        let v = (G * core_mass / r).sqrt();
        let inclination = 0.08 * (random() - 0.5);  // Slight orbital tilt
        let vx = approach_v - v * angle.sin();
        let vy = tangent_v + v * angle.cos() * (1.0 - inclination.abs());
        let vz = v * inclination;  // Z-component of velocity
        
        sim.add_body(Body::new_3d(x, y, z, vx, vy, vz, star_mass, 3.0));
    }
    
    // Galaxy 2 - approaching from other side, tilted galactic plane
    let g2_x = separation / 2.0;
    let g2_y = 80.0e9;  // Slight offset for off-center collision
    let g2_z = 40.0e9;
    sim.add_body(Body::new_3d(g2_x, g2_y, g2_z, -approach_v, -tangent_v * 0.8, -tangent_v * 0.3, core_mass, 20.0));
    
    // Stars orbiting galaxy 2 (rotating opposite direction, different plane)
    for i in 0..stars_per_galaxy {
        // Base angle with randomness
        let base_angle = 2.0 * std::f64::consts::PI * (i as f64) / (stars_per_galaxy as f64);
        let angle = base_angle + (random() - 0.5) * 0.5;
        
        // Radius with randomness
        let base_r = galaxy_radius * (0.2 + 0.6 * ((i % 3) as f64 + 1.0) / 3.0);
        let r = base_r * (0.85 + 0.3 * random());
        
        // Position - galaxy 2 is tilted ~25° relative to galaxy 1
        let tilt: f64 = 0.4;  // ~25 degrees in radians
        let x = g2_x + r * angle.cos();
        let y_flat = r * angle.sin();
        let y = g2_y + y_flat * tilt.cos();
        let z = g2_z + y_flat * tilt.sin() + galaxy_radius * 0.08 * (random() - 0.5);
        
        // Orbital velocity (opposite rotation, tilted plane)
        let v = (G * core_mass / r).sqrt();
        let inclination = 0.08 * (random() - 0.5);
        let vx = -approach_v + v * angle.sin();
        let vy = -tangent_v * 0.8 - v * angle.cos() * tilt.cos() * (1.0 - inclination.abs());
        let vz = -tangent_v * 0.3 - v * angle.cos() * tilt.sin() + v * inclination;
        
        sim.add_body(Body::new_3d(x, y, z, vx, vy, vz, star_mass, 3.0));
    }
    
    sim
}
