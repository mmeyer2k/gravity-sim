use wasm_bindgen::prelude::*;

const G: f64 = 6.674e-11; // Gravitational constant
const C: f64 = 2.998e8;   // Speed of light
const SOFTENING: f64 = 1e8; // Softening to prevent division issues at close range

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
    // - Neutron stars: ~4e17 kg/m³
    
    let density = if mass < 1e20 {
        3000.0  // Asteroid
    } else if mass < 1e25 {
        5000.0  // Rocky planet
    } else if mass < 1e28 {
        1300.0  // Gas giant
    } else if mass < 1e31 {
        1400.0  // Star
    } else {
        4.0e17  // Neutron star (super dense)
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

    /// Step simulation using leapfrog integration (stable for orbital mechanics)
    /// dt is in seconds
    pub fn step(&mut self, dt: f64) {
        let n = self.bodies.len();
        if n == 0 {
            return;
        }

        // Calculate accelerations for all bodies (3D)
        let mut ax = vec![0.0; n];
        let mut ay = vec![0.0; n];
        let mut az = vec![0.0; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dist = dist_sq.sqrt();
                let dist_soft = (dist_sq + SOFTENING * SOFTENING).sqrt();
                
                // F = G * m1 * m2 / r^2
                // a1 = G * m2 / r^2 (toward body 2)
                let accel_mag = G * self.gravity_mult / (dist_soft * dist_soft);
                
                // Unit vector from i to j
                let ux = dx / dist;
                let uy = dy / dist;
                let uz = dz / dist;
                
                // Accelerations (a = G * m_other / r^2)
                ax[i] += accel_mag * self.bodies[j].mass * ux;
                ay[i] += accel_mag * self.bodies[j].mass * uy;
                az[i] += accel_mag * self.bodies[j].mass * uz;
                ax[j] -= accel_mag * self.bodies[i].mass * ux;
                ay[j] -= accel_mag * self.bodies[i].mass * uy;
                az[j] -= accel_mag * self.bodies[i].mass * uz;
            }
        }

        // Symplectic Euler integration (3D)
        for i in 0..n {
            // Update velocity
            self.bodies[i].vx += ax[i] * dt;
            self.bodies[i].vy += ay[i] * dt;
            self.bodies[i].vz += az[i] * dt;
            
            // Update position
            self.bodies[i].x += self.bodies[i].vx * dt;
            self.bodies[i].y += self.bodies[i].vy * dt;
            self.bodies[i].z += self.bodies[i].vz * dt;
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
    
    /// Apply gravitational wave radiation energy loss (Peters formula approximation)
    /// This causes orbits to decay over time, simulating inspiral
    fn apply_gw_drag(&mut self, dt: f64) {
        let n = self.bodies.len();
        // Pre-factor: 32/5 * G^4 / c^5
        let gw_factor = (32.0 / 5.0) * G.powi(4) / C.powi(5);
        
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = self.bodies[j].x - self.bodies[i].x;
                let dy = self.bodies[j].y - self.bodies[i].y;
                let dz = self.bodies[j].z - self.bodies[i].z;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                let dist = dist_sq.sqrt();
                
                let m1 = self.bodies[i].mass;
                let m2 = self.bodies[j].mass;
                
                // Power radiated: P = (32/5) * (G^4/c^5) * (m1*m2)^2 * (m1+m2) / r^5
                let power = gw_factor * (m1 * m2).powi(2) * (m1 + m2) / dist.powi(5);
                
                // Relative velocity
                let dvx = self.bodies[j].vx - self.bodies[i].vx;
                let dvy = self.bodies[j].vy - self.bodies[i].vy;
                let dvz = self.bodies[j].vz - self.bodies[i].vz;
                let rel_speed_sq = dvx * dvx + dvy * dvy + dvz * dvz;
                let rel_speed = rel_speed_sq.sqrt();
                
                if rel_speed < 1.0 { continue; }
                
                // Total kinetic energy in CM frame
                let reduced_mass = (m1 * m2) / (m1 + m2);
                let kinetic_energy = 0.5 * reduced_mass * rel_speed_sq;
                
                if kinetic_energy < 1.0 { continue; }
                
                // Energy loss this timestep
                let energy_loss = power * dt;
                
                // Fraction of energy to remove (capped to prevent instability)
                let fraction = (energy_loss / kinetic_energy).min(0.01);
                
                // Reduce relative velocity (apply as drag toward each other)
                // This removes energy from the orbit, causing inspiral
                let drag = fraction * 0.5;  // Split between both bodies
                
                // Apply drag (reduce relative velocity)
                self.bodies[i].vx += drag * dvx * m2 / (m1 + m2);
                self.bodies[i].vy += drag * dvy * m2 / (m1 + m2);
                self.bodies[i].vz += drag * dvz * m2 / (m1 + m2);
                self.bodies[j].vx -= drag * dvx * m1 / (m1 + m2);
                self.bodies[j].vy -= drag * dvy * m1 / (m1 + m2);
                self.bodies[j].vz -= drag * dvz * m1 / (m1 + m2);
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

// Preset: Solar system
#[wasm_bindgen]
pub fn create_solar_system() -> Simulation {
    let mut sim = Simulation::new();
    
    let sun_mass = 1.989e30;
    sim.add_body(Body::new(0.0, 0.0, 0.0, 0.0, sun_mass, 30.0));
    
    // Earth: 1 AU = 150e9 m, orbital velocity ~29,780 m/s
    let earth_dist = 150.0e9;
    let earth_v = (G * sun_mass / earth_dist).sqrt();
    sim.add_body(Body::new(earth_dist, 0.0, 0.0, earth_v, 5.972e24, 8.0));
    
    // Mars: 1.52 AU
    let mars_dist = 228.0e9;
    let mars_v = (G * sun_mass / mars_dist).sqrt();
    sim.add_body(Body::new(0.0, mars_dist, -mars_v, 0.0, 6.39e23, 6.0));
    
    // Venus: 0.72 AU
    let venus_dist = 108.0e9;
    let venus_v = (G * sun_mass / venus_dist).sqrt();
    sim.add_body(Body::new(-venus_dist, 0.0, 0.0, -venus_v, 4.867e24, 7.0));
    
    // Mercury: 0.39 AU
    let mercury_dist = 58.0e9;
    let mercury_v = (G * sun_mass / mercury_dist).sqrt();
    sim.add_body(Body::new(0.0, -mercury_dist, mercury_v, 0.0, 3.285e23, 4.0));

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
    
    // L1: Between Sun and Earth (closer to Earth)
    // Distance from Earth toward Sun: r * alpha (approx)
    let l1_dist = earth_dist * (1.0 - alpha);
    let l1_v = (G * sun_mass / l1_dist).sqrt();  // Orbital velocity at L1
    sim.add_body(Body::new(l1_dist, 0.0, 0.0, l1_v, 1e15, 5.0));
    
    // L2: Beyond Earth (away from Sun)
    let l2_dist = earth_dist * (1.0 + alpha);
    let l2_v = (G * sun_mass / l2_dist).sqrt();
    sim.add_body(Body::new(l2_dist, 0.0, 0.0, l2_v, 1e15, 5.0));
    
    // L3: Opposite side of Sun from Earth
    // Slightly outside Earth's orbit
    let l3_dist = -earth_dist * (1.0 + 5.0 * mu / 12.0);
    let l3_v = (G * sun_mass / earth_dist).sqrt();  // ~same orbital velocity
    sim.add_body(Body::new(l3_dist, 0.0, 0.0, -l3_v, 1e15, 5.0));
    
    // L4: 60° ahead of Earth (leading Trojan point)
    // Forms equilateral triangle with Sun and Earth
    let angle_l4 = std::f64::consts::PI / 3.0;  // 60°
    let l4_x = earth_dist * angle_l4.cos();
    let l4_y = earth_dist * angle_l4.sin();
    // Velocity perpendicular to radius, same speed as Earth
    let l4_vx = -earth_v * angle_l4.sin();
    let l4_vy = earth_v * angle_l4.cos();
    sim.add_body(Body::new(l4_x, l4_y, l4_vx, l4_vy, 1e15, 5.0));
    
    // L5: 60° behind Earth (trailing Trojan point)
    let angle_l5 = -std::f64::consts::PI / 3.0;  // -60°
    let l5_x = earth_dist * angle_l5.cos();
    let l5_y = earth_dist * angle_l5.sin();
    let l5_vx = -earth_v * angle_l5.sin();
    let l5_vy = earth_v * angle_l5.cos();
    sim.add_body(Body::new(l5_x, l5_y, l5_vx, l5_vy, 1e15, 5.0));
    
    sim
}
