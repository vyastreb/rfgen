"""
Asphalt Concrete Surface Model - Gravity Settling with 2-Layer Packing.

Features:
1. Efficient gravity-based settling (drop from above)
2. Aggregates oriented with smallest axis vertical (normal to surface)
3. 2-layer packing - stops when second layer is assembled
4. Asphalt roller simulation - flattens top surface grains

Accelerated with Numba for performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import gaussian_filter
from numba import jit, prange
import time


# ============================================================================
# Numba-accelerated Rendering
# ============================================================================

@jit(nopython=True, cache=True)
def barycentric_coords(px, py, v0x, v0y, v1x, v1y, v2x, v2y):
    """Compute barycentric coordinates for point p relative to triangle."""
    denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
    if abs(denom) < 1e-10:
        return -1.0, -1.0, -1.0
    w0 = ((v1y - v2y) * (px - v2x) + (v2x - v1x) * (py - v2y)) / denom
    w1 = ((v2y - v0y) * (px - v2x) + (v0x - v2x) * (py - v2y)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


@jit(nopython=True, parallel=True, cache=True)
def render_mesh_to_heightmap(vertices, faces, Lx, Ly, Nx, Ny):
    """Parallelized Z-buffer rasterizer."""
    Z = np.full((Ny, Nx), -100.0, dtype=np.float64)
    dx = Lx / Nx
    dy = Ly / Ny
    n_faces = len(faces)
    
    for i in prange(n_faces):
        f = faces[i]
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]
        
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        
        ix_min = max(0, int(min_x / dx))
        ix_max = min(Nx - 1, int(max_x / dx) + 1)
        iy_min = max(0, int(min_y / dy))
        iy_max = min(Ny - 1, int(max_y / dy) + 1)
        
        for iy in range(iy_min, iy_max + 1):
            py = (iy + 0.5) * dy
            for ix in range(ix_min, ix_max + 1):
                px = (ix + 0.5) * dx
                w0, w1, w2 = barycentric_coords(px, py, v0[0], v0[1], v1[0], v1[1], v2[0], v2[1])
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    z_val = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                    if z_val > Z[iy, ix]:
                        Z[iy, ix] = z_val
    return Z


# ============================================================================
# Aggregate Mesh Generation (smallest axis = Z)
# ============================================================================

def generate_aggregate_mesh(center, size, axis_ratio, irregularity, n_control_points, rng):
    """
    Generate stone mesh with smallest axis aligned to Z (vertical).
    Only rotation around Z (yaw) is applied.
    """
    # Sort axes: smallest becomes Z
    sorted_axes = np.sort(np.array(axis_ratio, dtype=float))
    # Map: [Large, Medium, Small] -> X=Large, Y=Medium, Z=Small
    ar = np.array([sorted_axes[2], sorted_axes[1], sorted_axes[0]])
    ar = ar / np.max(ar) * (size / 2.0)
    
    # Generate control points on unit sphere
    u = rng.uniform(0, 1, n_control_points)
    v = rng.uniform(0, 1, n_control_points)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=1)
    
    # Scale to ellipsoid
    points[:, 0] *= ar[0]
    points[:, 1] *= ar[1]
    points[:, 2] *= ar[2]
    
    # Add irregularity
    noise = rng.uniform(1.0 - irregularity, 1.0 + irregularity, (n_control_points, 1))
    points *= noise
    
    # Rotate around Z only (yaw)
    angle = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(angle), np.sin(angle)
    new_x = points[:, 0] * c - points[:, 1] * s
    new_y = points[:, 0] * s + points[:, 1] * c
    points[:, 0] = new_x
    points[:, 1] = new_y
    
    # Convex hull
    try:
        hull = ConvexHull(points)
        vertices = points + center
        faces = hull.simplices
        return vertices, faces, hull
    except:
        return None, None, None


# ============================================================================
# Spatial Grid for Fast Collision Detection
# ============================================================================

class SpatialGrid3D:
    """Grid-based spatial indexing for fast neighbor queries."""
    
    def __init__(self, Lx, Ly, cell_size):
        self.cell_size = cell_size
        self.Lx = Lx
        self.Ly = Ly
        self.cols = max(1, int(np.ceil(Lx / cell_size)))
        self.rows = max(1, int(np.ceil(Ly / cell_size)))
        self.grid = {}  # (r, c) -> list of aggregate indices
    
    def _get_cell(self, x, y):
        c = int(x / self.cell_size) % self.cols
        r = int(y / self.cell_size) % self.rows
        return r, c
    
    def insert(self, x, y, agg_idx):
        r, c = self._get_cell(x, y)
        if (r, c) not in self.grid:
            self.grid[(r, c)] = []
        self.grid[(r, c)].append(agg_idx)
    
    def get_nearby_indices(self, x, y, search_radius):
        """Get indices of aggregates in cells within search_radius."""
        indices = []
        n_cells = int(np.ceil(search_radius / self.cell_size)) + 1
        
        center_r, center_c = self._get_cell(x, y)
        
        for dr in range(-n_cells, n_cells + 1):
            for dc in range(-n_cells, n_cells + 1):
                r = (center_r + dr) % self.rows
                c = (center_c + dc) % self.cols
                if (r, c) in self.grid:
                    indices.extend(self.grid[(r, c)])
        
        return indices


# ============================================================================
# Gravity Settling Algorithm with 2-Layer Tracking
# ============================================================================

def find_rest_height(x, y, radius_xy, radius_z, aggregates, spatial_grid, Lx, Ly):
    """
    Find the rest height for a new aggregate at position (x, y).
    Returns (rest_z, new_layer) where:
    - rest_z is the z-coordinate for the aggregate center
    - new_layer is 1 (resting on ground) or 2 (resting on layer 1 particles)
    """
    # New aggregate's bottom will be at center_z - radius_z
    # We need to find max surface height below (x, y)
    
    if not aggregates:
        # Rest on ground → layer 1
        return radius_z, 1
    
    # Get nearby aggregates
    search_radius = radius_xy * 3
    nearby_indices = spatial_grid.get_nearby_indices(x, y, search_radius)
    
    max_surface_z = 0.0  # Ground level
    supporting_agg_layer = 0  # 0 = ground
    
    for idx in nearby_indices:
        agg = aggregates[idx]
        ax, ay = agg['center'][0], agg['center'][1]
        
        # Check with periodic boundaries
        for dx in [0, Lx, -Lx]:
            for dy in [0, Ly, -Ly]:
                px, py = ax + dx, ay + dy
                
                # Distance in XY plane
                dist_xy = np.sqrt((x - px)**2 + (y - py)**2)
                
                # Check if we overlap in XY (simplified: use bounding circles)
                overlap_threshold = radius_xy + agg['radius_xy']
                
                if dist_xy < overlap_threshold:
                    # This aggregate is below us - find its top surface
                    agg_top_z = agg['z_top']
                    
                    if agg_top_z > max_surface_z:
                        max_surface_z = agg_top_z
                        supporting_agg_layer = agg['layer']
    
    # Rest height: center is at surface + radius_z
    rest_z = max_surface_z + radius_z
    
    # New layer = supporting layer + 1
    # Ground (0) → layer 1, layer 1 particle → layer 2, etc.
    new_layer = supporting_agg_layer + 1
    
    return rest_z, new_layer


def pack_aggregates_gravity_2layers(Lx, Ly, mean_size, size_std, axis_ratio, 
                                     irregularity, gap_min, n_control_points, 
                                     rng, target_coverage=0.7, max_aggregates=500):
    """
    Pack aggregates using gravity settling in 2 phases:
    Phase 1: Fill layer 1 (particles on ground)
    Phase 2: Fill layer 2 (particles on top of layer 1)
    
    Returns list of aggregates with: vertices, faces, center, layer, z_bottom, z_top, etc.
    """
    aggregates = []
    
    # Cell size based on mean aggregate size
    cell_size = mean_size * 2.0
    spatial_grid = SpatialGrid3D(Lx, Ly, cell_size)
    
    # Estimate target counts
    # For flat particles (1:2:2), effective XY area is larger
    effective_radius = mean_size / 2 * np.max(axis_ratio) / np.mean(axis_ratio)
    particle_area = np.pi * effective_radius**2
    domain_area = Lx * Ly
    target_layer1 = int(domain_area / particle_area * target_coverage)
    target_layer2 = int(target_layer1 * 1.2)  # Dense second layer (can have more)
    
    print(f"  Target: ~{target_layer1} in layer 1, ~{target_layer2} in layer 2")
    
    layer1_count = 0
    layer2_count = 0
    
    # ===== PHASE 1: Fill Layer 1 =====
    print("  Phase 1: Filling layer 1...")
    attempts = 0
    max_attempts = target_layer1 * 50  # More attempts for dense packing
    
    while layer1_count < target_layer1 and attempts < max_attempts:
        attempts += 1
        
        size = max(mean_size * 0.3, rng.normal(mean_size, size_std))
        sorted_axes = np.sort(np.array(axis_ratio, dtype=float))
        scale = size / np.max(sorted_axes)
        radius_xy = sorted_axes[2] * scale / 2
        radius_z = sorted_axes[0] * scale / 2
        
        x = rng.uniform(0, Lx)
        y = rng.uniform(0, Ly)
        
        rest_z, new_layer = find_rest_height(x, y, radius_xy, radius_z, aggregates, spatial_grid, Lx, Ly)
        
        # Only accept layer 1 placements in phase 1
        if new_layer != 1:
            continue
        
        center = np.array([x, y, rest_z])
        vertices, faces, hull = generate_aggregate_mesh(center, size, axis_ratio, irregularity, n_control_points, rng)
        
        if vertices is None:
            continue
        
        # Check overlaps - only with same-layer aggregates
        overlap = False
        nearby_indices = spatial_grid.get_nearby_indices(x, y, radius_xy * 3)
        
        for idx in nearby_indices:
            agg = aggregates[idx]
            if agg['layer'] != 1:  # Only check layer 1 for phase 1
                continue
            ax, ay, az = agg['center']
            for dx in [0, Lx, -Lx]:
                for dy in [0, Ly, -Ly]:
                    px, py = ax + dx, ay + dy
                    # XY distance only for same-layer collision
                    dist_xy = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = radius_xy + agg['radius_xy'] + gap_min
                    if dist_xy < min_dist * 0.65:  # Very tight packing
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                break
        
        if overlap:
            continue
        
        # Place aggregate
        z_bottom = np.min(vertices[:, 2])
        z_top = np.max(vertices[:, 2])
        
        agg_data = {
            'vertices': vertices, 'faces': faces, 'center': center,
            'size': size, 'layer': 1, 'z_bottom': z_bottom, 'z_top': z_top,
            'radius_xy': radius_xy, 'radius_z': radius_z
        }
        
        aggregates.append(agg_data)
        spatial_grid.insert(x, y, len(aggregates) - 1)
        layer1_count += 1
        
        if layer1_count % 20 == 0:
            print(f"    Layer 1: {layer1_count}/{target_layer1}")
    
    print(f"  Layer 1 complete: {layer1_count} aggregates")
    
    # ===== PHASE 2: Fill Layer 2 =====
    print("  Phase 2: Filling layer 2...")
    attempts = 0
    max_attempts = target_layer2 * 100  # Many more attempts for dense layer 2
    
    while layer2_count < target_layer2 and attempts < max_attempts:
        attempts += 1
        
        size = max(mean_size * 0.3, rng.normal(mean_size, size_std))
        sorted_axes = np.sort(np.array(axis_ratio, dtype=float))
        scale = size / np.max(sorted_axes)
        radius_xy = sorted_axes[2] * scale / 2
        radius_z = sorted_axes[0] * scale / 2
        
        x = rng.uniform(0, Lx)
        y = rng.uniform(0, Ly)
        
        rest_z, new_layer = find_rest_height(x, y, radius_xy, radius_z, aggregates, spatial_grid, Lx, Ly)
        
        # Only accept layer 2 placements in phase 2
        if new_layer != 2:
            continue
        
        center = np.array([x, y, rest_z])
        vertices, faces, hull = generate_aggregate_mesh(center, size, axis_ratio, irregularity, n_control_points, rng)
        
        if vertices is None:
            continue
        
        # Check overlaps - only with layer 2 aggregates (can overlap with layer 1)
        overlap = False
        nearby_indices = spatial_grid.get_nearby_indices(x, y, radius_xy * 3)
        
        for idx in nearby_indices:
            agg = aggregates[idx]
            if agg['layer'] != 2:  # Only check layer 2 for phase 2
                continue
            ax, ay, az = agg['center']
            for dx in [0, Lx, -Lx]:
                for dy in [0, Ly, -Ly]:
                    px, py = ax + dx, ay + dy
                    # XY distance only for same-layer collision
                    dist_xy = np.sqrt((x - px)**2 + (y - py)**2)
                    min_dist = radius_xy + agg['radius_xy'] + gap_min
                    if dist_xy < min_dist * 0.55:  # Very tight for layer 2
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                break
        
        if overlap:
            continue
        
        # Place aggregate
        z_bottom = np.min(vertices[:, 2])
        z_top = np.max(vertices[:, 2])
        
        agg_data = {
            'vertices': vertices, 'faces': faces, 'center': center,
            'size': size, 'layer': 2, 'z_bottom': z_bottom, 'z_top': z_top,
            'radius_xy': radius_xy, 'radius_z': radius_z
        }
        
        aggregates.append(agg_data)
        spatial_grid.insert(x, y, len(aggregates) - 1)
        layer2_count += 1
        
        if layer2_count % 20 == 0:
            print(f"    Layer 2: {layer2_count}/{target_layer2}")
    
    print(f"  Layer 2 complete: {layer2_count} aggregates")
    print(f"  Total: {len(aggregates)} aggregates (L1: {layer1_count}, L2: {layer2_count})")
    
    return aggregates


# ============================================================================
# Asphalt Roller Simulation
# ============================================================================

def apply_asphalt_roller(aggregates, target_z=None):
    """
    Simulate asphalt roller: push surface grains down so their tops align.
    
    If target_z is None, uses the average top z of layer 2 particles.
    Layer 2 particles can penetrate into layer 1 particles (this is fine).
    """
    if not aggregates:
        return aggregates
    
    # Find layer 2 particles (surface layer)
    layer2_particles = [agg for agg in aggregates if agg['layer'] == 2]
    
    if not layer2_particles:
        print("  No layer 2 particles to roll.")
        return aggregates
    
    # Determine target z (alignment height for tops)
    if target_z is None:
        # Use the average top z of layer 2
        top_zs = [agg['z_top'] for agg in layer2_particles]
        target_z = np.mean(top_zs)
    
    print(f"  Roller target z_top = {target_z:.3f}")
    
    # Push down each layer 2 particle so its top aligns with target_z
    for agg in aggregates:
        if agg['layer'] == 2:
            current_top = agg['z_top']
            
            if current_top > target_z:
                # Push down
                dz = target_z - current_top
                
                # Update all z coordinates
                agg['vertices'][:, 2] += dz
                agg['center'][2] += dz
                agg['z_top'] += dz
                agg['z_bottom'] += dz
    
    return aggregates


# ============================================================================
# Visualization
# ============================================================================

def plot_aggregates_3d(aggregates, Lx, Ly, ax=None, alpha=0.7, color_by_layer=True):
    """Plot aggregates in 3D, optionally colored by layer."""
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    layer_colors = {1: 'steelblue', 2: 'coral'}
    
    for agg in aggregates:
        vertices = agg['vertices']
        faces = agg['faces']
        
        if faces is not None:
            if color_by_layer:
                color = layer_colors.get(agg['layer'], 'gray')
            else:
                color = plt.cm.viridis(agg['z_top'] / 3.0)
            
            poly3d = [[vertices[idx] for idx in face] for face in faces]
            collection = Poly3DCollection(poly3d, alpha=alpha, 
                                          facecolor=color, edgecolor='darkgray', linewidth=0.3)
            ax.add_collection3d(collection)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    if aggregates:
        max_z = max(agg['z_top'] for agg in aggregates) * 1.2
        ax.set_zlim(0, max(max_z, 1))
    
    return ax


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    # ===== Parameters =====
    Lx, Ly = 10.0, 10.0           # Domain size
    Nx, Ny = 1024, 1024           # Height map resolution
    
    # Aggregate parameters
    mean_size = 1.0               # Mean aggregate size
    size_std = 0.2               # Size standard deviation
    axis_ratio = (1, 2, 2)        # Ellipsoid axis ratio (flat stones)
    irregularity = 0.2            # Shape irregularity
    n_control_points = 50         # Points for convex hull
    gap_min = -0.2               # Allow slight overlap (negative gap)
    
    target_coverage = 1.0         # Full coverage for layer 1
    max_aggregates = 1500          # Safety limit
    
    seed = 42
    rng = np.random.default_rng(seed)
    
    print("=" * 60)
    print("Asphalt Aggregate Model - 2-Layer Gravity Settling")
    print("=" * 60)
    print(f"Domain: {Lx} x {Ly}")
    print(f"Aggregate size: {mean_size} ± {size_std}")
    print(f"Axis ratio: {axis_ratio} (smallest axis = Z)")
    print(f"Target coverage: {target_coverage}")
    print()
    
    # ===== Step 1: Pack aggregates with gravity (2 layers) =====
    print("Step 1: Gravity settling (2 layers)...")
    t0 = time.time()
    
    aggregates = pack_aggregates_gravity_2layers(
        Lx, Ly, mean_size, size_std, axis_ratio,
        irregularity, gap_min, n_control_points,
        rng, target_coverage, max_aggregates
    )
    
    t1 = time.time()
    print(f"  Packing time: {t1-t0:.2f}s")
    
    # Show before roller
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(121, projection='3d')
    plot_aggregates_3d(aggregates, Lx, Ly, ax1, alpha=0.8)
    ax1.set_title('Before Roller (Blue=Layer1, Coral=Layer2)')
    
    # ===== Step 2: Apply asphalt roller =====
    print("\nStep 2: Applying asphalt roller...")
    
    aggregates = apply_asphalt_roller(aggregates)
    
    ax2 = fig.add_subplot(122, projection='3d')
    plot_aggregates_3d(aggregates, Lx, Ly, ax2, alpha=0.8)
    ax2.set_title('After Roller')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Step 3: Render height map =====
    print("\nStep 3: Rendering height map...")
    
    # Collect all vertices and faces (including periodic copies)
    all_verts = []
    all_faces = []
    offset = 0
    
    border_margin = mean_size * 2.0
    
    for agg in aggregates:
        v = agg['vertices']
        f = agg['faces']
        cx, cy = agg['center'][0], agg['center'][1]
        
        # Determine necessary shifts for periodicity
        shifts_x = [0]
        if cx < border_margin: shifts_x.append(Lx)
        if cx > Lx - border_margin: shifts_x.append(-Lx)
        
        shifts_y = [0]
        if cy < border_margin: shifts_y.append(Ly)
        if cy > Ly - border_margin: shifts_y.append(-Ly)
        
        for dx in shifts_x:
            for dy in shifts_y:
                v_shifted = v.copy()
                v_shifted[:, 0] += dx
                v_shifted[:, 1] += dy
                
                all_verts.append(v_shifted)
                all_faces.append(f + offset)
                offset += len(v)
    
    if all_verts:
        full_vertices = np.vstack(all_verts)
        full_faces = np.vstack(all_faces)
        
        print(f"  Total triangles: {len(full_faces)}")
        
        Z = render_mesh_to_heightmap(full_vertices, full_faces, Lx, Ly, Nx, Ny)
        
        # Set floor (binder level)
        binder_level = 0.0
        Z = np.maximum(Z, binder_level)
    else:
        Z = np.zeros((Ny, Nx))
    
    # ===== Step 4: Light smoothing =====
    print("\nStep 4: Applying smoothing...")
    sigma_metric = 0.03  # 3cm smoothing
    sigma_pixel = sigma_metric * (Nx / Lx)
    Z_smooth = gaussian_filter(Z, sigma=sigma_pixel, mode='wrap')
    
    # ===== Step 5: Visualization =====
    print("\nStep 5: Visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(Z, extent=[0, Lx, 0, Ly], origin='lower', cmap='terrain')
    ax.set_title('Height Map (Raw)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    ax = axes[1]
    im = ax.imshow(Z_smooth, extent=[0, Lx, 0, Ly], origin='lower', cmap='gray')
    ax.set_title('Height Map (Smoothed)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    plt.tight_layout()
    plt.show()
    
    # Save
    np.savez("asphalt_2layer.npz", Z=Z_smooth, Z_raw=Z, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    print(f"\nSaved: asphalt_2layer.npz")
    
    # Final stats
    layer1 = [a for a in aggregates if a['layer'] == 1]
    layer2 = [a for a in aggregates if a['layer'] == 2]
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Layer 1: {len(layer1)} aggregates")
    print(f"  Layer 2: {len(layer2)} aggregates")
    if aggregates:
        print(f"  Z range: {np.min(Z[Z > 0]):.3f} - {np.max(Z):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
