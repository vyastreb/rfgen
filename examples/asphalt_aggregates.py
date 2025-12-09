"""
Asphalt concrete surface model using 3D aggregates.

Steps:
1. Generate irregular stone-like aggregates (deformed ellipsoids with 1:2:2 axes)
2. Drop and settle aggregates with gravity (3-point contact)
3. Periodic boundaries for tileable result
4. Render the height map
5. Save results to npz

Accelerated with Numba for performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from numba import jit, prange
import os


# ============================================================================
# Numba-accelerated functions
# ============================================================================

@jit(nopython=True, cache=True)
def compute_min_dist_to_vertices(new_vertices, existing_vertices_flat, n_existing, n_verts_per_agg):
    """
    Compute minimum distance from any new vertex to any existing vertex.
    existing_vertices_flat: flattened array of all existing vertices
    """
    min_dist = np.inf
    n_new = len(new_vertices)
    
    for i in range(n_new):
        vx, vy, vz = new_vertices[i, 0], new_vertices[i, 1], new_vertices[i, 2]
        
        for j in range(n_existing):
            start_idx = j * n_verts_per_agg
            for k in range(n_verts_per_agg):
                idx = start_idx + k
                if idx >= len(existing_vertices_flat) // 3:
                    break
                ex = existing_vertices_flat[idx * 3]
                ey = existing_vertices_flat[idx * 3 + 1]
                ez = existing_vertices_flat[idx * 3 + 2]
                
                d = np.sqrt((vx - ex)**2 + (vy - ey)**2 + (vz - ez)**2)
                if d < min_dist:
                    min_dist = d
    
    return min_dist


@jit(nopython=True, cache=True)
def compute_min_dist_simple(new_vertices, all_existing_vertices):
    """
    Simple minimum distance computation between vertex sets.
    """
    min_dist = np.inf
    n_new = len(new_vertices)
    n_existing = len(all_existing_vertices)
    
    for i in range(n_new):
        for j in range(n_existing):
            dx = new_vertices[i, 0] - all_existing_vertices[j, 0]
            dy = new_vertices[i, 1] - all_existing_vertices[j, 1]
            dz = new_vertices[i, 2] - all_existing_vertices[j, 2]
            d = np.sqrt(dx*dx + dy*dy + dz*dz)
            if d < min_dist:
                min_dist = d
    
    return min_dist


@jit(nopython=True, cache=True)
def interpolate_z_triangle_numba(px, py, v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z):
    """Interpolate z value at point (px, py) if inside triangle."""
    # Barycentric coordinates
    denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
    if abs(denom) < 1e-10:
        return -1e10, False
    
    w0 = ((v1y - v2y) * (px - v2x) + (v2x - v1x) * (py - v2y)) / denom
    w1 = ((v2y - v0y) * (px - v2x) + (v0x - v2x) * (py - v2y)) / denom
    w2 = 1.0 - w0 - w1
    
    if w0 >= -0.001 and w1 >= -0.001 and w2 >= -0.001:
        z = w0 * v0z + w1 * v1z + w2 * v2z
        return z, True
    
    return -1e10, False


@jit(nopython=True, cache=True)
def render_height_map_faces_numba(all_vertices, all_faces, Lx, Ly, Nx, Ny):
    """
    Render height map by projecting triangle faces.
    Properly fills all pixels covered by triangles.
    """
    Z = np.zeros((Ny, Nx))
    dx = Lx / Nx
    dy = Ly / Ny
    
    n_faces = len(all_faces)
    
    for f_idx in range(n_faces):
        i0 = all_faces[f_idx, 0]
        i1 = all_faces[f_idx, 1]
        i2 = all_faces[f_idx, 2]
        
        v0x, v0y, v0z = all_vertices[i0, 0], all_vertices[i0, 1], all_vertices[i0, 2]
        v1x, v1y, v1z = all_vertices[i1, 0], all_vertices[i1, 1], all_vertices[i1, 2]
        v2x, v2y, v2z = all_vertices[i2, 0], all_vertices[i2, 1], all_vertices[i2, 2]
        
        # Bounding box of triangle in grid coordinates
        min_x = min(v0x, v1x, v2x)
        max_x = max(v0x, v1x, v2x)
        min_y = min(v0y, v1y, v2y)
        max_y = max(v0y, v1y, v2y)
        
        ix_min = int(min_x / dx) - 1
        ix_max = int(max_x / dx) + 2
        iy_min = int(min_y / dy) - 1
        iy_max = int(max_y / dy) + 2
        
        for iy in range(iy_min, iy_max):
            for ix in range(ix_min, ix_max):
                # Periodic indices
                niy = iy % Ny
                nix = ix % Nx
                
                # Grid point coordinates
                px = (nix + 0.5) * dx
                py = (niy + 0.5) * dy
                
                # Also check with shifts for periodicity
                for shift_x in [0.0, Lx, -Lx]:
                    for shift_y in [0.0, Ly, -Ly]:
                        z_val, inside = interpolate_z_triangle_numba(
                            px + shift_x, py + shift_y, 
                            v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z
                        )
                        if inside and z_val > Z[niy, nix]:
                            Z[niy, nix] = z_val
    
    return Z


# ============================================================================
# Non-numba functions (require Python objects)
# ============================================================================

def create_irregular_aggregate(center: np.ndarray, base_size: float, 
                                axis_ratio: tuple, n_control_points: int, 
                                irregularity: float, rng: np.random.Generator):
    """
    Create an irregular stone-like aggregate by placing random points on an 
    ellipsoid and taking their convex hull (preserves polygonal shape).
    """
    # Normalize axis ratio and scale
    axis_ratio = np.array(axis_ratio, dtype=float)
    axes = base_size * axis_ratio / np.mean(axis_ratio)
    
    # Random orientation (rotation matrix)
    q = rng.normal(0, 1, 4)
    q = q / np.linalg.norm(q)
    
    w, x, y, z = q
    orientation = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    # Generate control points on ellipsoid surface
    phi = rng.uniform(0, 2 * np.pi, n_control_points)
    cos_theta = rng.uniform(-1, 1, n_control_points)
    theta = np.arccos(cos_theta)
    
    x_pts = axes[0] * np.sin(theta) * np.cos(phi)
    y_pts = axes[1] * np.sin(theta) * np.sin(phi)
    z_pts = axes[2] * np.cos(theta)
    
    control_points = np.column_stack([x_pts, y_pts, z_pts])
    
    # Add irregularity
    for i in range(n_control_points):
        direction = control_points[i] / (np.linalg.norm(control_points[i]) + 1e-10)
        perturbation = irregularity * base_size * rng.uniform(-0.3, 0.3)
        control_points[i] += direction * perturbation
    
    # Apply rotation
    control_points = control_points @ orientation.T
    
    # Translate to center
    control_points += center
    
    # Create convex hull
    try:
        hull = ConvexHull(control_points)
        vertices = control_points
        faces = hull.simplices
    except Exception:
        vertices = control_points
        faces = None
        hull = None
    
    return vertices, faces, hull, axes, orientation


def get_periodic_copies(vertices: np.ndarray, center: np.ndarray, Lx: float, Ly: float):
    """
    Get all periodic copies of an aggregate that might affect the domain [0, Lx] x [0, Ly].
    """
    copies = []
    
    for dx in [-Lx, 0, Lx]:
        for dy in [-Ly, 0, Ly]:
            shifted_verts = vertices.copy()
            shifted_verts[:, 0] += dx
            shifted_verts[:, 1] += dy
            shifted_center = center.copy()
            shifted_center[0] += dx
            shifted_center[1] += dy
            copies.append((shifted_verts, shifted_center))
    
    return copies


def find_min_distance_to_ground(vertices: np.ndarray):
    """Find minimum z coordinate."""
    return np.min(vertices[:, 2])


def find_min_distance_to_aggregates_periodic(vertices: np.ndarray, existing_aggregates: list, 
                                              Lx: float, Ly: float):
    """
    Find minimum distance considering periodic boundaries.
    """
    if not existing_aggregates:
        return np.inf
    
    # Collect all existing vertices (including periodic copies)
    all_existing = []
    for agg in existing_aggregates:
        # Add main copy and periodic copies
        for dx in [-Lx, 0, Lx]:
            for dy in [-Ly, 0, Ly]:
                shifted = agg['vertices'].copy()
                shifted[:, 0] += dx
                shifted[:, 1] += dy
                all_existing.append(shifted)
    
    all_existing = np.vstack(all_existing)
    
    # Use numba-accelerated distance computation
    min_dist = compute_min_dist_simple(vertices, all_existing)
    
    return min_dist


def find_3_lowest_vertices(vertices: np.ndarray):
    """Find indices of 3 lowest vertices forming a stable base."""
    from itertools import combinations
    
    z_sorted_idx = np.argsort(vertices[:, 2])
    lowest_indices = z_sorted_idx[:min(6, len(z_sorted_idx))]
    
    best_triangle = None
    best_area = 0
    
    for idx_combo in combinations(lowest_indices, 3):
        p0, p1, p2 = vertices[idx_combo[0]], vertices[idx_combo[1]], vertices[idx_combo[2]]
        v1 = p1[:2] - p0[:2]
        v2 = p2[:2] - p0[:2]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
        
        if area > best_area:
            best_area = area
            best_triangle = idx_combo
    
    if best_triangle is None:
        best_triangle = tuple(z_sorted_idx[:3])
    
    return best_triangle


def compute_rotation_for_3_point_contact(vertices: np.ndarray, center: np.ndarray, 
                                          contact_indices: tuple, target_z: float):
    """Compute rotation to make 3 contact points at same z level."""
    p0, p1, p2 = vertices[contact_indices[0]], vertices[contact_indices[1]], vertices[contact_indices[2]]
    
    z0, z1, z2 = p0[2], p1[2], p2[2]
    
    if max(z0, z1, z2) - min(z0, z1, z2) < 0.01:
        return np.eye(3)
    
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-10:
        return np.eye(3)
    normal = normal / norm_len
    
    target_normal = np.array([0, 0, 1]) if normal[2] > 0 else np.array([0, 0, -1])
    
    axis = np.cross(normal, target_normal)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:
        return np.eye(3)
    
    axis = axis / axis_norm
    cos_angle = np.clip(np.dot(normal, target_normal), -1, 1)
    angle = np.arccos(cos_angle)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return R


def rotate_vertices(vertices: np.ndarray, center: np.ndarray, rotation_matrix: np.ndarray):
    """Rotate vertices around center."""
    centered = vertices - center
    rotated = centered @ rotation_matrix.T
    return rotated + center


def drop_and_settle_aggregate(vertices: np.ndarray, center: np.ndarray, 
                               existing_aggregates: list, gap_min: float,
                               Lx: float, Ly: float, max_iterations: int = 50):
    """
    Drop aggregate and settle with 3-point contact.
    Considers periodic boundaries.
    """
    current_vertices = vertices.copy()
    current_center = center.copy()
    
    # Step 1: Drop until contact
    for iteration in range(max_iterations):
        min_z = find_min_distance_to_ground(current_vertices)
        min_dist_agg = find_min_distance_to_aggregates_periodic(
            current_vertices, existing_aggregates, Lx, Ly
        )
        
        drop_to_ground = min_z - gap_min
        drop_to_agg = (min_dist_agg - gap_min) * 0.5 if min_dist_agg < np.inf else np.inf
        
        drop_amount = min(drop_to_ground, drop_to_agg)
        
        if drop_amount <= 0.001:
            break
        
        current_vertices[:, 2] -= drop_amount
        current_center[2] -= drop_amount
    
    # Step 2: Find stable base
    contact_indices = find_3_lowest_vertices(current_vertices)
    
    # Step 3: Rotate for 3-point contact
    contact_z = np.mean([current_vertices[i, 2] for i in contact_indices])
    R = compute_rotation_for_3_point_contact(current_vertices, current_center, 
                                              contact_indices, contact_z)
    current_vertices = rotate_vertices(current_vertices, current_center, R)
    
    # Step 4: Final adjustment
    min_z = find_min_distance_to_ground(current_vertices)
    min_dist_agg = find_min_distance_to_aggregates_periodic(
        current_vertices, existing_aggregates, Lx, Ly
    )
    
    final_drop = min(min_z - gap_min, (min_dist_agg - gap_min) * 0.5 if min_dist_agg < np.inf else np.inf)
    if final_drop > 0:
        current_vertices[:, 2] -= final_drop
        current_center[2] -= final_drop
    
    return current_vertices, current_center


def wrap_to_domain(vertices: np.ndarray, center: np.ndarray, Lx: float, Ly: float):
    """
    Wrap aggregate center to [0, Lx) x [0, Ly) domain.
    """
    # Wrap center
    wrapped_center = center.copy()
    dx = 0
    dy = 0
    
    while wrapped_center[0] < 0:
        wrapped_center[0] += Lx
        dx += Lx
    while wrapped_center[0] >= Lx:
        wrapped_center[0] -= Lx
        dx -= Lx
    while wrapped_center[1] < 0:
        wrapped_center[1] += Ly
        dy += Ly
    while wrapped_center[1] >= Ly:
        wrapped_center[1] -= Ly
        dy -= Ly
    
    # Apply same shift to vertices
    wrapped_vertices = vertices.copy()
    wrapped_vertices[:, 0] += dx
    wrapped_vertices[:, 1] += dy
    
    return wrapped_vertices, wrapped_center


def pack_aggregates(n_aggregates: int, Lx: float, Ly: float, 
                    aggregate_size: float, aggregate_rms: float,
                    axis_ratio: tuple, gap_min: float,
                    n_control_points: int, irregularity: float,
                    rng: np.random.Generator, max_attempts: int = 100):
    """
    Pack aggregates with gravity settling and periodic boundaries.
    """
    aggregates = []
    
    drop_height = 10 * aggregate_size * max(axis_ratio)
    
    for i in range(n_aggregates):
        size = max(0.3 * aggregate_size, rng.normal(aggregate_size, aggregate_rms))
        bounding_radius = size * max(axis_ratio) / np.mean(axis_ratio)
        
        placed = False
        for attempt in range(max_attempts):
            # Random (x, y) - can be anywhere, will be wrapped
            x = rng.uniform(0, Lx)
            y = rng.uniform(0, Ly)
            z = drop_height
            
            initial_center = np.array([x, y, z])
            
            vertices, faces, hull, axes, orientation = create_irregular_aggregate(
                initial_center, size, axis_ratio, n_control_points, irregularity, rng
            )
            
            if faces is None:
                continue
            
            # Drop and settle (periodic)
            settled_vertices, settled_center = drop_and_settle_aggregate(
                vertices, initial_center, aggregates, gap_min, Lx, Ly
            )
            
            # Wrap to domain
            wrapped_vertices, wrapped_center = wrap_to_domain(
                settled_vertices, settled_center, Lx, Ly
            )
            
            min_z = np.min(wrapped_vertices[:, 2])
            
            if min_z >= 0:
                # Recompute hull
                try:
                    hull = ConvexHull(wrapped_vertices)
                    faces = hull.simplices
                except:
                    faces = None
                
                aggregates.append({
                    'center': wrapped_center,
                    'size': size,
                    'bounding_radius': bounding_radius,
                    'vertices': wrapped_vertices,
                    'faces': faces,
                    'hull': hull,
                    'axes': axes,
                    'orientation': orientation
                })
                placed = True
                break
        
        if (i + 1) % 100 == 0:
            print(f"  Placed {len(aggregates)}/{n_aggregates} aggregates")
    
    return aggregates


def render_height_map_periodic(aggregates: list, Lx: float, Ly: float, Nx: int, Ny: int):
    """
    Render height map with periodic boundaries using face-based interpolation.
    Uses numba acceleration.
    """
    # Collect all vertices and faces including periodic copies
    all_vertices_list = []
    all_faces_list = []
    vertex_offset = 0
    
    for agg in aggregates:
        vertices = agg['vertices']
        faces = agg['faces']
        
        if faces is None:
            continue
        
        # Add main and periodic copies
        for dx in [-Lx, 0, Lx]:
            for dy in [-Ly, 0, Ly]:
                shifted = vertices.copy()
                shifted[:, 0] += dx
                shifted[:, 1] += dy
                
                # Check if this copy affects the domain (with some margin)
                margin = 0.5
                if (np.max(shifted[:, 0]) >= -margin and np.min(shifted[:, 0]) <= Lx + margin and
                    np.max(shifted[:, 1]) >= -margin and np.min(shifted[:, 1]) <= Ly + margin):
                    
                    all_vertices_list.append(shifted)
                    shifted_faces = faces.copy() + vertex_offset
                    all_faces_list.append(shifted_faces)
                    vertex_offset += len(vertices)
    
    if not all_vertices_list:
        x_grid = np.linspace(0, Lx, Nx, endpoint=False)
        y_grid = np.linspace(0, Ly, Ny, endpoint=False)
        X, Y = np.meshgrid(x_grid, y_grid)
        return X, Y, np.zeros((Ny, Nx))
    
    all_vertices = np.vstack(all_vertices_list).astype(np.float64)
    all_faces = np.vstack(all_faces_list).astype(np.int64)
    
    print(f"  Rendering height map: {len(all_faces)} faces, {len(all_vertices)} vertices...")
    
    # Use face-based rendering for accurate interpolation
    Z = render_height_map_faces_numba(all_vertices, all_faces, Lx, Ly, Nx, Ny)
    
    x_grid = np.linspace(0, Lx, Nx, endpoint=False)
    y_grid = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    return X, Y, Z


def plot_aggregates_3d(aggregates: list, Lx: float, Ly: float, ax=None, alpha=0.7):
    """Plot aggregates in 3D with periodic copies shown."""
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(aggregates))))
    
    for i, agg in enumerate(aggregates):
        vertices = agg['vertices']
        faces = agg['faces']
        
        if faces is not None:
            color = colors[i % len(colors)]
            poly3d = [[vertices[idx] for idx in face] for face in faces]
            collection = Poly3DCollection(poly3d, alpha=alpha, 
                                          facecolor=color, edgecolor='darkgray', linewidth=0.3)
            ax.add_collection3d(collection)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    
    return ax


def plot_single_aggregate(vertices, faces, center, ax=None):
    """Plot a single aggregate."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    if faces is not None:
        poly3d = [[vertices[idx] for idx in face] for face in faces]
        collection = Poly3DCollection(poly3d, alpha=0.7, 
                                      facecolor='steelblue', edgecolor='darkblue', linewidth=1)
        ax.add_collection3d(collection)
    
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=30, zorder=5)
    ax.scatter([center[0]], [center[1]], [center[2]], c='yellow', s=100, marker='*', zorder=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    return ax


def main():
    # ===== Parameters =====
    Lx, Ly = 10.0, 10.0           # Domain size
    n_aggregates = 1500           # Number of aggregates
    aggregate_size = .4          # Mean aggregate size
    aggregate_rms = 0.15          # Standard deviation of size
    axis_ratio = (1, 2, 2)        # Ellipsoid axis ratio (flat stones)
    gap_min = aggregate_size / 20 # Minimum gap between aggregates
    n_control_points = 10         # Control points for polygonal shape
    irregularity = 0.3            # Shape irregularity
    Nx, Ny = 1024, 1024           # Height map resolution
    seed = 42
    sigma_smooth = aggregate_size / 20.  # Gaussian smoothing (physical units)
    
    output_dir = "."              # Output directory for npz files
    
    rng = np.random.default_rng(seed)
    
    print("=" * 60)
    print("Asphalt Aggregate Model (Periodic, Numba-accelerated)")
    print("=" * 60)
    print(f"Domain: {Lx} x {Ly} (periodic)")
    print(f"Aggregates: {n_aggregates}")
    print(f"Aggregate size: {aggregate_size} ± {aggregate_rms}")
    print(f"Axis ratio: {axis_ratio} (ellipsoid)")
    print(f"Gap min: {gap_min}")
    print(f"Resolution: {Nx} x {Ny}")
    print()
    
    # ===== Step 1: Example aggregate =====
    print("Step 1: Creating example aggregate...")
    
    example_center = np.array([0.0, 0.0, 0.0])
    example_verts, example_faces, _, _, _ = create_irregular_aggregate(
        example_center, aggregate_size, axis_ratio, n_control_points, irregularity, rng
    )
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_single_aggregate(example_verts, example_faces, example_center, ax)
    ax.set_title(f'Single Aggregate (Ellipsoid {axis_ratio})')
    max_range = aggregate_size * max(axis_ratio) / np.mean(axis_ratio) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    plt.tight_layout()
    plt.show()
    
    # ===== Step 2: Size distribution =====
    print("\nStep 2: Size distribution...")
    
    rng_temp = np.random.default_rng(seed + 1)
    sizes = [max(0.3 * aggregate_size, rng_temp.normal(aggregate_size, aggregate_rms)) 
             for _ in range(100)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(aggregate_size, color='red', linestyle='--', label=f'Mean = {aggregate_size}')
    ax.set_xlabel('Aggregate Size')
    ax.set_ylabel('Count')
    ax.set_title('Aggregate Size Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # Reset RNG
    rng = np.random.default_rng(seed)
    
    # ===== Step 3: Pack aggregates =====
    print("\nStep 3: Packing aggregates (gravity settling, periodic)...")
    
    aggregates = pack_aggregates(
        n_aggregates, Lx, Ly, aggregate_size, aggregate_rms,
        axis_ratio, gap_min, n_control_points, irregularity, rng
    )
    
    print(f"  Successfully placed {len(aggregates)} aggregates")
    
    if aggregates:
        max_z = max(np.max(agg['vertices'][:, 2]) for agg in aggregates)
        print(f"  Max height: {max_z:.2f}")
    
    # 3D view
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plot_aggregates_3d(aggregates, Lx, Ly, ax, alpha=0.8)
    ax.set_title(f'Packed Aggregates ({len(aggregates)} stones, periodic)')
    plt.tight_layout()
    plt.show()
    
    # Top-down view - show actual polygon shapes
    fig, ax = plt.subplots(figsize=(10, 10))
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    
    colors = plt.cm.viridis(np.linspace(0, 1, min(20, len(aggregates))))
    patches = []
    patch_colors = []
    
    for i, agg in enumerate(aggregates):
        verts = agg['vertices']
        faces = agg['faces']
        color = colors[i % len(colors)]
        
        if faces is not None:
            # Project faces to XY plane and draw as polygons
            for face in faces:
                tri_verts = verts[face][:, :2]  # XY coordinates only
                poly = Polygon(tri_verts, closed=True)
                patches.append(poly)
                patch_colors.append(color)
        
        # Also draw for periodic copies near boundaries
        for dx in [-Lx, Lx]:
            shifted = verts.copy()
            shifted[:, 0] += dx
            if np.max(shifted[:, 0]) >= 0 and np.min(shifted[:, 0]) <= Lx:
                if faces is not None:
                    for face in faces:
                        tri_verts = shifted[face][:, :2]
                        poly = Polygon(tri_verts, closed=True)
                        patches.append(poly)
                        patch_colors.append(color)
        
        for dy in [-Ly, Ly]:
            shifted = verts.copy()
            shifted[:, 1] += dy
            if np.max(shifted[:, 1]) >= 0 and np.min(shifted[:, 1]) <= Ly:
                if faces is not None:
                    for face in faces:
                        tri_verts = shifted[face][:, :2]
                        poly = Polygon(tri_verts, closed=True)
                        patches.append(poly)
                        patch_colors.append(color)
    
    pc = PatchCollection(patches, alpha=0.7, edgecolor='black', linewidth=0.3)
    pc.set_facecolor(patch_colors)
    ax.add_collection(pc)
    
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_title('Top-Down View (Polygon Projections)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ===== Step 4: Render height map =====
    print("\nStep 4: Rendering height map (periodic)...")
    
    X, Y, Z = render_height_map_periodic(aggregates, Lx, Ly, Nx, Ny)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(Z, extent=[0, Lx, 0, Ly], origin='lower', cmap='terrain')
    ax.set_title('Height Map (Before Smoothing)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    ax = axes[1]
    im = ax.imshow(Z, extent=[0, Lx, 0, Ly], origin='lower', cmap='gray')
    ax.set_title('Height Map (Grayscale)')
    plt.colorbar(im, ax=ax, label='Height')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Step 5: Save before smoothing =====
    print("\nStep 5: Saving height map (before smoothing)...")
    
    npz_path_raw = os.path.join(output_dir, "asphalt_heightmap_raw.npz")
    np.savez(npz_path_raw, X=X, Y=Y, Z=Z, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    print(f"  Saved: {npz_path_raw}")
    
    # ===== Step 6: Apply smoothing =====
    print("\nStep 6: Applying Gaussian smoothing...")
    
    # Convert sigma from physical units to pixels
    # sigma_smooth is in physical units, convert to pixels: sigma_pixels = sigma_physical * (Nx / Lx)
    sigma_pixels = sigma_smooth * Nx / Lx
    print(f"  Sigma: {sigma_smooth:.4f} (physical) = {sigma_pixels:.1f} pixels")
    
    Z_smooth = gaussian_filter(Z, sigma=sigma_pixels, mode='wrap')  # periodic BC
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(Z, extent=[0, Lx, 0, Ly], origin='lower', cmap='terrain')
    ax.set_title('Before Smoothing')
    plt.colorbar(im, ax=ax, label='Height')
    
    ax = axes[1]
    im = ax.imshow(Z_smooth, extent=[0, Lx, 0, Ly], origin='lower', cmap='terrain')
    ax.set_title('After Smoothing')
    plt.colorbar(im, ax=ax, label='Height')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Step 7: Save after smoothing =====
    print("\nStep 7: Saving height map (after smoothing)...")
    
    npz_path_smooth = os.path.join(output_dir, "asphalt_heightmap_smooth.npz")
    np.savez(npz_path_smooth, X=X, Y=Y, Z=Z_smooth, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
             sigma_smooth=sigma_smooth)
    print(f"  Saved: {npz_path_smooth}")
    
    # ===== Final 3D view =====
    print("\nFinal 3D surface plot...")
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    stride = max(1, Nx // 100)
    ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], Z_smooth[::stride, ::stride], 
                    cmap='terrain', linewidth=0, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title('Final Asphalt Surface (Periodic)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Output files:")
    print(f"  - {npz_path_raw}")
    print(f"  - {npz_path_smooth}")
    print("=" * 60)


if __name__ == "__main__":
    main()
