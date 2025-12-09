"""
Voronoi-based surface construction with controlled height distribution.

Steps:
1. Place points randomly
2. Construct periodic Voronoi tessellation (using padding approach)
3. Split edges > hmin into segments of ~hmin
4. Create triangles from edge points to cell centers
5. Shift edge points by DeltaZ + Gaussian noise
6. Interpolate onto Cartesian grid
7. Apply Gaussian smoothing
8. Superpose small-scale self-affine roughness
9. Remap distribution to target PDF
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
from typing import Callable

from rfgen import selfaffine_field


def build_inverse_cdf_from_pdf(pdf: Callable, x_min: float, x_max: float, n_grid: int = 10_000):
    """
    Construct a numerical inverse CDF F_T^{-1} from a target PDF.
    """
    x = np.linspace(x_min, x_max, n_grid)
    p = np.maximum(pdf(x), 0.0)
    if not np.any(p > 0):
        raise ValueError("PDF is zero on the whole grid; check x_min/x_max.")

    cdf = np.cumsum(p)
    cdf = cdf / cdf[-1]
    cdf[0] = 0.0
    cdf[-1] = 1.0

    def F_T_inv(u):
        u = np.asarray(u)
        u_clipped = np.clip(u, 0.0, 1.0)
        return np.interp(u_clipped, cdf, x, left=x_min, right=x_max)

    return F_T_inv


def remap_distribution(z: np.ndarray, probability_distribution: Callable, x_min: float, x_max: float):
    """
    Remap field z to have target marginal via inverse CDF.
    """
    F_T_inv = build_inverse_cdf_from_pdf(probability_distribution, x_min, x_max)
    z_flat = z.ravel()
    ranks = np.argsort(np.argsort(z_flat))
    u = (ranks + 0.5) / len(z_flat)
    z_new_flat = F_T_inv(u)
    return z_new_flat.reshape(z.shape)


def generate_random_points(n_points: int, Lx: float, Ly: float, rng: np.random.Generator):
    """Generate random points in [0, Lx) x [0, Ly)."""
    points = rng.uniform(0, 1, (n_points, 2))
    points[:, 0] *= Lx
    points[:, 1] *= Ly
    return points


def add_periodic_padding(points: np.ndarray, Lx: float, Ly: float, margin: float):
    """
    Add periodic copies of points in margin regions around the domain.
    
    Creates copies in all 8 neighboring periodic cells, but only keeps
    points that are within 'margin' distance from the central domain.
    
    Returns:
        extended_points: all points including periodic copies in margin
        central_mask: boolean mask indicating which points are in the central domain
    """
    all_points = []
    central_mask = []
    
    # Add points from all 9 cells (center + 8 neighbors)
    for dx in [-Lx, 0, Lx]:
        for dy in [-Ly, 0, Ly]:
            shifted = points + np.array([dx, dy])
            
            for p in shifted:
                # Check if point is in the extended domain [-margin, Lx+margin] x [-margin, Ly+margin]
                if (-margin <= p[0] <= Lx + margin) and (-margin <= p[1] <= Ly + margin):
                    all_points.append(p)
                    # Mark if this point is in the central domain
                    is_central = (0 <= p[0] < Lx) and (0 <= p[1] < Ly) and (dx == 0) and (dy == 0)
                    central_mask.append(is_central)
    
    return np.array(all_points), np.array(central_mask)


def subdivide_edge(p1: np.ndarray, p2: np.ndarray, hmin: float):
    """
    Subdivide edge from p1 to p2 into segments of length ~hmin.
    Returns list of points along the edge (including endpoints).
    """
    length = np.linalg.norm(p2 - p1)
    if length <= hmin:
        return [p1, p2]
    
    n_segments = max(1, int(np.round(length / hmin)))
    t = np.linspace(0, 1, n_segments + 1)
    edge_points = [(1 - ti) * p1 + ti * p2 for ti in t]
    return edge_points


def compute_polygon_centroid(vertices: list):
    """
    Compute the centroid (barycenter) of a polygon given its vertices.
    Uses the formula for centroid of a polygon.
    """
    vertices = np.array(vertices)
    n = len(vertices)
    
    # Simple average of vertices (works well for convex polygons like Voronoi cells)
    centroid = np.mean(vertices, axis=0)
    return centroid


def cell_overlaps_domain(region_vertices: list, Lx: float, Ly: float):
    """
    Check if a Voronoi cell overlaps with the central domain [0, Lx] x [0, Ly].
    
    Returns True if any part of the cell could be inside the domain.
    """
    vertices = np.array(region_vertices)
    
    # Check if any vertex is inside or near the domain
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    
    # Cell overlaps if its bounding box overlaps with [0, Lx] x [0, Ly]
    overlaps_x = (x_max >= 0) and (x_min <= Lx)
    overlaps_y = (y_max >= 0) and (y_min <= Ly)
    
    return overlaps_x and overlaps_y


def build_voronoi_mesh(points: np.ndarray, hmin: float, gap: float, Lx: float, Ly: float, margin: float):
    """
    Build triangular mesh from Voronoi tessellation with periodic padding.
    
    Uses barycenters (centroids) of Voronoi cells instead of seed points.
    Includes all cells that overlap with the central domain [0, Lx] x [0, Ly].
    
    Creates a flat interior region and a rim near the edges:
    - Barycenter and intermediate points (at gap/2 from edge) are flat
    - Edge points are perturbed
    
    Parameters:
        points: seed points in [0, Lx) x [0, Ly)
        hmin: minimum edge segment length
        gap: width of the rim region (intermediate points at gap/2 from edge)
        Lx, Ly: domain size
        margin: padding margin (should be ~2x typical grain size)
    
    Returns:
        vertices: (N, 2) array of vertex positions
        triangles: (M, 3) array of triangle vertex indices
        vertex_types: array indicating vertex type:
            0 = barycenter (flat)
            1 = intermediate (flat, at gap/2 from edge)
            2 = edge (perturbed)
    """
    # Add periodic padding
    extended_points, central_mask = add_periodic_padding(points, Lx, Ly, margin)
    
    # Compute Voronoi on extended domain
    vor = Voronoi(extended_points)
    
    vertices = []
    triangles = []
    vertex_types = []
    vertex_map = {}  # (x, y, type) -> index
    
    def add_vertex(x, y, vtype):
        """Add vertex and return its index. vtype: 0=center, 1=intermediate, 2=edge"""
        key = (round(x, 8), round(y, 8), vtype)
        if key not in vertex_map:
            idx = len(vertices)
            vertex_map[key] = idx
            vertices.append([x, y])
            vertex_types.append(vtype)
        return vertex_map[key]
    
    # Process ALL cells that overlap with the central domain
    for point_idx in range(len(extended_points)):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        
        # Skip invalid regions
        if -1 in region or len(region) < 3:
            continue
        
        # Get vertices of this region
        region_vertices = [vor.vertices[v_idx] for v_idx in region]
        
        # Check if this cell overlaps with the central domain
        if not cell_overlaps_domain(region_vertices, Lx, Ly):
            continue
        
        # Use barycenter (centroid) instead of seed point
        centroid = compute_polygon_centroid(region_vertices)
        center_idx = add_vertex(centroid[0], centroid[1], vtype=0)
        
        # Collect all edge points and their corresponding intermediate points
        all_edge_points = []
        all_intermediate_points = []
        
        # For each edge of the cell, subdivide
        n_verts = len(region_vertices)
        for i in range(n_verts):
            p1 = np.array(region_vertices[i])
            p2 = np.array(region_vertices[(i + 1) % n_verts])
            
            # Subdivide edge
            edge_points = subdivide_edge(p1, p2, hmin)
            
            # For each edge point, compute intermediate point at gap/2 from edge toward centroid
            for ep in edge_points:
                ep = np.array(ep)
                # Direction from edge point toward centroid
                direction = centroid - ep
                dist = np.linalg.norm(direction)
                if dist > 1e-10:
                    direction = direction / dist
                    # Intermediate point at gap/2 from edge point toward centroid
                    intermediate = ep + direction * (gap / 2)
                else:
                    intermediate = ep
                
                all_edge_points.append(ep)
                all_intermediate_points.append(intermediate)
        
        # Remove duplicate points at polygon vertices (where edges meet)
        # We keep unique points by checking proximity
        unique_edge_points = []
        unique_intermediate_points = []
        
        for i, (ep, ip) in enumerate(zip(all_edge_points, all_intermediate_points)):
            # Check if this edge point is too close to the previous one
            if i == 0 or np.linalg.norm(ep - unique_edge_points[-1]) > 1e-8:
                unique_edge_points.append(ep)
                unique_intermediate_points.append(ip)
        
        # Close the loop - check if last point is same as first
        if len(unique_edge_points) > 1 and np.linalg.norm(unique_edge_points[-1] - unique_edge_points[0]) < 1e-8:
            unique_edge_points = unique_edge_points[:-1]
            unique_intermediate_points = unique_intermediate_points[:-1]
        
        n_pts = len(unique_edge_points)
        if n_pts < 3:
            continue
        
        # Add vertices and create triangles
        edge_indices = [add_vertex(p[0], p[1], vtype=2) for p in unique_edge_points]
        intermediate_indices = [add_vertex(p[0], p[1], vtype=1) for p in unique_intermediate_points]
        
        # Create inner triangles: [barycenter, intermediate_i, intermediate_i+1]
        for i in range(n_pts):
            i_next = (i + 1) % n_pts
            triangles.append([center_idx, intermediate_indices[i], intermediate_indices[i_next]])
        
        # Create outer rim triangles: quad [intermediate_i, edge_i, edge_i+1, intermediate_i+1]
        # Split into two triangles
        for i in range(n_pts):
            i_next = (i + 1) % n_pts
            # Triangle 1: [intermediate_i, edge_i, edge_i+1]
            triangles.append([intermediate_indices[i], edge_indices[i], edge_indices[i_next]])
            # Triangle 2: [intermediate_i, edge_i+1, intermediate_i+1]
            triangles.append([intermediate_indices[i], edge_indices[i_next], intermediate_indices[i_next]])
    
    return np.array(vertices), np.array(triangles), np.array(vertex_types)


def apply_height_displacement(vertices: np.ndarray, vertex_types: np.ndarray, 
                               delta_z: float, sigma: float, rng: np.random.Generator):
    """
    Apply height displacement to vertices.
    
    - Cell centers (type 0): z = 0 (flat)
    - Intermediate points (type 1): z = 0 (flat)
    - Edge points (type 2): z = delta_z + Gaussian noise with std = sigma
    
    Returns z values for each vertex.
    """
    n_verts = len(vertices)
    z = np.zeros(n_verts)
    
    # Only edge points (type 2) get height displacement
    edge_mask = vertex_types == 2
    n_edges = np.sum(edge_mask)
    
    z[edge_mask] = delta_z + rng.normal(0, sigma, n_edges)
    
    return z


def interpolate_to_grid(vertices: np.ndarray, triangles: np.ndarray, z: np.ndarray,
                        Nx: int, Ny: int, Lx: float, Ly: float):
    """
    Interpolate triangular mesh onto regular Cartesian grid.
    
    The mesh may extend beyond [0, Lx] x [0, Ly] due to padding,
    but we only interpolate onto the central domain.
    """
    # Create grid for central domain only
    x_grid = np.linspace(0, Lx, Nx, endpoint=False)
    y_grid = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create triangulation and interpolator
    tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    interpolator = LinearTriInterpolator(tri, z)
    
    # Interpolate
    Z = interpolator(X, Y)
    
    return X, Y, Z


def estimate_grain_size(n_points: int, Lx: float, Ly: float):
    """Estimate typical grain size from number of points and domain size."""
    area_per_point = (Lx * Ly) / n_points
    # Approximate grain as circle with this area
    typical_size = 2 * np.sqrt(area_per_point / np.pi)
    return typical_size


def plot_voronoi_with_padding(points: np.ndarray, Lx: float, Ly: float, margin: float, ax=None):
    """Plot Voronoi cells showing the padding region and barycenters."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get extended points
    extended_points, central_mask = add_periodic_padding(points, Lx, Ly, margin)
    
    # Compute Voronoi
    vor = Voronoi(extended_points)
    
    # Plot extended domain boundary
    ax.axhline(0, color='k', linewidth=2, linestyle='--')
    ax.axhline(Ly, color='k', linewidth=2, linestyle='--')
    ax.axvline(0, color='k', linewidth=2, linestyle='--')
    ax.axvline(Lx, color='k', linewidth=2, linestyle='--')
    
    # Plot Voronoi edges
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue
        v1 = vor.vertices[ridge_vertices[0]]
        v2 = vor.vertices[ridge_vertices[1]]
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'b-', linewidth=0.5)
    
    # Compute and plot barycenters for cells that overlap with domain
    barycenters = []
    for point_idx in range(len(extended_points)):
        region_idx = vor.point_region[point_idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        region_vertices = [vor.vertices[v_idx] for v_idx in region]
        if cell_overlaps_domain(region_vertices, Lx, Ly):
            centroid = compute_polygon_centroid(region_vertices)
            barycenters.append(centroid)
    
    barycenters = np.array(barycenters)
    
    # Plot seed points (faded)
    ax.scatter(extended_points[central_mask, 0], extended_points[central_mask, 1], 
               c='gray', s=20, zorder=3, alpha=0.3, marker='x', label='Seed points')
    
    # Plot barycenters
    if len(barycenters) > 0:
        ax.scatter(barycenters[:, 0], barycenters[:, 1], 
                   c='red', s=50, zorder=5, label='Barycenters')
    
    ax.set_xlim(-margin, Lx + margin)
    ax.set_ylim(-margin, Ly + margin)
    ax.set_aspect('equal')
    ax.set_title('Voronoi Tessellation with Barycenters')
    ax.legend()
    
    return ax


def plot_mesh(vertices: np.ndarray, triangles: np.ndarray, z: np.ndarray, 
              Lx: float, Ly: float, ax=None, show_domain_boundary=True):
    """Plot the triangular mesh with heights."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    
    # Plot triangles colored by mean z
    z_tri = np.mean(z[triangles], axis=1)
    tcf = ax.tripcolor(tri, z_tri, cmap='RdYlBu_r', shading='flat')
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
    
    if show_domain_boundary:
        ax.axhline(0, color='k', linewidth=2, linestyle='--')
        ax.axhline(Ly, color='k', linewidth=2, linestyle='--')
        ax.axvline(0, color='k', linewidth=2, linestyle='--')
        ax.axvline(Lx, color='k', linewidth=2, linestyle='--')
    
    ax.set_aspect('equal')
    ax.set_title('Triangular Mesh with Heights')
    plt.colorbar(tcf, ax=ax, label='Height')
    
    return ax


def main():
    # ===== Parameters =====
    Lx, Ly = 10.0, 10.0          # Domain size
    n_points = 50                 # Number of Voronoi cells
    hmin = 0.3                    # Minimum edge segment length
    delta_z = 5.0                # Base height displacement for edges (negative = grooves)
    sigma = 0.4                   # Gaussian noise std for edge heights
    Nx, Ny = 512, 512             # Grid resolution for interpolation
    smooth_sigma = 3.0            # Gaussian smoothing sigma (in pixels)
    seed = 42
    
    # Small-scale roughness parameters (self-affine)
    Hurst_roughness = 0.8         # Hurst exponent for small-scale roughness
    rms_roughness = 0.1           # RMS height of small-scale roughness
    k_low_roughness = 0.05        # Lower wavenumber bound (larger scale features)
    k_high_roughness = 0.5        # Upper wavenumber bound (Nyquist limit)
    
    # Target PDF parameters (Weibull distribution)
    def target_pdf(x):
        k, lambd = 2.0, 1.5
        return np.where(x > 0, 
                        k / lambd * (x / lambd)**(k - 1) * np.exp(-(x / lambd)**k),
                        0.0)
    height_bounds = (0.01, 8.0)
    
    rng = np.random.default_rng(seed)
    
    # Estimate grain size and set margin/gap
    grain_size = estimate_grain_size(n_points, Lx, Ly)
    margin = 2.0 * grain_size
    gap = grain_size / 4.0        # Width of rim region (intermediate points at gap/2 from edge)
    print(f"Estimated grain size: {grain_size:.2f}")
    print(f"Using margin: {margin:.2f}")
    print(f"Using gap: {gap:.2f}")
    
    # ===== Step 1: Generate random points =====
    print("\nStep 1: Generating random points...")
    points = generate_random_points(n_points, Lx, Ly, rng)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=50)
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_title('Step 1: Random Points')
    plt.tight_layout()
    plt.show()
    
    # ===== Step 2: Construct periodic Voronoi tessellation =====
    print("Step 2: Constructing periodic Voronoi tessellation...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_voronoi_with_padding(points, Lx, Ly, margin, ax)
    plt.tight_layout()
    plt.show()
    
    # ===== Steps 3-4: Build mesh with subdivided edges and triangles =====
    print("Steps 3-4: Building triangular mesh with flat interior and rim...")
    vertices, triangles, vertex_types = build_voronoi_mesh(
        points, hmin, gap, Lx, Ly, margin
    )
    print(f"  Mesh has {len(vertices)} vertices and {len(triangles)} triangles")
    print(f"  - Barycenters (flat): {np.sum(vertex_types == 0)}")
    print(f"  - Intermediate (flat): {np.sum(vertex_types == 1)}")
    print(f"  - Edge points (perturbed): {np.sum(vertex_types == 2)}")
    
    # Plot mesh (flat)
    fig, ax = plt.subplots(figsize=(10, 10))
    tri = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.triplot(tri, 'b-', linewidth=0.3)
    ax.scatter(vertices[vertex_types == 0, 0], vertices[vertex_types == 0, 1], 
               c='red', s=30, zorder=5, label='Barycenters (flat)')
    ax.scatter(vertices[vertex_types == 1, 0], vertices[vertex_types == 1, 1], 
               c='orange', s=10, zorder=4, label='Intermediate (flat)')
    ax.scatter(vertices[vertex_types == 2, 0], vertices[vertex_types == 2, 1], 
               c='green', s=5, zorder=3, label='Edge points (perturbed)')
    
    # Show domain boundary
    ax.axhline(0, color='k', linewidth=2, linestyle='--')
    ax.axhline(Ly, color='k', linewidth=2, linestyle='--')
    ax.axvline(0, color='k', linewidth=2, linestyle='--')
    ax.axvline(Lx, color='k', linewidth=2, linestyle='--')
    
    ax.set_aspect('equal')
    ax.set_title('Steps 3-4: Triangular Mesh (with padding)')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # ===== Step 5: Apply height displacement =====
    print("Step 5: Applying height displacement...")
    z = apply_height_displacement(vertices, vertex_types, delta_z, sigma, rng)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_mesh(vertices, triangles, z, Lx, Ly, ax)
    plt.tight_layout()
    plt.show()
    
    # ===== Step 6: Interpolate to Cartesian grid =====
    print("Step 6: Interpolating to Cartesian grid...")
    X, Y, Z = interpolate_to_grid(vertices, triangles, z, Nx, Ny, Lx, Ly)
    
    # Handle any masked values from interpolation
    Z_filled = np.ma.filled(Z, np.nanmean(Z))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z_filled, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title('Step 6: Interpolated Surface (before smoothing)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    plt.tight_layout()
    plt.show()
    
    # ===== Step 7: Apply Gaussian smoothing =====
    print(f"Step 7: Applying Gaussian smoothing (sigma={smooth_sigma} pixels)...")
    Z_smooth = gaussian_filter(Z_filled, sigma=smooth_sigma, mode='wrap')  # 'wrap' for periodic BC
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(Z_smooth, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title('Step 7: Smoothed Surface')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    plt.tight_layout()
    plt.show()
    
    # ===== Step 8: Superpose small-scale self-affine roughness =====
    print(f"Step 8: Adding small-scale roughness (H={Hurst_roughness}, rms={rms_roughness})...")
    
    # Generate self-affine roughness field
    # Use square grid size (max of Nx, Ny) for selfaffine_field
    N_roughness = max(Nx, Ny)
    field_roughness = selfaffine_field(
        dim=2, N=N_roughness, Hurst=Hurst_roughness,
        k_low=k_low_roughness, k_high=k_high_roughness, rng=rng
    )
    
    # Crop or pad to match Nx x Ny if needed
    field_roughness = field_roughness[:Ny, :Nx]
    
    # Normalize to desired RMS roughness
    field_roughness = field_roughness * rms_roughness / np.std(field_roughness)
    
    # Superpose roughness onto smoothed surface
    Z_with_roughness = Z_smooth + field_roughness
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(field_roughness, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title(f'Small-scale Roughness (H={Hurst_roughness}, rms={rms_roughness:.3f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    ax = axes[1]
    im = ax.imshow(Z_with_roughness, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title('Step 8: Surface with Small-scale Roughness')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    plt.tight_layout()
    plt.show()
    
    # ===== Step 9: Remap distribution to target PDF =====
    print("Step 9: Remapping distribution to target PDF...")
    
    Z_remapped = remap_distribution(Z_with_roughness, target_pdf, height_bounds[0], height_bounds[1])
    
    # Plot PDF comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before remapping
    ax = axes[0]
    ax.hist(Z_with_roughness.ravel(), bins=100, density=True, alpha=0.7, label='Before remapping')
    ax.set_title('Height Distribution Before Remapping')
    ax.set_xlabel('Height')
    ax.set_ylabel('Density')
    ax.legend()
    
    # After remapping
    ax = axes[1]
    ax.hist(Z_remapped.ravel(), bins=100, density=True, alpha=0.7, label='After remapping')
    z_plot = np.linspace(height_bounds[0], height_bounds[1], 200)
    ax.plot(z_plot, target_pdf(z_plot), 'r-', linewidth=2, label='Target PDF')
    ax.set_title('Height Distribution After Remapping')
    ax.set_xlabel('Height')
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # ===== Final result =====
    print("Plotting final surface...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    im = ax.imshow(-Z_with_roughness, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title('With Roughness (Before Remapping)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    ax = axes[1]
    im = ax.imshow(-Z_remapped, extent=[0, Lx, 0, Ly], origin='lower', cmap='RdYlBu_r')
    ax.set_title('Final (After Remapping)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Height')
    
    plt.tight_layout()
    plt.show()
    
    # 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, -Z_with_roughness, cmap='RdYlBu_r', linewidth=0, antialiased=True)
    ax.set_title('With Roughness')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(X, Y, -Z_remapped, cmap='RdYlBu_r', linewidth=0, antialiased=True)
    ax.set_title('Final (Remapped)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    
    plt.tight_layout()
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()
