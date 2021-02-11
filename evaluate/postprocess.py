import copy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class SurfaceExtractor:
    def __init__():
        pass

    def local_max_pool(self, mask, low_threshold, axis=1):
        mcrop = mask.copy()
        mcrop[mcrop < low_threshold] = 0
        agr_sl = np.zeros_like(mcrop)
        ep = argrelmax(mcrop, np.greater, axis=axis)
        agr_sl[ep] = mcrop[ep]
        return agr_sl

    def remove_small_components(self):
        pass

    
    def estimate_pcd_normals(self, pcd, radius=10, max_nn=10, knn=50)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))
        pcd.orient_normals_consistent_tangent_plane(knn)

        return pcd
    
    def mask_to_pcd(self, mask, threshold=0):
        m = (agr_sl > threshold).astype(np.float32)
        points = np.array(np.nonzero(m))
        x, y, z = [points[i, :] for i in range(3)]
        points = np.stack([x, y, z], axis=-1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def extract_surface(pcd, kdtree_depth=16, n_threads=4, show=False):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=kdtree_depth, n_threads=n_threads, width=0, scale=1.1, linear_fit=False)
        densities = np.asarray(densities)
        density_colors = plt.get_cmap('plasma')(
            (densities - densities.min()) / (densities.max() - densities.min()))
        density_colors = density_colors[:, :3]
        density_mesh = o3d.geometry.TriangleMesh()
        density_mesh.vertices = mesh.vertices
        density_mesh.triangles = mesh.triangles
        density_mesh.triangle_normals = mesh.triangle_normals
        density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        bbox = pcd.get_axis_aligned_bounding_box()
        d_mesh_crop = density_mesh.crop(bbox)
        o_mesh_crop = mesh.crop(bbox)
        
        if show:
            o3d.visualization.draw_geometries([d_mesh_crop])

        return o_mesh_crop, d_mesh_crop        
