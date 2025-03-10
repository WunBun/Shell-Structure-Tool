from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
import numpy as np
 
u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:100j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)
 
# Prepare for MeshLib PointCloud
verts = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1).reshape(-1, 3)
print(verts)
# Create MeshLib PointCloud from np ndarray
pc = mn.pointCloudFromPoints(verts)
print(pc)
# Remove duplicate points
samplingSettings = mm.UniformSamplingSettings()
samplingSettings.distance = 1e-3
pc.validPoints = mm.pointUniformSampling(pc, samplingSettings)
pc.invalidateCaches()
 
# Triangulate it
triangulated_pc = mm.triangulatePointCloud(pc)
print(triangulated_pc)
 
# Fix possible issues
offsetParams = mm.OffsetParameters()
offsetParams.voxelSize = mm.suggestVoxelSize( triangulated_pc, 5e6 )
triangulated_pc = mm.offsetMesh(triangulated_pc, 0.0, params=offsetParams)