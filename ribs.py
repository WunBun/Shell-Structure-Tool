from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
import meshlib.mrmeshpy as mrmeshpy

import plotly.graph_objects as go

class Point():
    def __init__(self, x, y, z, condition = "free", connected = None, index = 0):
        self.x = x
        self.y = y
        self.z = z
        self.condition = condition # free, fixed
        self.connected = set() if connected is None else connected

        self._radius = 2 if self.condition == "free" else 5

        self.index = index

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def set_cond(self, condition):
        self.condition = condition
        self._radius = 2 if self.condition == "free" else 5

    def interpolate(self, end, resolution, num_pts):
        dx = (end.x - self.x)/resolution
        dy = (end.y - self.y)/resolution
        dz = (end.z - self.z)/resolution

        ans = []

        for i in range(1, resolution):
            ans.append(Point(self.x + i * dx, self.y + i * dy, self.z + i * dz, "free", index = num_pts + i - 1))

        return ans
    
    def distance(self, p2):
        return ((self.x - p2.x) ** 2 + (self.y - p2.y) ** 2 + (self.z - p2.z) ** 2) ** 0.5
    
    def plane_distance(self, p2):
        return ((self.x - p2.x) ** 2 + (self.y - p2.y) ** 2) ** 0.5
    
    def __repr__(self):
        return f"{self.index}: ({self.x:0.0f}, {self.y:0.0f}, {self.z:0.0f}) -> {[point.index for point in self.connected]}"
    
    # def __eq__(self, other):
    #     return self.distance(other) < 10e-3

class Line():
    def __init__(self, pstart, pend):
        self.pstart = pstart
        self.pend = pend
        self.direction = np.array([pend.x - pstart.x, pend.y - pstart.y, pend.z - pstart.z])
        self.direction = self.direction / np.linalg.norm(self.direction)

    def distance_2d(self, pt):
        v = self.direction[:2][..., None]
        p0 = np.array([self.pstart.x, self.pstart.y])[..., None]
        p = np.array([pt.x, pt.y])[..., None]

        closest = (v @ v.T) @ (p - p0)/(v.T @ v) + p0

        dist = abs((self.pend.y - self.pstart.y) * pt.x
                   - (self.pend.x - self.pstart.x) * pt.y
                   + self.pend.x * self.pstart.y
                   - self.pend.y * self.pstart.x)/np.sqrt(
                       (self.pend.y - self.pstart.y) ** 2
                       + (self.pend.x - self.pstart.x) ** 2
                   )
        
        return closest, dist

    def __repr__(self):
        return f"{self.pstart.index}: ({self.pstart.x:0.0f}, {self.pstart.y:0.0f}, {self.pstart.z:0.0f}) -> {self.pend.index}: ({self.pend.x:0.0f}, {self.pend.y:0.0f}, {self.pend.z:0.0f})"

class ScreenSpace(Frame):
    def __init__(self):
        super().__init__()

        self.master.title("Screen Space")
        self.pack(fill = BOTH, expand = 1)

        self.canvas = Canvas(self)
        self.canvas.pack(fill = BOTH, expand = 1)

        self.points = []
        self.lines = []

        self.last_added = None

        self.epsilon = 10

        self.model = FDMMesh()

    def clear(self, event):
        self.points = [] # empty out the objects
        self.lines = []
        self.last_added = None

        self.canvas.delete("all")

        self.clear_labels()

    def draw(self):
        self.canvas.delete("all") # clear the canvas

        for point in self.points:
            self.canvas.create_oval(point.x - point.radius, point.y - point.radius,
                                    point.x + point.radius, point.y + point.radius,
                                    outline = "#fff", fill = "#fff", width = point.z/2)
            # draw dots at all the points in self.points

            for other in point.connected:
                self.canvas.create_line(point.x, point.y,
                                        other.x, other.y,
                                        fill = '#fff',
                                        width = 0.5)
                                        
    def plot(self, event):
        print("plotting")

        self.model.plot()
                
    def snap(self, event):
        event_pt = Point(event.x, event.y, 0, "fixed", index = len(self.points))

        min_dist = float("inf")
        min_pt = None
        min_line = None

        # closest point?
        if self.points:
            for point in self.points:
                dist = point.plane_distance(event_pt)
                if dist < min_dist:
                    min_dist = dist
                    min_pt = point

        if min_dist < self.epsilon:
            return min_pt, None
        
        # no closest point -> closest line
        min_dist = float("inf")
        min_pt = None

        if self.lines:
            for line in self.lines:
                point, dist = line.distance_2d(event_pt)
                if dist < min_dist:
                    min_dist = dist
                    min_pt = point
                    min_line = line

            min_pt = Point(float(min_pt[0]), float(min_pt[1]), 0, "free", index = len(self.points))

        return (min_pt, min_line) if min_dist < self.epsilon else (event_pt, None)
    
    def add_point(self, event):
        newpt, line = self.snap(event)

        if line:
            line.pstart.connected.remove(line.pend)
            line.pstart.connected.add(newpt)
            newpt.connected.add(line.pend)

            self.lines.remove(line)
            self.lines.append(Line(line.pstart, newpt))
            self.lines.append(Line(newpt, line.pend))
        
        if newpt not in self.points:
            self.points.append(newpt)

        self.last_added = newpt

    def add_chain(self, event):
        startpt = self.last_added
        newpt, line = self.snap(event)

        if line:
            line.pstart.connected.remove(line.pend)
            line.pstart.connected.add(newpt)
            newpt.connected.add(line.pend)
            startpt.connected.add(newpt)

            self.lines.remove(line)
            self.lines.append(Line(line.pstart, newpt))
            self.lines.append(Line(newpt, line.pend))
            self.lines.append(Line(startpt, newpt))

            self.points.append(newpt)
            self.last_added = newpt
        else:
            startpt.connected.add(newpt)
            
            self.lines.append(Line(startpt, newpt))

            if newpt not in self.points:
                self.points.append(newpt)
            self.last_added = newpt

    def dump(self, event):
        print(self.points)

    def clear_labels(self, _ = None):
        for widget in root.winfo_children():
            if isinstance(widget, Label):
                widget.destroy()

    def add_labels(self, event):
        self.clear_labels()

        for point in self.points:
            tx = Label(text = f"{point.index}", font=("Ariel", 10, "italic"), fg = "white")
            tx.place(x = point.x, y = point.y)

    def create_mesh(self, event):
        # model = FDMModel(self.points)
        # model.update(self.points)

        self.model = FDMMesh(self.points, self.lines)

    def simulate_model(self, event):
        self.model.create_FDMModel()

    def update_model(self, event):
        self.model.update()

class FDMMesh():
    def __init__(self, points = None, lines = None):
        self.epsilon = 10e-2

        if not points or not lines:
            return
        
        self.base_points = points
        self.ribs = lines

        self.verts = np.array(
            [
                np.array([float(point.x) for point in points]),
                np.array([float(point.y) for point in points]),
                np.array([float(point.z) for point in points])
            ]
        ).T

        # Create MeshLib PointCloud from np ndarray
        self.pc = mn.pointCloudFromPoints(self.verts)

        # Remove duplicate points
        samplingSettings = mm.UniformSamplingSettings()
        samplingSettings.distance = 1e-3
        self.pc.validPoints = mm.pointUniformSampling(self.pc, samplingSettings)
        self.pc.invalidateCaches()

        # Triangulate it
        self.tri = mm.triangulatePointCloud(self.pc)

        props = mrmeshpy.SubdivideSettings()
        props.maxDeviationAfterFlip = 0
        props.maxAngleChangeAfterFlip = 0

        no_flip = mm.UndirectedEdgeBitSet(
            mm.BitSet(
                self.count_edges(mn.getNumpyFaces(self.tri.topology)),
                True
            )
        )
        props.notFlippable = no_flip
        # props.criticalAspectRatioFlip = 1.0
        props.maxEdgeSplits = 500
        mrmeshpy.subdivideMesh(self.tri, props)

        self.verts = mn.getNumpyVerts(self.tri)
        self.faces = mn.getNumpyFaces(self.tri.topology)
        
        # # Fix possible issues
        # offsetParams = mm.OffsetParameters()
        # offsetParams.voxelSize = mm.suggestVoxelSize(self.tri, 5e6)
        # self.tri = mm.offsetMesh(self.tri, 0.0, params=offsetParams)

        self.free_edges = None
        self.rib_edges = None

    def count_edges(self, faces, list_edges = False):
        edges = set()

        for face in faces:
            for i in range(3):
                if (face[i], face[(i + 1) % 3]) not in edges and (face[(i + 1) % 3], face[i]) not in edges:
                    edges.add((face[i], face[(i + 1) % 3]))

        if list_edges:
            return edges, len(edges)
        
        return len(edges)

    def plot(self):
        print("plotting")

        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "autoscale_on": "True"})

        ax.grid(False)

        ax.set_box_aspect((
            np.ptp(self.verts[:, 0]),
            np.ptp(self.verts[:, 1]),
            np.ptp(self.verts[:, 2]) if np.ptp(self.verts[:, 2]) > 0 else np.ptp(self.verts[:, 1])))

        ax.set(
            xticklabels=[],
            yticklabels=[],
            zticklabels=[]
            )
        
        ax.view_init(elev=-30, azim=45, roll=180)
        mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

        face_polys = np.array(
            [
                [
                    self.verts[face[0]],
                    self.verts[face[1]],
                    self.verts[face[2]],
                    self.verts[face[0]]
                ]
                for face in self.faces
            ]
        )
        
        poly = Poly3DCollection(
            face_polys,
            alpha = 0.7,
            shade = True,
            facecolors = "xkcd:light green",
            # zorder = 0
            # lightsource
            )
        
        ax.add_collection3d(poly)

        
        if not self.free_edges:
            for face in self.faces:
                for i in range(3):
                    ax.plot(
                        [self.verts[face[i]][0], self.verts[face[(i + 1)%3]][0]],
                        [self.verts[face[i]][1], self.verts[face[(i + 1)%3]][1]],
                        [self.verts[face[i]][2], self.verts[face[(i + 1)%3]][2]],
                        # zorder = 1
                    )
        else:
            for edge in self.free_edges:
                ax.plot(
                        [self.verts[edge[0]][0], self.verts[edge[1]][0]],
                        [self.verts[edge[0]][1], self.verts[edge[1]][1]],
                        [self.verts[edge[0]][2], self.verts[edge[1]][2]],
                        "g-",
                        linewidth = 0.5,
                        # zorder = 2
                    )
            for edge in self.rib_edges:
                ax.plot(
                        [self.verts[edge[0]][0], self.verts[edge[1]][0]],
                        [self.verts[edge[0]][1], self.verts[edge[1]][1]],
                        [self.verts[edge[0]][2], self.verts[edge[1]][2]],
                        "k-",
                        linewidth = 1,
                        # zorder = 3
                    )

        plt.show()

    def create_FDMModel(self):
        # force densities
        q_gen = 2
        q_rib = 40

        orig_fixed = [p for p in self.base_points if p.condition == "fixed"]
        orig_free = [p for p in self.base_points if p.condition == "free"]

        # Create matrixes of fixed and free points and determine their indices

        x = np.array(self.verts)
        x_fixed = np.empty((0, 3))
        x_free = np.empty((0, 3))

        fixed_ind = []
        self.fixed_ind = fixed_ind
        free_ind = []
        self.free_ind = free_ind

        for i, vert in enumerate(self.verts):
            vert_pt = Point(vert[0], vert[1], vert[2])
            if min(vert_pt.distance(p) for p in orig_fixed) < self.epsilon:
                x_fixed = np.vstack((x_fixed, np.array(vert)))
                fixed_ind.append(i)
            else:
                x_free = np.vstack((x_free, np.array(vert)))
                free_ind.append(i)

        # Create the connectivity matrix and list of force densities corresponding to each edge

        self.rib_edges = set()
        self.free_edges = set()
        qs = []
        C = np.zeros((0, len(fixed_ind + free_ind)))

        for face in self.faces:
            for i in range(3):
                edge_start = x[face[i], :]
                edge_end = x[face[(i+1) % 3], :]

                # skip repeat edges
                if (face[i], face[(i+1) % 3]) in self.rib_edges or (face[(i+1) % 3], face[i]) in self.rib_edges:
                    continue

                if (face[i], face[(i+1) % 3]) in self.free_edges or (face[(i+1) % 3], face[i]) in self.free_edges:
                    continue

                if self.is_rib(edge_start, edge_end):
                    qs.append(q_rib)
                    self.rib_edges.add((face[i], face[(i+1) % 3]))
                else:
                    qs.append(q_gen)
                    self.free_edges.add((face[i], face[(i+1) % 3]))

                # create a new row in the connectivity matrix
                newrow = np.zeros((1, len(fixed_ind + free_ind)))
                newrow[0, face[i]] = 1
                newrow[0, face[(i+1) % 3]] = -1
                C = np.vstack([C, newrow])

        # solve FDM

        Q = np.diag(qs)
        print(Q)
        
        C_fixed = C[:, fixed_ind]
        C_free = C[:, free_ind]

        p = np.array([[0, 0, -9.8] for i in range(len(free_ind))])

        Dn = C_free.T @ Q @ C_free
        Df = C_free.T @ Q @ C_fixed

        self.new_x = np.linalg.solve(Dn, p - Df @ x_fixed)

        print(self.new_x)

    def is_rib(self, edge_start, edge_end):
        edge_mid = (edge_start + edge_end) / 2

        for rib in self.ribs:
            dot_prod = (
                (edge_mid[0] - rib.pstart.x) * (rib.pend.x - rib.pstart.x) +
                (edge_mid[1] - rib.pstart.y) * (rib.pend.y - rib.pstart.y) +
                (edge_mid[2] - rib.pstart.z) * (rib.pend.z - rib.pstart.z)
            )

            if (rib.distance_2d(Point(edge_mid[0], edge_mid[1], edge_mid[2]))[1] < self.epsilon
                and
                dot_prod >= 0
                and
                dot_prod <= rib.pstart.distance(rib.pend) ** 2
            ):
                return True
            
        return False

    def update(self):
        for i, ind in enumerate(self.free_ind):
            self.verts[ind, :] = self.new_x[i, :]

class FDMModel():
    def __init__(self, points, q = 10):
        C = np.zeros((1, len(points)))

        self.fixed = []
        self.free = []

        x_fixed = [[], [], []]
        x_free = [[], [], []]

        for i, startpt in enumerate(points):
            for endpt in startpt.connected:
                newrow = np.zeros((1, len(points)))
                newrow[0, startpt.index] = 1
                newrow[0, endpt.index] = -1
                C = np.vstack([C, newrow])

            if startpt.condition == "free":
                self.free.append(i)
                x_free[0].append(startpt.x)
                x_free[1].append(startpt.y)
                x_free[2].append(startpt.z)
            else:
                self.fixed.append(i)
                x_fixed[0].append(startpt.x)
                x_fixed[1].append(startpt.y)
                x_fixed[2].append(startpt.z)

        C = C[1:, :] # connectivity matrix

        C_fixed = C[:, self.fixed]
        C_free = C[:, self.free]

        Q =  np.diag([q]*C_free.shape[0])
        p = np.array([[0, 0, 9.8] for i in range(len(self.free))])

        Dn = C_free.T @ Q @ C_free
        Df = C_free.T @ Q @ C_fixed

        x_fixed = np.array(x_fixed).T
        x_free = np.array(x_free).T

        self.new_x = np.linalg.solve(Dn, p - Df @ x_fixed)

    def update(self, points):
        for i, point_ind in enumerate(self.free):
            point = points[point_ind]
            point.x = self.new_x[i, 0]
            point.y = self.new_x[i, 1]
            point.z = self.new_x[i, 2]

root = Tk()
root.geometry('500x400')

ex = ScreenSpace()

root.bind("<Button-1>", ex.add_point)
root.bind("<Shift-Button-1>", ex.add_chain)
root.bind("<Command-KeyPress-d>", ex.clear)
root.bind("<KeyPress-p>", ex.dump)
root.bind("<KeyPress-l>", ex.add_labels)
root.bind("<Command-KeyPress-l>", ex.clear_labels)
root.bind("<Command-KeyPress-c>", ex.create_mesh)
root.bind("<Command-KeyPress-s>", ex.simulate_model)
root.bind("<Command-KeyPress-u>", ex.update_model)
root.bind("<Command-KeyPress-p>", ex.plot)

while True:
    ex.draw()
    root.update_idletasks()
    root.update()