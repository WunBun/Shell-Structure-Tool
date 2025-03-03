from tkinter import *
import numpy as np
import matplotlib.pyplot as plt

from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn

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

        # xs = [point.x for point in self.points]
        # ys = [point.y for point in self.points]
        # zs = [point.z for point in self.points]

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.scatter(xs, ys, zs)

        # ax.set(xticklabels=[],
        #     yticklabels=[],
        #     zticklabels=[])

        # plt.show()
                
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

    def create_model(self, event):
        # model = FDMModel(self.points)
        # model.update(self.points)

        self.model = FDMMesh(self.points, self.lines)

class FDMMesh():
    def __init__(self, points = None, lines = None):
        if not points or not lines:
            return

        self.verts = np.array(
            [
                np.array([float(point.x) for point in points]),
                np.array([float(point.y) for point in points]),
                np.array([float(point.z) for point in points])
            ]
        ).T

        print(self.verts)

        # Create MeshLib PointCloud from np ndarray
        self.pc = mn.pointCloudFromPoints(self.verts)

        # Remove duplicate points
        samplingSettings = mm.UniformSamplingSettings()
        samplingSettings.distance = 1e-3
        self.pc.validPoints = mm.pointUniformSampling(self.pc, samplingSettings)
        self.pc.invalidateCaches()

        # Triangulate it
        self.tri = mm.triangulatePointCloud(self.pc)
        
        self.faces = mn.getNumpyFaces(self.tri.topology)
        
        # # Fix possible issues
        # offsetParams = mm.OffsetParameters()
        # offsetParams.voxelSize = mm.suggestVoxelSize(self.tri, 5e6)
        # self.tri = mm.offsetMesh(self.tri, 0.0, params=offsetParams)

    def plot(self):
        print("plotting")

        print(self.faces)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(
            xs = [v[0] for v in self.verts],
            ys = [v[1] for v in self.verts],
            zs = [v[2] for v in self.verts]
        )

        ax.set(xticklabels=[],
            yticklabels=[],
            zticklabels=[])
        
        for face in self.faces:
            for i in range(3):
                ax.plot(
                    [self.verts[face[i]][0], self.verts[face[(i + 1)%3]][0]],
                    [self.verts[face[i]][1], self.verts[face[(i + 1)%3]][1]],
                    [self.verts[face[i]][2], self.verts[face[(i + 1)%3]][2]],
                )

        plt.show()



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
root.bind("<Command-KeyPress-c>", ex.create_model)
root.bind("<Command-KeyPress-p>", ex.plot)

while True:
    ex.draw()
    root.update_idletasks()
    root.update()