from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from meshlib import mrmeshpy as mm
from meshlib import mrmeshnumpy as mn
import meshlib.mrmeshpy as mrmeshpy

import scipy.optimize as opt

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

        self.epsilon = 5

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
        self.model = FDMMesh(self.points, self.lines)

    def simulate_model(self, event):
        self.model.create_FDMModel()

    def update_model(self, event):
        self.model.update()

    def csu(self, event):
        self.model = FDMMesh(self.points, self.lines)
        self.model.solve_FDMModel()
        self.model.update()

    def opt_q(self, event):
        self.model = FDMMesh(self.points, self.lines)
        self.model.create_q_optimized_model()

    def plot(self, event):
        self.model.plot()

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
        props.maxEdgeSplits = 500
        mrmeshpy.subdivideMesh(self.tri, props)

        self.orig_verts = np.array(mn.getNumpyVerts(self.tri))
        self.verts = self.orig_verts
        self.faces = np.array(mn.getNumpyFaces(self.tri.topology))

        self.free_edges = None
        self.rib_edges = None

        self.FDM_invariants()

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
        fig, ax = plt.subplots(subplot_kw={"projection": "3d", "autoscale_on": "True"})

        ax.grid(False)

        ax.set_box_aspect((
            np.ptp(self.verts[:, 0]),
            np.ptp(self.verts[:, 1]),
            np.ptp(self.verts[:, 2]) if np.ptp(self.verts[:, 2]) > 0 else np.ptp(self.verts[:, 1])))

        ax.set(
            xticklabels=[],
            yticklabels=[],
            zticklabels=[],
            title = f"gen = {self.q_gen:0.2f}, rib = {self.q_rib:0.2f}, FL = {self.sum_FL():0.2f}, \n area = {self.total_area():0.2f}, supports = {self.support_reactions():0.2f}"
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

    def FDM_invariants(self):
        self.orig_fixed = [p for p in self.base_points if p.condition == "fixed"]
        self.orig_free = [p for p in self.base_points if p.condition == "free"]

        # Create matrixes of fixed and free points and determine their indices

        self.x_fixed = np.empty((0, 3))
        self.x_free = np.empty((0, 3))

        self.fixed_ind = []
        self.free_ind = []

        for i, vert in enumerate(self.orig_verts):
            vert_pt = Point(vert[0], vert[1], vert[2])
            if min(vert_pt.distance(p) for p in self.orig_fixed) < self.epsilon:
                self.x_fixed = np.vstack((self.x_fixed, np.array(vert)))
                self.fixed_ind.append(i)
            else:
                self.x_free = np.vstack((self.x_free, np.array(vert)))
                self.free_ind.append(i)

        # ribs, free edges, and support edges

        self.rib_edges = set()
        self.free_edges = set()
        self.support_edges = set()

        self.C = np.zeros((0, len(self.fixed_ind + self.free_ind)))
        self.q_types = []

        for face in self.faces:
            for i in range(3):
                edge_start = self.orig_verts[face[i], :]
                edge_end = self.orig_verts[face[(i+1) % 3], :]

                # skip repeat edges
                if (face[i], face[(i+1) % 3]) in self.rib_edges or (face[(i+1) % 3], face[i]) in self.rib_edges:
                    continue

                if (face[i], face[(i+1) % 3]) in self.free_edges or (face[(i+1) % 3], face[i]) in self.free_edges:
                    continue

                if self.is_rib(edge_start, edge_end):
                    self.rib_edges.add((face[i], face[(i+1) % 3]))
                    self.q_types.append("rib")

                    if face[i] in self.fixed_ind or face[(i+1) % 3] in self.fixed_ind:
                        self.support_edges.add(((face[i], face[(i+1) % 3]), "rib"))
                else:
                    self.free_edges.add((face[i], face[(i+1) % 3]))
                    self.q_types.append("gen")

                    if face[i] in self.fixed_ind or face[(i+1) % 3] in self.fixed_ind:
                        self.support_edges.add(((face[i], face[(i+1) % 3]), "gen"))

                # create a new row in the connectivity matrix
                newrow = np.zeros((1, len(self.fixed_ind + self.free_ind)))
                newrow[0, face[i]] = 1
                newrow[0, face[(i+1) % 3]] = -1
                self.C = np.vstack([self.C, newrow])

    def solve_FDMModel(self, qs = (2, 40)):
        # force densities
        self.q_gen = qs[0]
        self.q_rib = qs[1]

        x = self.orig_verts

        # Create the list of force densities corresponding to each edge

        qs = [self.q_gen if t == "gen" else self.q_rib for t in self.q_types]

        # solve FDM

        Q = np.diag(qs)
        
        C_fixed = self.C[:, self.fixed_ind]
        C_free = self.C[:, self.free_ind]

        p = np.array([[0, 0, -9.8] for i in range(len(self.free_ind))])

        Dn = C_free.T @ Q @ C_free
        Df = C_free.T @ Q @ C_fixed

        self.new_x = np.linalg.solve(Dn, p - Df @ self.x_fixed)

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
    
    #### AUTOMATIC OPTIMIZATION

    def create_q_optimized_model(self):
        def objective(qs):
            self.solve_FDMModel(qs)
            self.update()
            return self.sum_FL()
            
        x0 = [2, 40]
        
        # data = []

        # for q_gen in range(1, 30, 2):
        #     for q_rib in range(q_gen, 30, 2):
        #         datum = [q_gen, q_rib, objective([q_gen, q_rib])]
        #         data.append(datum)
        #         print(datum)

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d", "autoscale_on": "True"})
        # mpl.rcParams['axes3d.mouserotationstyle'] = 'azel'

        # ax.set(
        #     xlabel = "q_gen",
        #     ylabel = "q_rib"
        # )

        # ax.scatter(
        #     [datum[0] for datum in data],
        #     [datum[1] for datum in data],
        #     [datum[2] for datum in data],
        #     cmap = "berlin"
        # )
        
        # plt.show()

        method = "COBYLA"
        bounds = ((0.1, 200), (0.1, 200))
        cstr = opt.LinearConstraint(
            A = [1, -1],
            ub = -1,
            keep_feasible = True
        )

        result = opt.minimize(
            objective,
            x0,
            method = method,
            bounds = bounds,
            constraints = cstr
        )

        print(result)
        print(result.x)

        self.q_gen, self.q_rib = result.x

        self.solve_FDMModel(result.x)

        self.update()

        self.plot()
    
    #### STATISTICS FOR OBJECTIVES

    def tri_area(self, face, project = True):
        """
        Calculates the area of a triangular face. If project = True, projects the face to
        the xy plane before calculating the area.
        """

        AB = self.verts[face[0], :] - self.verts[face[1], :]
        AC = self.verts[face[0], :] - self.verts[face[2], :]

        if project:
            AB[2] = 0
            AC[2] = 0

        return 0.5 * np.linalg.norm(np.cross(AB, AC))
    
    def total_area(self, project = True):
        """
        Returns the total area of the structure. If project is true,
        it's the covered area on the xy-plane. Else, it's the surface area.
        """

        return sum(self.tri_area(face, project) for face in self.faces)
    
    def sum_FL(self):
        """
        Returns sum of force times length of mesh members as an analog
        for structural weight.
        """
        ans = 0

        for edge in self.rib_edges:
            length = np.linalg.norm(self.verts[edge[0]] - self.verts[edge[1]])
            ans += self.q_rib * length * length

        for edge in self.free_edges:
            length = np.linalg.norm(self.verts[edge[0]] - self.verts[edge[1]])
            ans += self.q_gen * length * length

        return ans
    
    def support_reactions(self):
        """
        Returns the sum of the magnitudes of the forces going to the supports.
        Supports are fixed vertices.
        """
        ans = 0

        for edge, q in self.support_edges:
            length = np.linalg.norm(self.verts[edge[0]] - self.verts[edge[1]])
            ans += length * (self.q_gen if q == "gen" else self.q_rib)

        return ans

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

root.bind("<Command-KeyPress-a>", ex.csu)
root.bind("<Command-KeyPress-o>", ex.opt_q)

while True:
    ex.draw()
    root.update_idletasks()
    root.update()