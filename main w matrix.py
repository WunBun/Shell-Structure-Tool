from tkinter import *
from numpy import *

# matplotlib 3D

class Point():
    def __init__(self, x, y, condition, connected = None, index = 0):
        self.x = x
        self.y = y
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

        ans = []

        for i in range(1, resolution):
            ans.append(Point(self.x + i * dx, self.y + i * dy, "free", index = num_pts + i - 1))

        return ans
    
    def distance(self, p2):
        return ((self.x - p2.x) ** 2 + (self.y - p2.y) ** 2) ** 0.5
    
    def __repr__(self):
        return f"{self.index}: ({self.x:0.0f}, {self.y:0.0f}) -> {[point.index for point in self.connected]}"

class ScreenSpace(Frame):
    def __init__(self):
        super().__init__()

        self.master.title("Screen Space")
        self.pack(fill = BOTH, expand = 1)

        self.canvas = Canvas(self)
        self.canvas.pack(fill = BOTH, expand = 1)

        self.points = []

        self.last_added = None

        self.epsilon = 10

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
                                    outline = "#fff", fill = "#fff", width = 0)
            # draw dots at all the points in self.points

            for other in point.connected:
                self.canvas.create_line(point.x, point.y,
                                        other.x, other.y,
                                        fill = '#fff',
                                        width = 0.5)
                
    def snap(self, event):
        event_pt = Point(event.x, event.y, "fixed", index = len(self.points))

        min_dist = float("inf")
        min_pt = None

        if self.points:
            for point in self.points:
                dist = point.distance(event_pt)
                if dist < min_dist:
                    min_dist = dist
                    min_pt = point

        return min_pt if min_dist < self.epsilon else event_pt
    
    def add_point(self, event):
        newpt = self.snap(event)
        if newpt not in self.points:
            self.points.append(newpt)
        self.last_added = newpt

    def add_chain(self, event):
        startpt = self.last_added
        endpt = self.snap(event)
        midpts = startpt.interpolate(endpt, 30, len(self.points))

        chain_pts = [startpt, *midpts, endpt]
        for i, point in enumerate(chain_pts):
            if i > 0:
                point.connected.add(chain_pts[i-1])
            # if i < len(chain_pts) - 1:
            #     point.connected.add(chain_pts[i+1])
        
        self.points.extend(chain_pts[1:-1])

        if endpt not in self.points:
            endpt.index += len(midpts)
            self.points.append(endpt)
        
        self.last_added = endpt

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
        model = FDMModel(self.points)
        model.update(self.points)

class FDMModel():
    def __init__(self, points, q = 10):
        C = zeros((1, len(points)))

        self.fixed = []
        self.free = []

        x_fixed = [[], []]
        x_free = [[], []]

        for i, startpt in enumerate(points):
            for endpt in startpt.connected:
                newrow = zeros((1, len(points)))
                newrow[0, startpt.index] = 1
                newrow[0, endpt.index] = -1
                C = vstack([C, newrow])

            if startpt.condition == "free":
                self.free.append(i)
                x_free[0].append(startpt.x)
                x_free[1].append(startpt.y)
            else:
                self.fixed.append(i)
                x_fixed[0].append(startpt.x)
                x_fixed[1].append(startpt.y)

        C = C[1:, :] # connectivity matrix

        C_fixed = C[:, self.fixed]
        C_free = C[:, self.free]

        Q = diag([q]*C_free.shape[0])
        p = array([[0, 9.8] for i in range(len(self.free))])

        Dn = C_free.T @ Q @ C_free
        Df = C_free.T @ Q @ C_fixed

        x_fixed = array(x_fixed).T
        x_free = array(x_free).T

        self.new_x = linalg.solve(Dn, p - Df @ x_fixed)

    def update(self, points):
        for i, point_ind in enumerate(self.free):
            point = points[point_ind]
            point.x = self.new_x[i, 0]
            point.y = self.new_x[i, 1]




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

while True:
    ex.draw()
    root.update_idletasks()
    root.update()