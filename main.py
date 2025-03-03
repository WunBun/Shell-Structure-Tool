from tkinter import *

# comment

# another comment

class Point():
    def __init__(self, x, y, condition):
        self.x = x
        self.y = y
        self.condition = condition # free, fixed

        self.radius = 3 if self.condition == "free" else 5

    def set_cond(self, condition):
        self.condition = condition
        self.radius = 3 if self.condition == "free" else 5

class Chain():
    def __init__(self, start, end, thickness, resolution):
        self.start = start
        self.end = end
        self.th = thickness
        self.res = resolution
        self.pts = []

        self.interpolate(start, end, resolution)

    def interpolate(self, start, end, resolution):
        dx = (end.x - start.x)/resolution
        dy = (end.y - start.y)/resolution

        for i in range(resolution + 1):
            self.pts.append(Point(start.x + i * dx, start.y + i * dy, "free"))

    def closest_point(self, event):
        clickpt = Point(event.x, event.y, "free")

        min_dist = self.distance(self.pts[0], clickpt)
        ans = self.pts[0]

        for point in self.pts:
            dist = self.distance(point, clickpt)
            if dist < min_dist:
                min_dist = dist
                ans = point

        return (ans, min_dist)

    def distance(self, p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

class ScreenSpace(Frame):
    def __init__(self):
        super().__init__()

        self.master.title("Screen Space")
        self.pack(fill = BOTH, expand = 1)

        self.canvas = Canvas(self)
        self.canvas.pack(fill = BOTH, expand = 1)

        self.objects = {
            "points": [],
            "lines": [],
            "chains": []
        }

        self.last_added = None

        self.epsilon = 10

    def clear(self, event):
        self.objects = {
            "points": [],
            "chains": []
        }

        self.canvas.delete("all")

    def draw(self):
        self.canvas.delete("all") # clear the canvas

        for point in self.objects["points"]:
            self.canvas.create_oval(point.x - point.radius, point.y - point.radius,
                                    point.x + point.radius, point.y + point.radius,
                                    outline = "#fff", fill = "#fff", width = 0)
            # draw dots at all the points in self.points
            
        for chain in self.objects["chains"]:
            for i in range(len(chain.pts) - 1):
                st = chain.pts[i]
                en = chain.pts[i+1]

                self.canvas.create_line(st.x, st.y,
                                        en.x, en.y,
                                        fill = "#fff",
                                        width = chain.th)
                
                self.canvas.create_oval(st.x - st.radius, st.y - st.radius,
                                        st.x + st.radius, st.y + st.radius,
                                        outline = "#fff", fill = "#fff", width = 0)
                
    def snap_pt(self, event):
        if self.objects["chains"]:
            ans = None
            min_dist = float("inf")

            for chain in self.objects["chains"]:
                closest, dist = chain.closest_point(event)

                if dist < min_dist:
                    ans = closest
                    min_dist = dist

            return ans if min_dist < self.epsilon else None
        
        return None
    
    def add_point(self, event):
        snap = self.snap_pt(event)
        newpt = snap if snap else Point(event.x, event.y, "fixed")
        self.objects["points"].append(newpt)
        self.last_added = newpt

    def add_chain(self, event):
        startpt = self.objects["points"][-1]

        snap = self.snap_pt(event)
        endpt = snap if snap else Point(event.x, event.y, "fixed")
        self.objects["points"].append(endpt)

        newchain = Chain(startpt, endpt, 1, 10)
        self.objects["chains"].append(newchain)
        self.last_added = endpt


root = Tk()
root.geometry('500x400')

ex = ScreenSpace()

root.bind("<Button-1>", ex.add_point)
root.bind("<Shift-Button-1>", ex.add_chain)
root.bind("<Command-KeyPress-d>", ex.clear)

while True:
    ex.draw()
    root.update_idletasks()
    root.update()