#!/usr/bin/env python3

import json
import sys
import tkinter
from tkinter import N, S, E, W
from PIL import ImageTk, Image


class GUI(tkinter.Frame):
    def __init__(self, master, coder, root=None):
        super().__init__(master=None)

        print(self._usage())

        if root is None:
            root = master

        self.canvas = tkinter.Canvas(self, cursor='cross',
                                     width=800, height=600)

        self._init_scrollbar(self.canvas)

        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.sbarv.grid(row=0, column=1, sticky=N+S)
        self.sbarh.grid(row=1, column=0, sticky=E+W)

        self._img = None
        self._tk_img = None
        self._canvas_img = None

        self._bind_events(root)

        self._lines = []
        self.reset()

        self._coder = coder

    def _usage(self):
        return '''How to use this:

        Draw a rectangle using the mouse.
        Fine tune the rectangles using the arrow keys.
        TAB changes the active side.
        Press ENTER when done.'''

    def reset(self):
        self._rects = []
        self._active_rect = -1

        for rect in self._lines:
            for line in rect:
                self.canvas.delete(line)
        self._lines = []
        self._active_half = -1
        self._dragging = False

    def load_image(self, path):
        self._img = Image.open(path)
        self.width, self.height = self._img.size
        self.canvas.config(scrollregion=(0, 0, self.width, self.height))
        self._tk_img = ImageTk.PhotoImage(self._img)

        if self._canvas_img is None:
            self._canvas_img = self.canvas.create_image(
                0, 0, anchor='nw', image=self._tk_img)
        else:
            self.canvas.itemconfig(self._canvas_img,
                                   image=self._tk_img)

    def _init_scrollbar(self, canvas):
        self.sbarv = tkinter.Scrollbar(self, orient=tkinter.VERTICAL)
        self.sbarh = tkinter.Scrollbar(self, orient=tkinter.HORIZONTAL)
        self.sbarv.config(command=canvas.yview)
        self.sbarh.config(command=canvas.xview)

        canvas.config(yscrollcommand=self.sbarv.set,
                      xscrollcommand=self.sbarh.set)

    def _bind_events(self, root):
        self.canvas.bind('<ButtonPress-1>', self._on_button_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_button_release)
        root.bind('<Return>', self._on_return)
        root.bind('<Tab>', self._on_tab)
        for key in ('<Left>', '<Right>', '<Up>', '<Down>'):
            root.bind(key, self._on_arrow)

    def _on_button_press(self, event):
        self._dragging = True

        self._rects.append([
            [self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)],
            [self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)]
        ])

        self._update_rect()
        self._set_active(half=-1, rect=len(self._rects)-1)
        return 'break'

    def _on_motion(self, event):
        self._rects[-1][1] = [self.canvas.canvasx(event.x),
                              self.canvas.canvasy(event.y)]

        self._update_rect()
        return 'break'

    def _on_button_release(self, _):
        self._dragging = False
        self._set_active(half=0)
        return 'break'

    def _on_return(self, ev):
        if self._dragging:
            return

        self._coder.done(self._rects)
        self.reset()
        self._rects = []
        self._active_rect = -1
        self._lines = []
        self._active_half = -1
        return 'break'

    def _on_tab(self, ev):
        if self._dragging:
            return

        # TODO: Why is this Control?
        if ev.state == 0x0004:
            if self._active_rect == -1:
                next_rect = 0
            else:
                next_rect = (self._active_rect + 1) % len(self._rects)
            self._set_active(rect=next_rect)
        else:
            if self._active_half == -1:
                next_half = 0
            else:
                next_half = (self._active_half + 1) % 2
            self._set_active(half=next_half)

        return 'break'

    def _on_arrow(self, event):
        if len(self._lines) == 0 or self._dragging:
            return

        if event.keysym == 'Left':
            self._move_x(-1)
        elif event.keysym == 'Right':
            self._move_x(1)
        elif event.keysym == 'Up':
            self._move_y(-1)
        elif event.keysym == 'Down':
            self._move_y(1)
        else:
            print('What is this? {}'.format(event.keysym))
        return 'break'

    def _update_rect(self):
        ((x1, y1), (x2, y2)) = self._rects[self._active_rect]

        if len(self._lines) < len(self._rects):
            self._lines.append([
                self.canvas.create_line(
                    x1, y1, x1, y1, fill='red'
                ) for i in range(4)
            ])

        lines = self._lines[self._active_rect]
        self.canvas.coords(lines[0], x1, y1,
                           x1, y2)
        self.canvas.coords(lines[1], x1, y2,
                           x2, y2)
        self.canvas.coords(lines[2], x2, y2,
                           x2, y1)
        self.canvas.coords(lines[3], x2, y1,
                           x1, y1)

    def _set_active(self, *, half=None, rect=None):
        if rect is not None:
            # Recolor old active rect
            for line in self._lines[self._active_rect]:
                self.canvas.itemconfig(line, fill='red')

            self._active_rect = rect
            # Force recolor of new active rect
            if half is None:
                half = self._active_half

        if half is not None and self._active_rect != -1:
            self._active_half = half
            colors = ['red', 'red']
            if half != -1:
                colors[self._active_half] = 'blue'
            for i, line in enumerate(self._lines[self._active_rect]):
                self.canvas.itemconfig(line, fill=colors[i // 2])

    def _move_x(self, dx):
        self._rects[self._active_rect][self._active_half][0] += dx
        self._update_rect()

    def _move_y(self, dy):
        self._rects[self._active_rect][(self._active_half + 1) % 2][1] += dy
        self._update_rect()


class Coder:
    def __init__(self):
        if len(sys.argv) < 3:
            print('Usage: {} OUTPUT IMAGE...'.format(sys.argv[0]))
            sys.exit(64)

        self._output = sys.argv[1]
        self._init_data(self._output)

        self._images = sys.argv[2:]

        self._root = tkinter.Tk()
        self._root.resizable(0, 0)
        self._gui = GUI(self._root, self)

        self._image = -1

    def _init_data(self, path):
        try:
            with open(path) as fh:
                self._data = json.load(fh)
        except FileNotFoundError:
            self._data = {}

    def _next_image(self):
        for i in range(self._image + 1, len(self._images)):
            img = self._images[i]
            if img not in self._data:
                print('Loading image {}'.format(img))
                self._image = i
                self._gui.load_image(img)
                return True

        print('Done!')
        self._root.destroy()
        return False

    def run(self):
        if self._next_image():
            self._gui.pack()
            self._root.mainloop()

    def done(self, rect):
        self._data[self._images[self._image]] = rect

        with open(self._output, 'w') as fh:
            json.dump(self._data, fh, indent=2)

        self._next_image()


if __name__ == '__main__':
    Coder().run()
