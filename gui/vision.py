#!/usr/bin/env python2

import threading

import gi
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import cairo

from naoqi import ALProxy
import numpy as np
import matplotlib.pyplot as plot
import cv2


class Vision(object):
    def __init__(self, address, canvas):
        self.pixbuf = None
        self.pixbuf_lock = threading.Lock()
        self.canvas = canvas
        print "Draw connect: %d" % self.canvas.connect('draw',
                                                       self.update_image)
        self.proxy = ALProxy('RobocupVision', address,
                             9559)

    def get_image(self, camera_id=0):
        data = self.proxy.getBGR24Image(camera_id)
        image = np.fromstring(data, dtype=np.uint8).reshape(
            (480, 640, 3)
        )
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gbytes = GLib.Bytes.new(rgb_img.tostring())

        return GdkPixbuf.Pixbuf.new_from_bytes(
            gbytes,
            GdkPixbuf.Colorspace.RGB,
            False,
            8,
            640,
            480,
            640 * 3,
        )

    def run(self):
        self._running = True
        while self._running:
            img = self.get_image()
            print "Run: WAITING"
            with self.pixbuf_lock:
                print "Run: ACQUIRED"
                self.pixbuf = img
            self.canvas.queue_draw()
            print "Got image!"

    def update_image(self, _, context):
        print "Update: WAITING"
        with self.pixbuf_lock:
            print "Update: ACQUIRED"
            if self.pixbuf is None:
                print "No pixbuf!"
                return

            print "Updating"
            width = self.canvas.get_allocated_width()
            height = self.canvas.get_allocated_height()
            context.set_source_rgb(0, 0, 0)
            context.paint()
            print "Scaling"
            img = self.pixbuf.scale_simple(width, height,
                                           GdkPixbuf.InterpType.BILINEAR)

        print "Painting"
        Gdk.cairo_set_source_pixbuf(context, img, 0, 0)
        context.paint()
        # self.gtk_image.set_from_pixbuf(self.pixbuf)

        print "Update done"
        return False

    def stop(self):
        self._running = False
