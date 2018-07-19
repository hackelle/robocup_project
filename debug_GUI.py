#!/usr/bin/env python2

import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib

from naoqi import ALProxy
import numpy as np
import matplotlib.pyplot as plot
import cv2

vision = ALProxy('RobocupVision', '10.0.7.15', 9559)
cameraId = 0 # 0 for
data = vision.getBGR24Image(cameraId)
image = np.fromstring(data, dtype=np.uint8).reshape((480, 640, 3))
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gbytes = GLib.Bytes.new(rgb_img.tostring())

pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
    gbytes,
    GdkPixbuf.Colorspace.RGB,
    False,
    8,
    640,
    480,
    640 * 3,
)

builder = Gtk.Builder()
builder.add_from_file("debug_GUI.glade")

window = builder.get_object("window1")
window.show_all()

img = builder.get_object("head_detect")
img.set_from_pixbuf(pixbuf)

Gtk.main()
