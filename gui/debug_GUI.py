#!/usr/bin/env python2

import threading
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import vision


class GUI(object):
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file("debug_GUI.glade")

        self.window = self.builder.get_object("window1")
        self.window.show_all()
        print "Destroy connect: %d" % self.window.connect('destroy',
                                                          self.destroyed)

    def destroyed(self, _):
        print("Destroyed")
        Gtk.main_quit()

    def get_camera(self):
        return self.builder.get_object("camera")

    def run(self):
        Gtk.main()


if __name__ == "__main__":
    gui = GUI()
    vision = vision.Vision('10.0.7.15', gui.get_camera())
    vision_thread = threading.Thread(target=vision.run)

    vision_thread.start()

    gui.run()
    vision.stop()
    vision_thread.join()
