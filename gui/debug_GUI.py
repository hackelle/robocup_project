#!/usr/bin/env python2

import threading
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf

import vision


class GUI(object):
    def __init__(self):
        self.builder = Gtk.Builder()
        self.builder.add_from_file('debug_GUI.glade')

        self.window = self.builder.get_object('window')
        self.window.show_all()
        self.camera = self.builder.get_object('camera')
        self.temp = self.builder.get_object('temp')
        self.top = self.builder.get_object('top')
        self.images = {}
        self.images_lock = threading.Lock()
        self.connect_signals()

    def connect_signals(self):
        print 'Destroy connect: %d' % self.window.connect('destroy',
                                                          self.destroyed)
        print 'Camera draw connect: %d' % self.camera.connect(
            'draw', self.update_camera
        )
        print 'Temp draw connect: %d' % self.temp.connect(
            'draw', self.update_temp
        )

    def destroyed(self, _):
        print('Destroyed')
        Gtk.main_quit()

    def update_images(self, images):
        with self.images_lock:
            self.images = images
        print('UPDATE IMAGES: DONE')
        self.camera.queue_draw()
        self.temp.queue_draw()

    def update_camera(self, _, context):
        print('UPDATE CAMERA: WAITING')
        with self.images_lock:
            print('UPDATE CAMERA: LOCKED')
            if 'camera' not in self.images:
                return False

            width = self.camera.get_allocated_width()
            height = self.camera.get_allocated_height()
            print 'Scaling'
            img = self.images['camera'].scale_simple(
                width, height, GdkPixbuf.InterpType.BILINEAR
            )
        print('UPDATE CAMERA: DONE')

        context.set_source_rgb(0, 0, 0)
        context.paint()
        print 'Painting'
        Gdk.cairo_set_source_pixbuf(context, img, 0, 0)
        context.paint()

        print 'Update done'
        return False

    def update_temp(self, _, context):
        print('UPDATE TEMP: WAITING')
        with self.images_lock:
            print('UPDATE TEMP: LOCKED')
            if 'temp' not in self.images:
                return False

            width = self.temp.get_allocated_width()
            height = self.temp.get_allocated_height()
            print 'Scaling'
            img = self.images['temp'].scale_simple(
                width, height, GdkPixbuf.InterpType.BILINEAR
            )
        print('UPDATE TEMP: DONE')

        context.set_source_rgb(0, 0, 0)
        context.paint()
        print 'Painting'
        Gdk.cairo_set_source_pixbuf(context, img, 0, 0)
        context.paint()

        print 'Update done'
        return False

    def run(self):
        Gtk.main()


if __name__ == '__main__':
    gui = GUI()
    # vision = vision.Vision('10.0.7.15', gui)
    vision = vision.Vision(
        gui,
        vision.StorageVisionProvider(
            '/home/jasper/dev/robocup_project/reference-images'
            '/full/converted-1m-front-straight.png'
        )
    )
    vision_thread = threading.Thread(target=vision.run)

    vision_thread.start()

    gui.run()
    vision.stop()
    vision_thread.join()
