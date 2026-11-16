"""Small GLFW viewer with no MuJoCo Simulate keyboard shortcuts."""
from contextlib import nullcontext

import glfw
import mujoco


class ControlViewer:
    def __init__(self, model, data, key_callback):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.window = glfw.create_window(1000, 720, "Go2 | Arrow keys move | Space stop | Q quit", None, None)
        if not self.window:
            raise RuntimeError("Could not create control window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.model, self.data = model, data
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 135
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.last_mouse = None
        self.hud = ""

        def key(window, key, scancode, action, mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                key_callback(key)
        def scroll(window, dx, dy):
            self.cam.distance = max(.5, min(30, self.cam.distance * (1 - .1 * dy)))
        def cursor(window, x, y):
            if self.last_mouse is not None and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                dx, dy = x - self.last_mouse[0], y - self.last_mouse[1]
                self.cam.azimuth -= dx * .3
                self.cam.elevation = max(-89, min(0, self.cam.elevation - dy * .3))
            self.last_mouse = (x, y)
        glfw.set_key_callback(self.window, key)
        glfw.set_scroll_callback(self.window, scroll)
        glfw.set_cursor_pos_callback(self.window, cursor)

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.window is not None and not glfw.window_should_close(self.window)

    def sync(self):
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width and height:
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.context)
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                               "UP/DOWN: forward/back | LEFT/RIGHT: turn\nSpace: stop | Q: finish\nMouse drag: orbit | Wheel: zoom\n" + self.hud,
                               "", self.context)
            glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            self.scene = None
            glfw.destroy_window(self.window)
            self.window = None
