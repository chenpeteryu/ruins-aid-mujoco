"""Small GLFW viewer with no MuJoCo Simulate keyboard shortcuts."""
from contextlib import nullcontext

import glfw
import mujoco
import numpy as np
from PIL import Image


class ControlViewer:
    def __init__(self, model, data, key_callback):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        # MuJoCo's offscreen GLFW renderer leaves VISIBLE=0 as a global hint.
        # Reset those hints before creating the interactive control window.
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(1000, 720, "Go2 | Arrow keys move | Space stop | Q quit", None, None)
        if not self.window:
            raise RuntimeError("Could not create control window")
        glfw.show_window(self.window)
        glfw.focus_window(self.window)
        if not glfw.get_window_attrib(self.window, glfw.VISIBLE):
            raise RuntimeError("Control window is unexpectedly hidden")
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


class CameraViewer:
    """Display the exact RGB array used by the synchronized recorder."""
    def __init__(self, model, key_callback):
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self.window = glfw.create_window(640, 480, "Go2 front camera", None, None)
        if not self.window:
            raise RuntimeError("Could not create camera window")
        glfw.show_window(self.window)
        if not glfw.get_window_attrib(self.window, glfw.VISIBLE):
            raise RuntimeError("Camera window is unexpectedly hidden")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
        def key(window, key, scancode, action, mods):
            if action in (glfw.PRESS, glfw.REPEAT):
                key_callback(key)
        glfw.set_key_callback(self.window, key)

    def show(self, rgb, sim_time_ns):
        if self.window is None:
            return
        if glfw.window_should_close(self.window):
            self.close()
            return
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        if width and height:
            # Letterbox on resize; only display is scaled, recorded RGB is unchanged.
            scale = min(width / rgb.shape[1], height / rgb.shape[0])
            size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            image = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
            x, y = (width - size[0]) // 2, (height - size[1]) // 2
            canvas[y:y + size[1], x:x + size[0]] = image
            mujoco.mjr_drawPixels(np.ascontiguousarray(canvas[::-1]).reshape(-1), None, mujoco.MjrRect(0, 0, width, height), self.context)
            glfw.set_window_title(self.window, f"Go2 front camera | sim {sim_time_ns / 1e9:.2f}s | 10 Hz")
            glfw.swap_buffers(self.window)

    def close(self):
        if self.window is not None:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            self.window = None
