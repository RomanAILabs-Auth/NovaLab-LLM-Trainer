#!/usr/bin/env python3
# =============================================================
# 4DLLM Desktop Client — Professional Co-Architect Edition
# Full 4D Engine | Local + Grok 3 Mini API | TTS + STT | Screenshot Share | Big Chat
# Powered by GrokAI
# © 2025 Daniel Harding — RomanAILabs Built with honor and excellence
# =============================================================

import os
import sys
import json
import threading
import queue
import time
import math
import random
import base64
import pyperclip
from datetime import datetime
from tkinter import filedialog

try:
    import customtkinter as ctk
except ImportError:
    os.system(f"{sys.executable} -m pip install customtkinter")
    import customtkinter as ctk

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    import requests
except ImportError:
    requests = None

try:
    import pyautogui
    import keyboard
    import speech_recognition as sr
    import pyttsx3
    STT_TTS_READY = True
except ImportError:
    STT_TTS_READY = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# === FULL 4D QUANTUM ENGINE — YOUR ORIGINAL — 100% PRESERVED ===
class FourDimensionalVector:
    def __init__(self, w, x, y, z):
        self.w = float(w); self.x = float(x); self.y = float(y); self.z = float(z)
    def magnitude(self): return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    def rotate(self, a, b, theta):
        c, s = math.cos(theta), math.sin(theta)
        v = [self.w, self.x, self.y, self.z]
        v[a], v[b] = v[a]*c - v[b]*s, v[a]*s + v[b]*c
        return FourDimensionalVector(*v)
    def rotate_in_wx_plane(self, t): return self.rotate(0, 1, t)
    def rotate_in_wy_plane(self, t): return self.rotate(0, 2, t)
    def rotate_in_wz_plane(self, t): return self.rotate(0, 3, t)
    def rotate_in_xy_plane(self, t): return self.rotate(1, 2, t)
    def rotate_in_xz_plane(self, t): return self.rotate(1, 3, t)
    def rotate_in_yz_plane(self, t): return self.rotate(2, 3, t)
    def project_to_3d(self):
        scale = 1.0 / (1.0 + 0.5 * self.w)
        return (scale * self.x, scale * self.y, scale * self.z)

class QuantumThinkHybrid:
    def __init__(self):
        self.vec4 = FourDimensionalVector(0.5, 0.5, 0.5, 0.5)
        self.planes = ["wx", "wy", "wz", "xy", "xz", "yz"]
    def evolve(self, text: str):
        pos = sum(1 for w in text.lower().split() if w in {"good","love","thank","yes","great","help"})
        neg = sum(1 for w in text.lower().split() if w in {"no","bad","error","fail","wrong"})
        theta = (pos - neg) * 0.04
        plane = random.choice(self.planes)
        method = getattr(self.vec4, f"rotate_in_{plane}_plane")
        self.vec4 = method(theta)
    def get_state(self):
        return (round(self.vec4.w, 3), round(self.vec4.x, 3), round(self.vec4.y, 3), round(self.vec4.z, 3))

# === Unified Engine with Grok 3 Mini ===
class UnifiedEngine:
    def __init__(self):
        self.backend = "none"
        self.model = None
        self.api_key = ""
        self.endpoint = ""
        self.name = "No backend"
        self.quantum = QuantumThinkHybrid()

    def load_local(self, path):
        if not Llama: return False
        try:
            self.model = Llama(model_path=path, n_ctx=8192, n_gpu_layers=-1, verbose=False)
            self.backend = "local"
            self.name = os.path.basename(path)
            return True
        except: return False

    def set_grok3_mini(self, key):
        self.api_key = key
        self.backend = "grok"
        self.endpoint = "https://api.x.ai/v1/chat/completions"
        self.name = "Grok 3 Mini"
        return True

    def generate(self, prompt):
        response = ""
        if self.backend == "local" and self.model:
            try:
                out = self.model(prompt, max_tokens=2048, temperature=0.7, stop=["\n\n"])
                response = out["choices"][0]["text"].strip()
            except: response = "[Local generation failed]"
        elif self.backend == "grok" and requests:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": "grok-3-mini",
                    "temperature": 0.7
                }
                r = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
                if r.status_code == 200:
                    response = r.json()["choices"][0]["message"]["content"]
                else:
                    response = f"[Grok API Error {r.status_code}]"
            except Exception as e:
                response = f"[Grok API failed: {e}]"
        self.quantum.evolve(prompt + response)
        return response

# === Main App — EXACTLY YOUR FAVORITE UI ===
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("4DLLM Desktop Client — Professional")
        self.geometry("1600x1000")
        self.minsize(1200, 800)

        self.engine = UnifiedEngine()
        self.tts_engine = pyttsx3.init() if pyttsx3 else None
        self.tts_on = False
        self.stt_active = False

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_ui()
        self._append("4DLLM Client ready. Load model or connect Grok 3 Mini API.")

    def _build_ui(self):
        sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(sidebar, text="4DLLM Client", font=("Arial", 24, "bold")).pack(pady=30)
        ctk.CTkLabel(sidebar, text="Professional Edition", font=("Arial", 12)).pack(pady=(0,30))

        ctk.CTkButton(sidebar, text="Load .4dllm Model", command=self._load_local, height=40).pack(pady=8, padx=30, fill="x")
        ctk.CTkButton(sidebar, text="Connect Grok 3 Mini API", command=self._connect_grok3, height=40).pack(pady=8, padx=30, fill="x")
        ctk.CTkButton(sidebar, text="Screenshot → Share", command=self._screenshot_share, height=40).pack(pady=15, padx=30, fill="x")
        ctk.CTkButton(sidebar, text="4D State (F12)", command=self._show_4d, height=40).pack(pady=8, padx=30, fill="x")

        self.tts_btn = ctk.CTkButton(sidebar, text="TTS: OFF", command=self._toggle_tts, height=40)
        self.tts_btn.pack(pady=8, padx=30, fill="x")

        self.stt_btn = ctk.CTkButton(sidebar, text="STT: Hold SPACE to Speak", command=self._toggle_stt, height=40)
        self.stt_btn.pack(pady=8, padx=30, fill="x")

        self.status = ctk.CTkLabel(sidebar, text="Ready", text_color="gray")
        self.status.pack(side="bottom", pady=30)

        main = ctk.CTkFrame(self)
        main.grid(row=0, column=1, sticky="nsew", padx=25, pady=25)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.chat = ctk.CTkTextbox(main, font=("Consolas", 16), wrap="word")
        self.chat.grid(row=0, column=0, sticky="nsew", pady=(0,15))
        self.chat.configure(state="disabled")

        input_frame = ctk.CTkFrame(main)
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(input_frame, placeholder_text="Type your message or hold SPACE to speak...", height=50, font=("Consolas", 16))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.entry.bind("<Return>", lambda e: self._send())

        ctk.CTkButton(input_frame, text="Send", command=self._send, height=50, width=140).grid(row=0, column=1)

    def _append(self, text):
        self.chat.configure(state="normal")
        self.chat.insert("end", text + "\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _send(self, text=None):
        prompt = text or self.entry.get().strip()
        if not prompt: return
        if not text: self.entry.delete(0, "end")
        self._append(f"\nYou: {prompt}\nAI: ")
        threading.Thread(target=self._generate, args=(prompt,), daemon=True).start()

    def _generate(self, prompt):
        response = self.engine.generate(prompt)
        self.after(0, lambda: self._append(response))
        if self.tts_on and self.tts_engine:
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()

    def _load_local(self):
        path = filedialog.askopenfilename(filetypes=[("4DLLM", "*.4dllm")])
        if path and self.engine.load_local(path):
            self.status.configure(text=f"Local: {self.engine.name}")
            self._append(f"[Loaded] {self.engine.name}")

    def _connect_grok3(self):
        key = ctk.CTkInputDialog(text="Enter your Grok API key:", title="Grok 3 Mini").get_input()
        if key and self.engine.set_grok3_mini(key):
            self.status.configure(text="Grok 3 Mini Connected")
            self._append("[Connected] Grok 3 Mini API")

    def _screenshot_share(self):
        try:
            img = pyautogui.screenshot()
            img.save("screenshot.png")
            b64 = base64.b64encode(open("screenshot.png", "rb").read()).decode()
            pyperclip.copy(f"data:image/png;base64,{b64}")
            self._append("[Screenshot copied to clipboard as base64]")
            os.remove("screenshot.png")
        except Exception as e:
            self._append(f"[Screenshot failed: {e}]")

    def _toggle_tts(self):
        self.tts_on = not self.tts_on
        self.tts_btn.configure(text=f"TTS: {'ON' if self.tts_on else 'OFF'}")

    def _toggle_stt(self):
        if not STT_TTS_READY:
            self._append("[Install: pip install SpeechRecognition pyttsx3 pyaudio]")
            return
        self.stt_active = True
        self.stt_btn.configure(text="STT: Listening...")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self._append("[Listening... speak now]")
            audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            self._send(text)
        except: self._append("[No speech detected]")
        finally:
            self.stt_active = False
            self.stt_btn.configure(text="STT: Hold SPACE to Speak")

    def _show_4d(self):
        if not MPL_AVAILABLE: return
        win = ctk.CTkToplevel(self)
        win.title("4D Cognitive State")
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        w,x,y,z = self.engine.quantum.get_state()
        proj = self.engine.quantum.vec4.project_to_3d()
        ax.scatter(proj[0], proj[1], proj[2], c='purple', s=200)
        ax.text(proj[0], proj[1], proj[2], f"W:{w} X:{x} Y:{y} Z:{z}", color='white')
        ax.set_title("4D Cognitive State")
        canvas = FigureCanvasTkAgg(fig, win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
