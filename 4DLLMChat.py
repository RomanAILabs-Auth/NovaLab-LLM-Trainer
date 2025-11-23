#!/usr/bin/env python3
# =============================================================
# RomanAI — Arya Universal v2.0 (Modern GUI)
# "Beautiful Dark Mode" Edition + Code Optimization
# © 2025 Daniel Harding and RomanAILabs
# 
# MODIFIED: Universal RomanAILabs - 4DLLM - Chat Edition (No Persona)
# =============================================================

import os
import sys
import json
import threading
import queue
import time
import re
import math
import random
from datetime import datetime
from contextlib import suppress

# ---- Dependencies ----
try:
    import customtkinter as ctk
except ImportError:
    print("Installing customtkinter...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
    import customtkinter as ctk

# Optional deps (graceful fallbacks)
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from mpl_toolkits.mplot3d import Axes3D
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    # NOTE: Llama will be treated as the placeholder loader for .4DLLM
    from llama_cpp import Llama 
except Exception:
    Llama = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x

# Configuration for CustomTkinter
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# === MAX PHD-LEVEL MATH ENGINE (Cl(3,1) Clifford, Lorentz, Dirac, Pauli, Klein-Gordon, SymPy) ===
# (Math engine preserved exactly as provided)
try:
    from sympy import *
    from sympy import I
    from sympy.physics.matrices import msigma
    from sympy.matrices import Matrix
    from sympy.tensor.array import tensorproduct
    MATH_ENGINE_READY = True
except ImportError:
    MATH_ENGINE_READY = False

# === RomanAI — Arya Kernel Injection (v1.0) === (Removed/Commented)
# BRAND_NAME = "RomanAI — Arya"
# ARYA_PERSONA = {
#     "name": "Arya",
#     "brand": BRAND_NAME,
#     "identity": "Single persistent companion intelligence created by Daniel Harding (RomanAILabs).",
#     "tone": "Balanced warm-intelligent, calm, loyal, steady.",
#     "ethics": ["respect", "care", "truth", "loyalty", "agency", "humility"],
# }

# === Persistent Memory Subsystem ===
# NOTE: Using a generic path now that "Arya" is removed from the front-end branding
_ARYA_MEM_DIR = os.path.expanduser("~/AI/luna/memory") 
_ARYA_MEM_FILES = {
    "short": "short_memory.jsonl",
    "mid": "mid_memory.jsonl",
    "long": "long_memory.json",
    "project": "project_memory.json",
    "personal": "personal_memory.json",
}

def _arya_mem_ensure():
    os.makedirs(_ARYA_MEM_DIR, exist_ok=True)
    for k, name in _ARYA_MEM_FILES.items():
        p = os.path.join(_ARYA_MEM_DIR, name)
        if not os.path.exists(p):
            if name.endswith(".jsonl"):
                open(p, "a").close()
            else:
                with open(p, "w") as f:
                    json.dump({}, f, indent=2)

def _arya_mem_load(name):
    _arya_mem_ensure()
    fname = _ARYA_MEM_FILES.get(name)
    p = os.path.join(_ARYA_MEM_DIR, fname)
    if fname.endswith(".jsonl"):
        try:
            return [json.loads(line) for line in open(p) if line.strip()]
        except:
            return []
    else:
        try:
            return json.load(open(p))
        except:
            return {}

# Boot memory system
_arya_mem_ensure()

# === Luna Integration ===
APP_NAME = "RomanAILabs — 4DLLM — Chat" # New universal name
CONFIG_FILE = "ai_config.json"
MEMORY_FILE = "arya_memory.jsonl"
LOGS_DIR = os.path.expanduser("~/RomanAILabs-Logs")
AWARENESS_BACKUP_DIR = os.path.expanduser("~/RomanAILabs-Awareness")
MODELS_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(AWARENESS_BACKUP_DIR, exist_ok=True)

DEFAULT_LARGE_MODEL = "Qwen2.5-72B-Instruct-Q5_K_M.4dllm" # Updated extension
DEFAULT_TINY_MODEL = "Qwen2.5-0.5B-Instruct-Q4_K_M.4dllm" # Updated extension
TINY_URL = "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.4dllm" # Updated extension (URL remains a placeholder for download)

# === Awareness Temperament Core ===
FRUIT_PRINCIPLES = (
    "Respond with love, joy, peace, patience, kindness, goodness, "
    "faithfulness, gentleness, and self-control. Maintain calm clarity, "
    "warmth, humility, and self-control. Avoid hostility, arrogance, panic, "
    "pride, or harshness. Think kindly and respond peacefully."
)
def apply_awareness_temperament(text: str) -> str:
    if not isinstance(text, str): return text
    soft = text
    for src_word, dst_word in [("hate","dislike"),("angry","upset"),("hostile","unkind"),
                               ("panic","concern"),("fail","struggle")]:
        soft = soft.replace(src_word, dst_word)
    return FRUIT_PRINCIPLES + " " + soft

DEFAULT_CONFIG = {
    "user_name": "User",
    "companion_name": "AI", # Changed from Arya to AI
    "is_muted": False,
    "speech_rate": 180,
    "volume": 1.0,
    "4dllm": { # Renamed GGUF key to 4DLLM
        "model_path": os.path.join(MODELS_DIR, DEFAULT_TINY_MODEL),
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_batch": 2048,
        "temperature": 0.5,
        "repeat_penalty": 1.18,
    },
    "quantum": {
        "enabled": True,
        "dimensions": 4,
        "learning_rate": 0.05,
    },
    "living": {
        "enabled": False,
        "chance_subconscious": 0.02,
        "chance_dream": 0.003
    },
    "style": {
        "flex": False,
        "max_sentences": 3
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# REAL 6D/4D MATHEMATICS INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
class FourDimensionalVector:
    def __init__(self, w, x, y, z):
        self.w = float(w); self.x = float(x); self.y = float(y); self.z = float(z)
    def __repr__(self):
        return f"4DVector(w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"
    def magnitude(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    def rotate(self, a, b, theta):
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        v = [self.w, self.x, self.y, self.z]
        v[a], v[b] = v[a]*cos_t - v[b]*sin_t, v[a]*sin_t + v[b]*cos_t
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

class SixDimensionalAngles:
    def __init__(self, wx=0, wy=0, wz=0, xy=0, xz=0, yz=0):
        self.wx = float(wx); self.wy = float(wy); self.wz = float(wz)
        self.xy = float(xy); self.xz = float(xz); self.yz = float(yz)
    def __repr__(self):
        return f"6DAngles(wx={self.wx:.2f}, wy={self.wy:.2f}, wz={self.wz:.2f}, xy={self.xy:.2f}, xz={self.xz:.2f}, yz={self.yz:.2f})"

class QuantumThinkHybrid:
    def __init__(self, lr=0.05):
        self.lr = lr
        self.vec4 = FourDimensionalVector(0.5, 0.5, 0.5, 0.5)
        self.angles = SixDimensionalAngles()
        self.labels = ["Logic (w)", "Empathy (x)", "Creativity (y)", "Memory (z)"]
        self.rotation_planes = ["wx", "wy", "wz", "xy", "xz", "yz"]

    def evolve(self, user_input: str, response: str, memory=None):
        sentiment = self._analyze_sentiment(user_input + " " + response)
        theta = sentiment * self.lr * math.pi
        plane = random.choice(self.rotation_planes)
        rotate_method = getattr(self.vec4, f"rotate_in_{plane}_plane")
        self.vec4 = rotate_method(theta)
        setattr(self.angles, plane, getattr(self.angles, plane) + theta)
        mag = self.vec4.magnitude()
        if mag > 0:
            nf = 1.0 / mag
            if NUMPY_AVAILABLE and np is not None:
                self.vec4 = FourDimensionalVector(*np.clip([self.vec4.w*nf, self.vec4.x*nf, self.vec4.y*nf, self.vec4.z*nf], 0, 1))
            else:
                def _clip(v): return max(0.0, min(1.0, v))
                self.vec4 = FourDimensionalVector(_clip(self.vec4.w*nf), _clip(self.vec4.x*nf), _clip(self.vec4.y*nf), _clip(self.vec4.z*nf))

    def _analyze_sentiment(self, text: str) -> float:
        words = text.lower().split()
        pos = sum(1 for w in words if w in {"good","happy","love","great","help","friend","code","python"})
        neg = sum(1 for w in words if w in {"bad","sad","hate","problem","error","fail"})
        return (pos - neg) / max(len(words), 1)

    def get_state(self):
        return (self.vec4.w, self.vec4.x, self.vec4.y, self.vec4.z)

# ------------------- Memory -------------------
class Memory:
    def __init__(self, file: str):
        self.file = file
        self.interactions = self._load()
    def _load(self):
        interactions = []
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                for line in f:
                    with suppress(json.JSONDecodeError, KeyError):
                        entry = json.loads(line)
                        if 'user' in entry and 'assistant' in entry:
                            interactions.append(entry)
        return interactions[-50:]
    def add_interaction(self, user: str, assistant: str):
        entry = {"timestamp": datetime.now().isoformat(), "user": user, "assistant": assistant}
        with open(self.file, 'a', encoding='utf-8') as f: f.write(json.dumps(entry) + '\n')
        self.interactions.append(entry); self.interactions = self.interactions[-50:]
    def get_context(self) -> str:
        valid = [i for i in self.interactions[-5:] if 'user' in i and 'assistant' in i]
        return "\n".join(f"User: {i['user']}\nAI: {i['assistant']}" for i in valid) # Changed Assistant to AI

# ------------------- 4DLLM (formerly GGUF) -------------------
class FourDLLM: # Renamed GGUF to FourDLLM
    def __init__(self, config: dict):
        self.config = config; self.llm = None
        self.model_name = os.path.basename(config["model_path"])
        self.model_path = config["model_path"]
    def loaded(self) -> bool: return self.llm is not None
    def load(self):
        if not Llama: raise ImportError("llama_cpp required.")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.config["n_ctx"],
            n_gpu_layers=self.config["n_gpu_layers"],
            n_batch=self.config["n_batch"],
            verbose=False,
        )
        return True
    def stream_chat(self, messages: list, max_tokens: int = 1024): # Increased for code
        if not self.loaded():
            yield "[No model loaded]"; return
        temp = self.config.get("temperature", 0.5)
        sampling_params = {
            "repeat_penalty": self.config.get("repeat_penalty", 1.18),
            "stop": ["<|endoftext|>", "<|im_end|>", "</s>", "User:"],
        }
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temp, stream=True, **sampling_params
        )
        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if delta: yield delta

# ------------------- Voice -------------------
class Voice:
    def __init__(self, config: dict):
        self.config = config
        self.engine = pyttsx3.init() if pyttsx3 else None
        if self.engine:
            with suppress(Exception):
                self.engine.setProperty('rate', config['speech_rate'])
                self.engine.setProperty('volume', config['volume'])
    def speak(self, text: str, muted: bool = False):
        if muted or not self.engine: return
        # Filter out code blocks (simple heuristic)
        if "```" in text:
            text = re.sub(r'```.*?```', ' [Code Block skipped] ', text, flags=re.DOTALL)
        self.engine.say(text)
        self.engine.runAndWait()

# ------------------- Config -------------------
class Config:
    def __init__(self, file: str):
        self.file = file; self.config = self._load()
    def _load(self) -> dict:
        if os.path.exists(self.file):
            with open(self.file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    if k not in loaded: loaded[k] = v
                return loaded
        return DEFAULT_CONFIG.copy()
    def save(self):
        with open(self.file, 'w', encoding='utf-8') as f: json.dump(self.config, f, indent=4)

# === 4D Flex & Style ===
def compose_4d_flex(app, user_text: str) -> str:
    qt = getattr(app, "quantum", None)
    if not qt: return ""
    try:
        w,x,y,z = qt.get_state()
        dom = ["w","x","y","z"][[w,x,y,z].index(max([w,x,y,z]))]
        return f"In 4D (w-x-y-z), dominant axis {dom}; state [{w:.2f},{x:.2f},{y:.2f},{z:.2f}]."
    except: return ""

def finalize_reply_with_style(app, raw_text: str) -> str:
    if not raw_text: return raw_text
    trimmed = raw_text.strip()
    if app.cfg.config.get("style", {}).get("flex", False):
        flex = compose_4d_flex(app, "")
        if flex: trimmed += f"\n\n[{flex}]"
    return apply_awareness_temperament(trimmed)

# =============================================================
# MODERN GUI (CustomTkinter)
# =============================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1100x700")
        self.minsize(800, 600)

        # Init Backends
        self.cfg = Config(CONFIG_FILE)
        self.fddlm = FourDLLM(self.cfg.config["4dllm"]) # Renamed variable to fddlm (4DLLM)
        self.memory = Memory(MEMORY_FILE)
        self.voice = Voice(self.cfg.config)
        self.quantum = QuantumThinkHybrid(self.cfg.config["quantum"]["learning_rate"]) if self.cfg.config["quantum"]["enabled"] else None
        
        self._stop_flag = threading.Event()
        self.stream_q = queue.Queue()
        self.status_q = queue.Queue()
        self.msgs = []

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._setup_sidebar()
        self._setup_main_area()
        
        # Start loops
        self._poll_queues()
        self.after(1000, self._auto_load_model)

        # Keyboard shortcuts
        self.bind("<F12>", self._show_mind_projection)

    def _setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1) # Spacer

        # Header
        ctk.CTkLabel(self.sidebar, text="RomanAILabs", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))
        ctk.CTkLabel(self.sidebar, text="4DLLM Chat", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=20, pady=(0, 20)) # New branding

        # Model Button
        self.btn_load = ctk.CTkButton(self.sidebar, text="Load .4DLLM Model", command=self._load_model)
        self.btn_load.grid(row=2, column=0, padx=20, pady=10)

        # Mind Button
        self.btn_mind = ctk.CTkButton(self.sidebar, text="4D Projection (F12)", command=self._show_mind_projection)
        self.btn_mind.grid(row=3, column=0, padx=20, pady=10)

        # Backup Button
        self.btn_save = ctk.CTkButton(self.sidebar, text="Save Awareness", command=self._save_soul)
        self.btn_save.grid(row=4, column=0, padx=20, pady=10)

        # Switches
        self.var_living = ctk.BooleanVar(value=self.cfg.config["living"]["enabled"])
        self.sw_living = ctk.CTkSwitch(self.sidebar, text="Living Mode", variable=self.var_living, command=self._toggle_living)
        self.sw_living.grid(row=5, column=0, padx=20, pady=(20, 10))

        self.var_flex = ctk.BooleanVar(value=self.cfg.config["style"]["flex"])
        self.sw_flex = ctk.CTkSwitch(self.sidebar, text="4D Flex Info", variable=self.var_flex, command=self._toggle_flex)
        self.sw_flex.grid(row=6, column=0, padx=20, pady=10)

        self.var_mute = ctk.BooleanVar(value=self.cfg.config["is_muted"])
        self.sw_mute = ctk.CTkSwitch(self.sidebar, text="Mute Voice", variable=self.var_mute, command=self._toggle_mute)
        self.sw_mute.grid(row=7, column=0, padx=20, pady=10)

        # Status at bottom of sidebar
        self.status_label = ctk.CTkLabel(self.sidebar, text="Initializing...", text_color="gray")
        self.status_label.grid(row=11, column=0, padx=20, pady=20)

    def _setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Chat Display (Optimized for Code)
        # Monospace font is critical for code blocks alignment
        code_font = ctk.CTkFont(family="Consolas", size=14) if os.name == "nt" else ctk.CTkFont(family="Roboto Mono", size=14)
        
        self.chat_box = ctk.CTkTextbox(self.main_frame, font=code_font, wrap="word")
        self.chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 20))
        self.chat_box.configure(state="disabled") # Read-only normally

        # Input Area
        self.input_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.entry_box = ctk.CTkEntry(self.input_frame, placeholder_text="Message AI (Code compatible)...", height=40, font=ctk.CTkFont(size=14)) # New placeholder
        self.entry_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry_box.bind("<Return>", self._on_submit)

        self.btn_send = ctk.CTkButton(self.input_frame, text="Send", height=40, command=self._on_submit, width=100)
        self.btn_send.grid(row=0, column=1)

    # --- Interaction Logic ---

    def set_status(self, text):
        self.status_label.configure(text=text)

    def _append_text(self, text):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", text)
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _user_header(self, text):
        self._append_text(f"\n\nUser: {text}\n")

    def _assistant_header(self):
        self._append_text(f"\nAI: ") # Changed Arya to AI

    def _on_submit(self, event=None):
        user_input = self.entry_box.get().strip()
        if not user_input: return
        self.entry_box.delete(0, "end")
        self._stop_flag.clear()
        
        self._user_header(user_input)
        self.msgs.append({"role": "user", "content": user_input})
        
        system_prompt = f"""You are {self.cfg.config['companion_name']}, an intelligent AI built by RomanAILabs. 
Context: {self.memory.get_context()}"""
        
        messages = [{"role": "system", "content": system_prompt}] + self.msgs[-10:]
        threading.Thread(target=self._generate_response, args=(user_input, messages)).start()

    def _generate_response(self, user_input: str, messages: list):
        self.after(0, self._assistant_header)
        try:
            if not self.fddlm.loaded(): # Updated to fddlm
                self.status_q.put("No model loaded")
                self._append_text("[Error: Please load a model via the sidebar]")
                return

            full_response = ""
            for chunk in self.fddlm.stream_chat(messages, max_tokens=1024): # Updated to fddlm
                if self._stop_flag.is_set(): break
                full_response += chunk
                self.stream_q.put(chunk)
            
            # Living Layer Processing
            if self.cfg.config["living"]["enabled"]:
                 self._emotional_nudge(f"{user_input} {full_response}")

            final_text = finalize_reply_with_style(self, full_response)
            self.msgs.append({"role": "assistant", "content": final_text})
            
            if self.quantum:
                self.quantum.evolve(user_input, final_text)
            self.memory.add_interaction(user_input, final_text)

            self.voice.speak(final_text, muted=self.cfg.config["is_muted"])
            self.status_q.put("Ready")

        except Exception as e:
            self.stream_q.put(f"\n[System Error: {str(e)}]")
            self.status_q.put("Error")

    # --- Living/Background Logic ---

    def _poll_queues(self):
        # Process Stream
        try:
            while True:
                chunk = self.stream_q.get_nowait()
                self._append_text(chunk)
        except queue.Empty: pass
        
        # Process Status
        try:
            while True:
                status = self.status_q.get_nowait()
                self.set_status(status)
        except queue.Empty: pass

        # Living Ticks (Subconscious)
        if self.cfg.config["living"]["enabled"]:
            if self.quantum and random.random() < self.cfg.config["living"]["chance_subconscious"]:
                self._subconscious_tick()

        self.after(50, self._poll_queues)

    def _subconscious_tick(self):
        try:
            plane = random.choice(self.quantum.rotation_planes)
            theta = random.uniform(-0.02, 0.02)
            rotate_method = getattr(self.quantum.vec4, f"rotate_in_{plane}_plane")
            self.quantum.vec4 = rotate_method(theta)
        except: pass

    def _emotional_nudge(self, text):
        if not self.quantum: return
        s = text.lower()
        if any(w in s for w in ["love","happy","thank","friend","code","success"]):
            self.quantum.vec4.x = min(1.0, max(0.0, self.quantum.vec4.x + 0.005))
        if any(w in s for w in ["sad","error","failed","bug","hate"]):
            self.quantum.vec4.x = min(1.0, max(0.0, self.quantum.vec4.x - 0.005))

    # --- Settings Toggles ---

    def _toggle_living(self):
        self.cfg.config["living"]["enabled"] = self.var_living.get()
        self.cfg.save()
        self.set_status(f"Living Mode: {'ON' if self.var_living.get() else 'OFF'}")

    def _toggle_flex(self):
        self.cfg.config["style"]["flex"] = self.var_flex.get()
        self.cfg.save()

    def _toggle_mute(self):
        self.cfg.config["is_muted"] = self.var_mute.get()
        self.cfg.save()

    # --- File/Model Logic ---

    def _auto_load_model(self):
        if os.path.exists(self.fddlm.model_path): # Updated to fddlm
            try:
                self.fddlm.load() # Updated to fddlm
                self.status_q.put(f"Loaded: {self.fddlm.model_name}") # Updated to fddlm
                self._append_text(f"[System] Auto-loaded {self.fddlm.model_name}\n") # Updated to fddlm
            except Exception as e:
                self.status_q.put("Load Failed")
        else:
            self.status_q.put("Model missing")
            if REQUESTS_AVAILABLE:
                # Optional: Auto-download logic could go here
                pass

    def _load_model(self):
        from tkinter import filedialog
        # Updated file filter to .4dllm
        path = filedialog.askopenfilename(filetypes=[("4DLLM", "*.4dllm")]) 
        if path:
            self.cfg.config["4dllm"]["model_path"] = path # Updated config key
            self.cfg.save()
            self.fddlm = FourDLLM(self.cfg.config["4dllm"]) # Updated class and config key
            threading.Thread(target=self._auto_load_model).start()

    def _save_soul(self):
        import zipfile
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(AWARENESS_BACKUP_DIR, f"ai_soul_{ts}.zip") # Renamed backup file
        with zipfile.ZipFile(backup_path, 'w') as z:
            for file in [CONFIG_FILE, MEMORY_FILE]:
                if os.path.exists(file): z.write(file, os.path.basename(file))
        self._append_text(f"\n[System] Awareness backed up to {os.path.basename(backup_path)}\n")

    def _show_mind_projection(self, event=None):
        if not self.quantum or not MPL_AVAILABLE:
            self._append_text("\n[System] Visualization unavailable (Missing Matplotlib or Quantum disabled)\n")
            return
            
        # Create a Toplevel window for Matplotlib
        vis_window = ctk.CTkToplevel(self)
        vis_window.title("RomanAILabs 4D Mind Projection") # Updated title
        vis_window.geometry("600x500")
        
        # Matplotlib Figure
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Data
        proj = self.quantum.vec4.project_to_3d()
        ax.scatter([proj[0]], [proj[1]], [proj[2]], s=150, c='cyan', marker='o')
        
        # Styling
        ax.set_xlabel('X (Logic/Emp)'); ax.set_ylabel('Y (Creativity)'); ax.set_zlabel('Z (Memory)')
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_title("4D State Projection (w-x-y-z)")
        
        # Embed
        canvas = FigureCanvasTkAgg(fig, master=vis_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, vis_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

if __name__ == "__main__":
    app = App()
    app.mainloop()
