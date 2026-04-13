# ============================================================
# app.py — FINAL PRO PIPELINE
# (Includes: Smart Topology, Physics Fixes, Dev Mode, Multi-User Chat)
# ============================================================

import os
import re
import io
import json
import shutil
import tempfile
from pathlib import Path
from functools import lru_cache

import pandas as pd
import numpy as np
import torch as th
from PIL import Image

import drawSvg as drawsvg
import cairosvg
from tqdm import tqdm

# --- house_diffusion imports ---
from house_diffusion import dist_util
from house_diffusion.script_util import create_model_and_diffusion

# --- Gradio ---
import gradio as gr

# ============================================================
# CONFIGURATION
# ============================================================

# Best practice: export GEMINI_API_KEY="your_key" in terminal
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDUKyhoXaoBY7WwRs71rH3OrSDTvbQ8OJ0") 
GEMINI_AVAILABLE = False

try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("✅ Gemini configured.")
    else:
        print("⚠️ Gemini API key not found; using fallback parser.")
except Exception as e:
    print(f"⚠️ Gemini configuration failed: {e}")
    GEMINI_AVAILABLE = False

ROOM_CLASS = {
    'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4,
    'Balcony': 5, 'Entrance': 6, 'Dining Room': 7, 'Study Room': 8,
    'Storage': 10, 'Front Door': 11, 'Interior Door': 12, 'Unknown': 13
}

# Ensure model path is correct
MODEL_PATH = os.environ.get("MODEL_PATH", "ckpt/model250000.pt")

NUM_WORDS = {
    'zero': 0, 'one': 1, 'a': 1, 'an': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10
}

# Ensure output directories exist
Path("./outputs/pred").mkdir(parents=True, exist_ok=True)
Path("./generated_svgs").mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS & PHYSICS ENGINE
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def extract_count_from_phrase(phrase_tokens, idx):
    if idx > 0:
        prev = phrase_tokens[idx - 1]
        if prev.isdigit(): return int(prev)
        if prev.lower() in NUM_WORDS: return NUM_WORDS[prev.lower()]
    return 1

def resolve_collisions(polys_by_room, iterations=50, strength=0.2):
    """
    Physics engine: Pushes overlapping rooms apart.
    """
    # 1. Extract Centers and Sizes
    room_props = {}
    for rid, poly in polys_by_room.items():
        if not poly: continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        room_props[rid] = {
            "cx": (min_x + max_x) / 2, "cy": (min_y + max_y) / 2,
            "w": (max_x - min_x), "h": (max_y - min_y),
            "orig_poly": poly
        }

    # 2. Iterative Repulsion
    for _ in range(iterations):
        for r1, p1 in room_props.items():
            for r2, p2 in room_props.items():
                if r1 >= r2: continue # Avoid double checking

                dx = p1["cx"] - p2["cx"]
                dy = p1["cy"] - p2["cy"]
                
                min_dist_x = (p1["w"]/2 + p2["w"]/2) * 0.95 
                min_dist_y = (p1["h"]/2 + p2["h"]/2) * 0.95

                if abs(dx) < min_dist_x and abs(dy) < min_dist_y:
                    # Overlap detected! Push apart.
                    push_x = (min_dist_x - abs(dx)) * strength
                    push_y = (min_dist_y - abs(dy)) * strength

                    dir_x = 1 if dx > 0 else -1
                    dir_y = 1 if dy > 0 else -1

                    # Apply force (push along shallowest axis)
                    if push_x < push_y:
                        p1["cx"] += push_x * dir_x
                        p2["cx"] -= push_x * dir_x
                    else:
                        p1["cy"] += push_y * dir_y
                        p2["cy"] -= push_y * dir_y

    # 3. Reconstruct Polygons
    new_polys = {}
    for rid, props in room_props.items():
        xs = [p[0] for p in props["orig_poly"]]
        ys = [p[1] for p in props["orig_poly"]]
        orig_cx = (min(xs) + max(xs)) / 2
        orig_cy = (min(ys) + max(ys)) / 2
        
        shift_x = props["cx"] - orig_cx
        shift_y = props["cy"] - orig_cy
        new_polys[rid] = [(x + shift_x, y + shift_y) for x, y in props["orig_poly"]]
        
    return new_polys

# -----------------------------
# Smart Topology Optimizer
# -----------------------------
def optimize_connections(rooms, connections):
    """
    Rewires connections to prevent 'Living Room Gravity'.
    Converts STAR topology to TREE/CHAIN topology.
    """
    living_room_indices = [i for i, r in enumerate(rooms) if r['room_type'] == 'Living Room']
    dining_indices = [i for i, r in enumerate(rooms) if r['room_type'] == 'Dining Room']
    kitchen_indices = [i for i, r in enumerate(rooms) if r['room_type'] == 'Kitchen']
    bedroom_indices = [i for i, r in enumerate(rooms) if r['room_type'] == 'Bedroom']
    bathroom_indices = [i for i, r in enumerate(rooms) if r['room_type'] == 'Bathroom']
    
    if not living_room_indices:
        return connections

    main_hub = living_room_indices[0]
    new_connections = []
    
    def is_connected(r1, r2):
        for c in new_connections:
            if (c['room1'] == r1 and c['room2'] == r2) or (c['room1'] == r2 and c['room2'] == r1):
                return True
        return False

    # 1. Kitchen -> Dining -> Living
    for k_idx in kitchen_indices:
        if dining_indices:
            new_connections.append({"room1": k_idx, "room2": dining_indices[0]})
            if not is_connected(dining_indices[0], main_hub):
                new_connections.append({"room1": dining_indices[0], "room2": main_hub})
        else:
            new_connections.append({"room1": k_idx, "room2": main_hub})

    # 2. Bathroom -> Bedroom (Ensuite)
    available_beds = bedroom_indices.copy()
    for bath_idx in bathroom_indices:
        if available_beds:
            bed_target = available_beds.pop(0) 
            new_connections.append({"room1": bath_idx, "room2": bed_target})
        else:
            new_connections.append({"room1": bath_idx, "room2": main_hub})

    # 3. Bedroom Daisy Chain (LR -> Bed1 -> Bed2...)
    direct_slots = 2
    previous_bed = None
    for i, bed_idx in enumerate(bedroom_indices):
        if is_connected(bed_idx, main_hub): continue 

        if direct_slots > 0:
            new_connections.append({"room1": bed_idx, "room2": main_hub})
            direct_slots -= 1
            previous_bed = bed_idx
        else:
            if previous_bed is not None:
                new_connections.append({"room1": bed_idx, "room2": previous_bed})
            else:
                new_connections.append({"room1": bed_idx, "room2": main_hub})
            previous_bed = bed_idx

    # 4. Others (Study, Entrance, Balcony)
    for i, room in enumerate(rooms):
        is_linked = False
        for c in new_connections:
            if c['room1'] == i or c['room2'] == i: is_linked = True; break
        
        if not is_linked:
            if room['room_type'] == 'Balcony' and bedroom_indices:
                new_connections.append({"room1": i, "room2": bedroom_indices[-1]})
            else:
                new_connections.append({"room1": i, "room2": main_hub})

    return new_connections

# -----------------------------
# Model loader (cached)
# -----------------------------
_MODEL_CACHE = {}

def load_model_once(model_path=MODEL_PATH):
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["diffusion"], _MODEL_CACHE["device"]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please place 'model250000.pt' in the 'ckpt' folder.")

    # Standard Housediffusion Config
    args = {
        "input_channels": 18, "condition_channels": 89, "num_channels": 512,
        "out_channels": 2, "dataset": "rplan", "use_checkpoint": False,
        "use_unet": False, "learn_sigma": False, "diffusion_steps": 1000,
        "noise_schedule": "cosine", "timestep_respacing": "ddim100",
        "use_kl": False, "predict_xstart": False, "rescale_timesteps": False,
        "rescale_learned_sigmas": False, "analog_bit": False, "target_set": -1, "set_name": "",
    }

    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(**args) # Unpack args for brevity

    sd = dist_util.load_state_dict(model_path, map_location="cpu")
    model.load_state_dict(sd)
    device = dist_util.dev()
    model.to(device)
    model.eval()

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["diffusion"] = diffusion
    _MODEL_CACHE["device"] = device
    print(f"✅ Model loaded to {device}")
    return model, diffusion, device

# -----------------------------
# Geometry Logic
# -----------------------------
def function_test(org_graphs, corners, room_type):
    get_one_hot = lambda x, z: np.eye(z)[x]
    max_num_points = 100
    house = []
    corner_bounds = []
    num_points = 0

    for i, room in enumerate(room_type):
        num_room_corners = corners[i]
        rtype = np.repeat(np.array([get_one_hot(room, 25)]), num_room_corners, 0)
        room_index = np.repeat(np.array([get_one_hot(len(house) + 1, 32)]), num_room_corners, 0)
        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
        padding_mask = np.repeat(1, num_room_corners)
        padding_mask = np.expand_dims(padding_mask, 1)
        connections = np.array([[i, (i + 1) % num_room_corners] for i in range(num_room_corners)])
        connections += num_points
        corner_bounds.append([num_points, num_points + num_room_corners])
        num_points += num_room_corners
        room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index,
                               padding_mask, connections), 1)
        house.append(room)

    house_layouts = np.concatenate(house, 0)
    padding = np.zeros((max_num_points - len(house_layouts), 94))
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
    house_layouts = np.concatenate((house_layouts, padding), 0)

    door_mask = np.ones((max_num_points, max_num_points))
    self_mask = np.ones((max_num_points, max_num_points))

    living_room_index = 0
    for i, room in enumerate(room_type):
        if room == 1: 
            living_room_index = i
            break

    for i in range(len(corner_bounds)):
        is_connected = False
        for j in range(len(corner_bounds)):
            if i == j:
                self_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0
            elif any(np.equal([i, 1, j], org_graphs).all(1)) or any(np.equal([j, 1, i], org_graphs).all(1)):
                door_mask[corner_bounds[i][0]:corner_bounds[i][1], corner_bounds[j][0]:corner_bounds[j][1]] = 0
                is_connected = True
        if not is_connected and i != living_room_index:
            door_mask[corner_bounds[i][0]:corner_bounds[i][1],
                      corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

    syn_graph = np.concatenate((org_graphs, np.zeros([200 - len(org_graphs), 3])), 0)

    cond = {
        'syn_door_mask': door_mask, 'syn_self_mask': self_mask, 'syn_gen_mask': gen_mask,
        'syn_room_types': house_layouts[:, 2:2 + 25], 'syn_corner_indices': house_layouts[:, 2 + 25:2 + 57],
        'syn_room_indices': house_layouts[:, 2 + 57:2 + 89], 'syn_src_key_padding_mask': 1 - house_layouts[:, 2 + 89],
        'syn_connections': house_layouts[:, 2 + 90:2 + 92], 'syn_graph': syn_graph,
    }
    return cond

# -----------------------------
# save_samples (with Physics Fix)
# -----------------------------
def save_samples(sample, ext, model_kwargs, tmp_count, num_room_types,
                 save_gif=True, door_indices=[11, 12, 13], ID_COLOR=None,
                 is_syn=False, draw_graph=False, save_svg=False, metrics=False):
    
    prefix = 'syn_' if is_syn else ''
    
    def polygon_centroid(points):
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        return float(xs.mean()), float(ys.mean())

    def polygon_bbox(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def dominant_angle(points):
        max_len = 0; angle = 0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length > max_len:
                max_len = length
                angle = np.degrees(np.arctan2(dy, dx))
        return angle

    ensure_dir(f'outputs/{ext}')
    s_dim = sample.shape[1]

    for i in range(s_dim):
        resolution = 256
        draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
        draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))

        room_indices = model_kwargs.get(f'{prefix}room_indices')[0].cpu().numpy()
        room_idx_per_point = np.argmax(room_indices, axis=1) if room_indices.ndim == 2 else room_indices

        room_types = model_kwargs.get(f'{prefix}room_types')[0].cpu().numpy()
        room_types_per_point = np.argmax(room_types, axis=1) if room_types.ndim == 2 else np.zeros_like(room_idx_per_point)

        polys_by_room = {}
        types_by_room = {}
        P = sample.shape[2]
        
        for p_idx in range(P):
            if model_kwargs[f'{prefix}src_key_padding_mask'][0][p_idx] == 1: continue
            pt = sample[0, i, p_idx].cpu().numpy()
            pt = (pt / 2.0 + 0.5) * resolution
            
            room_idx = int(room_idx_per_point[p_idx])
            if room_idx not in polys_by_room:
                polys_by_room[room_idx] = []
                types_by_room[room_idx] = int(room_types_per_point[p_idx])
            polys_by_room[room_idx].append((float(pt[0]), float(pt[1])))

        # --- PHYSICS FIX: Push overlapping rooms apart ---
        if len(polys_by_room) > 4:
            polys_by_room = resolve_collisions(polys_by_room)
        # ------------------------------------------------

        # DRAW ROOMS
        for room_idx, poly in polys_by_room.items():
            room_type = types_by_room.get(room_idx, 0)
            if room_type in door_indices or room_type == 0: continue

            color = ID_COLOR.get(room_type, '#ffffff') if ID_COLOR else '#ffffff'
            draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=color, fill_opacity=1.0, stroke='black', stroke_width=1))

            # LABEL
            room_name = next((k for k, v in ROOM_CLASS.items() if v == room_type), "Room")
            cx, cy = polygon_centroid(poly)
            minx, miny, maxx, maxy = polygon_bbox(poly)
            font_size = max(6, min(maxx - minx, maxy - miny) * 0.18)
            rotation = dominant_angle(poly)

            draw_color.append(drawsvg.Text(room_name, font_size, cx, cy, fill='black', text_anchor='middle', alignment_baseline='middle', transform=f"rotate({rotation},{cx},{cy})"))

        # DRAW DOORS + METRICS
        for room_idx, poly in polys_by_room.items():
            room_type = types_by_room.get(room_idx, 0)
            if room_type not in door_indices: continue
            color = ID_COLOR.get(room_type, '#ffffff') if ID_COLOR else '#ffffff'
            draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=color, fill_opacity=1.0, stroke='black', stroke_width=1))

            if metrics and len(poly) >= 2:
                for j in range(len(poly)):
                    p1 = np.array(poly[j]); p2 = np.array(poly[(j + 1) % len(poly)])
                    length = np.linalg.norm(p1 - p2)
                    mx, my = (p1 + p2) / 2
                    draw_color.append(drawsvg.Text(f"{int(length)}", 5, mx, my, fill='black', text_anchor='middle', alignment_baseline='middle'))

        if save_svg: return draw_color
        else:
            out_png = f'outputs/{ext}/{tmp_count + i}c_{ext}.png'
            png_bytes = cairosvg.svg2png(draw_color.asSvg())
            Image.open(io.BytesIO(png_bytes)).save(out_png)
    return []

# -----------------------------
# Create Layout (Dev Params)
# -----------------------------
def create_layout(graphs, corners, room_type, metrics=False, ddim_steps=100, num_samples=1, seed=None, clip_denoised=True):
    model, diffusion, device = load_model_once()

    # --- DEV MODE: SEED CONTROL ---
    if seed is not None and seed != -1:
        seed = int(seed)
        th.manual_seed(seed)
        if th.cuda.is_available(): th.cuda.manual_seed_all(seed)
        print(f"🔧 Developer Mode: Using fixed seed {seed}")
    # ------------------------------

    ID_COLOR = {
        1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 
        7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171', 13: '#785A67', 12: '#D3A2C7'
    }

    model_kwargs = function_test(graphs, corners, room_type)
    for key in model_kwargs:
        arr = np.array([model_kwargs[key]])
        model_kwargs[key] = th.from_numpy(arr).to(device)

    sample_fn = diffusion.ddim_sample_loop
    png_paths = []
    svg_paths = []
    ensure_dir('outputs/pred')

    for count in range(num_samples):
        sample = sample_fn(
            model, th.Size([1, 2, 100]), clip_denoised=clip_denoised, 
            model_kwargs=model_kwargs, progress=True
        )
        sample = sample.permute([0, 1, 3, 2])

        pred = save_samples(sample, 'pred', model_kwargs, count, 14, ID_COLOR=ID_COLOR, is_syn=True, save_svg=True, metrics=metrics)

        temp_svg = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
        pred.saveSvg(temp_svg.name)
        
        png_path = f'./generated_svgs/output_{count}.png'
        Image.open(io.BytesIO(cairosvg.svg2png(pred.asSvg()))).save(png_path)
        
        persistent_svg = Path(f"./generated_svgs/output_{count}.svg")
        shutil.move(temp_svg.name, persistent_svg)

        svg_paths.append(str(persistent_svg))
        png_paths.append(png_path)

    return png_paths, svg_paths

# -----------------------------
# Parsers
# -----------------------------
def simple_prompt_parser(prompt, auto_add_entrance=False):
    room_keywords = {
        'living room': 'Living Room', 'kitchen': 'Kitchen', 'bedroom': 'Bedroom', 'bathroom': 'Bathroom',
        'entrance': 'Entrance', 'dining': 'Dining Room', 'study': 'Study Room', 'storage': 'Storage', 'balcony': 'Balcony'
    }
    result = {"rooms": [], "connections": []}
    tokens = re.findall(r'\w+|\d+', prompt.lower())
    room_counts = {}

    for i, token in enumerate(tokens):
        for key, room_type in room_keywords.items():
            if tokens[i:i + len(key.split())] == key.split():
                count = extract_count_from_phrase(tokens, i)
                room_counts[room_type] = room_counts.get(room_type, 0) + count

    for r_type, count in room_counts.items():
        for _ in range(count): result["rooms"].append({"room_type": r_type, "num_corners": 4})

    if auto_add_entrance and not any(r['room_type'] == 'Entrance' for r in result['rooms']):
        result["rooms"].append({"room_type": "Entrance", "num_corners": 4})
    if not any(r['room_type'] == 'Living Room' for r in result['rooms']):
        result["rooms"].append({"room_type": "Living Room", "num_corners": 4})

    # Basic Star Connection
    lr_idx = next((i for i, r in enumerate(result['rooms']) if r['room_type'] == 'Living Room'), None)
    if lr_idx is not None:
        for i, room in enumerate(result['rooms']):
            if i != lr_idx: result["connections"].append({"room1": lr_idx, "room2": i})
    
    return result

_GEMINI_CACHE = {}
def process_prompt_with_gemini(prompt, auto_add_entrance=True):
    if not GEMINI_AVAILABLE: return simple_prompt_parser(prompt, auto_add_entrance)
    
    key = (prompt, auto_add_entrance)
    if key in _GEMINI_CACHE: return _GEMINI_CACHE[key]

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        system_prompt = """
You are an expert architectural assistant specializing in converting natural language house descriptions into structured and logically consistent JSON layouts for architectural design purposes.

Your goal is to generate a clean, practical, and valid house layout JSON that strictly follows architectural best practices, Vastu principles (as soft guidelines), and real-world logic.

───────────────────────────────────────────────
🏗️ JSON OUTPUT FORMAT (STRICTLY ADHERE TO THIS):
{
  "rooms": [
    {"room_type": "Room Name", "num_corners": 4},
    ...
  ],
  "connections": [
    {"room1": 0, "room2": 1},
    ...
  ]
}

───────────────────────────────────────────────
✅ MANDATORY STRUCTURAL RULES:
1. Only the following room types are allowed:
    ["Living Room", "Kitchen", "Bedroom", "Bathroom", "Entrance", "Dining Room", "Study Room", "Storage", "Balcony"]
2. rooms must usually have:
    "num_corners": 4 (rectangular shape).
3. Always add balcony connected to the living room

───────────────────────────────────────────────
🧱 CONNECTION LOGIC (LOGICAL PRACTICALITY GUIDELINES):
- The "Living Room" acts as the central hub and should connect to:
    - "Kitchen", "Bedroom", "Study Room", "Balcony", "Dining Room","Entrance"
- "Bathroom" should preferably connect to a "Bedroom" if there is only 1 bedroom ,else connect it to livingroom.
-if there are same no of batrooms as of bedroom.then connect each bathroom to the bedroom.
-if there is less no of batroom then bedroom then connect one bathroom to the bedroom and the remaining to the bedroom
- "Kitchen" should connect to:
    - "Dining Room" or "Living Room", "Storage"
- "Storage" should connect to:
    - "Kitchen" 
- "Study Room" should connect to:
    - "Bedroom" or "Living Room"
- "Balcony" should connect to:
    - "Living Room" or "Bedroom"
- Strictly Avoid illogical connections, such as:
    - "Bathroom" directly connected to "Entrance"
    - Multiple "Bathrooms" connected only to "Living Room" without any "Bedroom"
- "Bathroom" should have only one connection i.e  to bedroom or livingroom:
─────────────────────────────────────────────
⚡ PRACTICAL RULE ENFORCEMENTS:
- If any "Bedroom" is present, ensure at least one "Bathroom" is added.
- Do not add extra rooms not mentioned in the description.
- Each room must be connected to at least one other room.
- Avoid creating illogical isolated rooms or circular connection loops.

───────────────────────────────────────────────
🚨 FAILURE HANDLING:
- If insufficient details are provided, generate the minimal valid structure:
    - Add "Bathroom" if "Bedroom" exists.
- Never produce invalid or incomplete JSON.
- Always keep consistency in room indices.

───────────────────────────────────────────────
📚 EXAMPLE

Input description:
"A simple 2 bedroom house with kitchen"

Expected output:
{
  "rooms": [
    {"room_type": "Living Room", "num_corners": 4},
    {"room_type": "Kitchen", "num_corners": 4},
    {"room_type": "Bedroom", "num_corners": 4},
    {"room_type": "Bedroom", "num_corners": 4},
    {"room_type": "Bathroom", "num_corners": 4}
  ],
  "connections": [
    {"room1": 0, "room2": 1},  // Living Room → Kitchen
    {"room1": 0, "room2": 2},  // Living Room → Bedroom
    {"room1": 0, "room2": 3},  // Living Room → Bedroom
    {"room1": 2, "room2": 4}   // Bedroom → Bathroom
  ]
}

───────────────────────────────────────────────
🎯 FINAL INSTRUCTION:
Return ONLY the structured JSON object as output. No extra text or explanations.  
Be logical, consistent, and always obey structural rules.  
Soft Vastu rules are preferences—adhere if possible without compromising practicality.

Now process this user request:
        """
        response = model.generate_content(system_prompt + "\n\n" + prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if auto_add_entrance and not any(r.get("room_type") == "Entrance" for r in result.get("rooms", [])):
                result["rooms"].append({"room_type": "Entrance", "num_corners": 4})
            _GEMINI_CACHE[key] = result
            return result
    except Exception as e:
        print(f"Gemini error: {e}")
    
    return simple_prompt_parser(prompt, auto_add_entrance)

# -----------------------------
# Main Pipeline
# -----------------------------
def generate_from_prompt(prompt, metrics, ddim_steps, num_samples, auto_add_entrance, seed, clip_denoised):
    try:
        # 1. Parse
        result = process_prompt_with_gemini(prompt, auto_add_entrance) if GEMINI_AVAILABLE else simple_prompt_parser(prompt, auto_add_entrance)
        if not result["rooms"]: raise gr.Error("No rooms identified.")

        # 2. OPTIMIZATION: Smart Rewiring for Large Houses
        if len(result["rooms"]) > 5:
            print("⚡ Large house detected: Optimizing graph topology...")
            result["connections"] = optimize_connections(result["rooms"], result["connections"])

        # 3. Graph Building
        room_list = []
        room_corners = []
        for room in result["rooms"]:
            rtype = room.get("room_type", "Unknown")
            room_list.append(ROOM_CLASS.get(rtype, ROOM_CLASS["Unknown"]))
            room_corners.append(room.get("num_corners", 4))

        edges = []
        existing_connections = set()
        for conn in result.get("connections", []):
            r1, r2 = conn["room1"], conn["room2"]
            if r1 >= len(result["rooms"]) or r2 >= len(result["rooms"]): continue
            if (r1, r2) in existing_connections or (r2, r1) in existing_connections: continue
            
            door_idx = len(room_list)
            room_list.append(12) # Interior Door
            room_corners.append(4)
            edges.extend([[r1, 1, door_idx], [r2, 1, door_idx]])
            existing_connections.add((r1, r2))

        # Add Front Door
        ent_indices = [i for i, r in enumerate(result['rooms']) if r['room_type'] == 'Entrance']
        if ent_indices:
            front_door_idx = len(room_list)
            room_list.append(11) 
            room_corners.append(4)
            edges.append([front_door_idx, 1, ent_indices[0]])

        # 4. Generate
        pngs, svgs = create_layout(edges, room_corners, room_list, metrics, ddim_steps, num_samples, seed, clip_denoised)

        # 5. Tables
        rooms_df = pd.DataFrame([{"id": i, "type": r["room_type"]} for i, r in enumerate(result["rooms"])])
        conn_df = pd.DataFrame([{"from": c["room1"], "to": c["room2"]} for c in result["connections"]])

        return pngs, (svgs[0] if svgs else None), rooms_df, conn_df, result

    except Exception as e:
        import traceback; traceback.print_exc()
        raise gr.Error(f"Error: {e}")

# ============================================================
# UI LOGIC
# ============================================================

def chatbot_handler(message, history, state):
    if state is None: state = {"base_prompt": None, "edits": []}
    
    if state["base_prompt"] is None:
        state["base_prompt"] = message
        reply = "Initial description recorded."
    else:
        state["edits"].append(message)
        reply = "Modification noted."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, "", state

def generate_wrapper(state, metrics, ddim_steps, num_samples, seed, clip):
    if not state or state["base_prompt"] is None: raise gr.Error("Please describe house first.")
    full_prompt = state["base_prompt"] + "\n" + "\n".join(state["edits"])
    return generate_from_prompt(full_prompt, metrics, ddim_steps, num_samples, True, seed, clip)

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🏠 House Plan Generator (Pro Version)")
        session_state = gr.State({"base_prompt": None, "edits": []})

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=350, type="messages")
                chat_input = gr.Textbox(placeholder="Describe house (e.g., 'Luxury 4BHK')...")
                send_btn = gr.Button("Send")
                reset_btn = gr.Button("Reset")

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Result", columns=2)
                svg_download = gr.File(label="SVG")

        with gr.Row():
            metrics = gr.Checkbox(label="Show Metrics", value=True)
            num_samples = gr.Slider(1, 4, value=1, step=1, label="Batch Size")
            gen_btn = gr.Button("Generate Layout", variant="primary")

        with gr.Accordion("🛠️ Developer Options", open=False):
            with gr.Row():
                seed_input = gr.Number(value=-1, label="Seed (-1=Random)", precision=0)
                ddim_input = gr.Slider(10, 1000, value=100, step=10, label="Steps")
                clip_input = gr.Checkbox(value=True, label="Clip Denoised")
            debug_json = gr.JSON(label="Debug Graph")

        with gr.Accordion("Data Tables", open=False):
            rooms_table = gr.DataFrame(label="Rooms")
            conn_table = gr.DataFrame(label="Connections")

        send_btn.click(chatbot_handler, [chat_input, chatbot, session_state], [chatbot, chat_input, session_state])
        reset_btn.click(lambda: ([], {"base_prompt": None, "edits": []}), None, [chatbot, session_state])
        gen_btn.click(generate_wrapper, [session_state, metrics, ddim_input, num_samples, seed_input, clip_input], 
                      [gallery, svg_download, rooms_table, conn_table, debug_json])

    return demo

if __name__ == "__main__":
    try: load_model_once()
    except Exception as e: print(f"Warning: {e}")
    build_ui().launch(share=True)