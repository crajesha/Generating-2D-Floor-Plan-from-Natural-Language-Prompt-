# ============================================================
# app.py — FINAL CORRECT PIPELINE WITH CHATBOT (FIXED)
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

# --- house_diffusion imports (UNCHANGED) ---
from house_diffusion import dist_util
from house_diffusion.script_util import create_model_and_diffusion

# --- Gradio ---
import gradio as gr

# ============================================================
# CONFIGURATION (UNCHANGED)
# ============================================================

GEMINI_API_KEY = "AIzaSyDUKyhoXaoBY7WwRs71rH3OrSDTvbQ8OJ0"
GEMINI_AVAILABLE = False

try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print("✅ Gemini configured.")
    else:
        print("⚠️ Gemini API key not set; using fallback parser.")
except Exception as e:
    print(f"⚠️ Gemini configuration failed: {e}")
    GEMINI_AVAILABLE = False

ROOM_CLASS = {
    'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4,
    'Balcony': 5, 'Entrance': 6, 'Dining Room': 7, 'Study Room': 8,
    'Storage': 10, 'Front Door': 11, 'Interior Door': 12, 'Unknown': 13
}

MODEL_PATH = os.environ.get("MODEL_PATH", "ckpt/model250000.pt")

NUM_WORDS = {
    'zero': 0, 'one': 1, 'a': 1, 'an': 1, 'two': 2, 'three': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10
}

Path("./outputs/pred").mkdir(parents=True, exist_ok=True)
Path("./generated_svgs").mkdir(parents=True, exist_ok=True)

# ============================================================
# CHATBOT STATE (TEXT ONLY — SAFE)
# ============================================================

CHAT_STATE = {
    "base_prompt": None,
    "edits": []
}

def reset_chat():
    CHAT_STATE["base_prompt"] = None
    CHAT_STATE["edits"].clear()

def build_full_prompt():
    """
    Build ONE complete description string.
    This is the only thing passed to the generator.
    """
    if CHAT_STATE["base_prompt"] is None:
        return None
    prompt = CHAT_STATE["base_prompt"]
    for edit in CHAT_STATE["edits"]:
        prompt += "\n" + edit
    return prompt

# ============================================================
# ALL ORIGINAL FUNCTIONS BELOW ARE UNCHANGED
# (model loader, geometry, save_samples, create_layout,
#  simple_prompt_parser, process_prompt_with_gemini,
#  generate_from_prompt)
# ============================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_drawsvg_as_files(draw_obj, out_svg_path, out_png_path):
    out_svg_path = Path(out_svg_path)
    out_svg_path.parent.mkdir(parents=True, exist_ok=True)
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    # save svg
    draw_obj.saveSvg(str(out_svg_path))
    # save png
    png_bytes = cairosvg.svg2png(draw_obj.asSvg())
    Image.open(io.BytesIO(png_bytes)).save(str(out_png_path))

def extract_count_from_phrase(phrase_tokens, idx):
    """Return numeric count when preceding token is number/word, else default 1."""
    if idx > 0:
        prev = phrase_tokens[idx - 1]
        if prev.isdigit():
            return int(prev)
        if prev.lower() in NUM_WORDS:
            return NUM_WORDS[prev.lower()]
    return 1

# -----------------------------
# Model loader (cached)
# -----------------------------
_MODEL_CACHE = {}

def load_model_once(model_path=MODEL_PATH):
    """
    Load diffusion model + diffusion object once and cache them.
    Raises informative error if checkpoint missing.
    """
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["diffusion"], _MODEL_CACHE["device"]

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Set MODEL_PATH env var or place checkpoint there.")

    args = {
        "input_channels": 18,
        "condition_channels": 89,
        "num_channels": 512,
        "out_channels": 2,
        "dataset": "rplan",
        "use_checkpoint": False,
        "use_unet": False,
        "learn_sigma": False,
        "diffusion_steps": 1000,
        "noise_schedule": "cosine",
        "timestep_respacing": "ddim100",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "analog_bit": False,
        "target_set": -1,
        "set_name": "",
    }

    dist_util.setup_dist()
    model, diffusion = create_model_and_diffusion(
        args['input_channels'],
        args['condition_channels'],
        args['num_channels'],
        args['out_channels'],
        args['dataset'],
        args['use_checkpoint'],
        args['use_unet'],
        args['learn_sigma'],
        args['diffusion_steps'],
        args['noise_schedule'],
        args['timestep_respacing'],
        args['use_kl'],
        args['predict_xstart'],
        args['rescale_timesteps'],
        args['rescale_learned_sigmas'],
        args['analog_bit'],
        args['target_set'],
        args['set_name'],
    )

    # load checkpoint
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
# Geometry / conditioning helpers (kept mostly same)
# -----------------------------
def function_test(org_graphs, corners, room_type):
    # identical structure but with safer numpy operations and comments
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
        if room == 1:  # Living Room ID
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
        'syn_door_mask': door_mask,
        'syn_self_mask': self_mask,
        'syn_gen_mask': gen_mask,
        'syn_room_types': house_layouts[:, 2:2 + 25],
        'syn_corner_indices': house_layouts[:, 2 + 25:2 + 57],
        'syn_room_indices': house_layouts[:, 2 + 57:2 + 89],
        'syn_src_key_padding_mask': 1 - house_layouts[:, 2 + 89],
        'syn_connections': house_layouts[:, 2 + 90:2 + 92],
        'syn_graph': syn_graph,
    }

    return cond

# -----------------------------
# save_samples (refactored & fixed)
# -----------------------------
def save_samples(
        sample, ext, model_kwargs,
        tmp_count, num_room_types,
        save_gif=True,
        door_indices=[11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False, metrics=False):
    """
    save_samples with:
    - Original polygon rendering
    - Door rendering
    - Measurement labels (unchanged)
    - NEW: Auto-fitting, auto-rotating room labels
    """

    prefix = 'syn_' if is_syn else ''
    graph_errors = []

    def polygon_centroid(points):
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        return float(xs.mean()), float(ys.mean())

    def polygon_bbox(points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def dominant_angle(points):
        max_len = 0
        angle = 0
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
        resolution = 1024
        draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
        draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))

        room_indices = model_kwargs.get(f'{prefix}room_indices')[0].cpu().numpy()
        if room_indices.ndim == 2:
            room_idx_per_point = np.argmax(room_indices, axis=1)
        else:
            room_idx_per_point = room_indices

        room_types = model_kwargs.get(f'{prefix}room_types')[0].cpu().numpy()
        if room_types.ndim == 2:
            room_types_per_point = np.argmax(room_types, axis=1)
        else:
            room_types_per_point = np.zeros_like(room_idx_per_point)

        polys_by_room = {}
        types_by_room = {}

        P = sample.shape[2]
        for p_idx in range(P):
            if model_kwargs[f'{prefix}src_key_padding_mask'][0][p_idx] == 1:
                continue

            pt = sample[0, i, p_idx].cpu().numpy()
            pt = (pt / 2.0 + 0.5) * resolution

            room_idx = int(room_idx_per_point[p_idx])
            if room_idx not in polys_by_room:
                polys_by_room[room_idx] = []
                types_by_room[room_idx] = int(room_types_per_point[p_idx])

            polys_by_room[room_idx].append((float(pt[0]), float(pt[1])))

        # -------------------------
        # DRAW ROOMS + LABELS
        # -------------------------
        for room_idx, poly in polys_by_room.items():
            room_type = types_by_room.get(room_idx, 0)

            if room_type in door_indices or room_type == 0:
                continue

            color = ID_COLOR.get(room_type, '#ffffff') if ID_COLOR else '#ffffff'

            draw_color.append(
                drawsvg.Lines(
                    *np.array(poly).flatten().tolist(),
                    close=True,
                    fill=color,
                    fill_opacity=1.0,
                    stroke='black',
                    stroke_width=1
                )
            )

            # -------- TEXT LABEL --------
            room_name = next(
                (k for k, v in ROOM_CLASS.items() if v == room_type),
                "Room"
            )

            cx, cy = polygon_centroid(poly)
            minx, miny, maxx, maxy = polygon_bbox(poly)

            width = maxx - minx
            height = maxy - miny

            font_size = max(6, min(width, height) * 0.18)
            rotation = dominant_angle(poly)

            draw_color.append(
                drawsvg.Text(
                    room_name,
                    font_size,
                    cx,
                    cy,
                    fill='black',
                    text_anchor='middle',
                    alignment_baseline='middle',
                    transform=f"rotate({rotation},{cx},{cy})"
                )
            )

        # -------------------------
        # DRAW DOORS + METRICS
        # -------------------------
        for room_idx, poly in polys_by_room.items():
            room_type = types_by_room.get(room_idx, 0)
            if room_type not in door_indices:
                continue

            color = ID_COLOR.get(room_type, '#ffffff') if ID_COLOR else '#ffffff'

            draw_color.append(
                drawsvg.Lines(
                    *np.array(poly).flatten().tolist(),
                    close=True,
                    fill=color,
                    fill_opacity=1.0,
                    stroke='black',
                    stroke_width=1
                )
            )

            if metrics and len(poly) >= 2:
                for j in range(len(poly)):
                    p1 = np.array(poly[j])
                    p2 = np.array(poly[(j + 1) % len(poly)])
                    length = np.linalg.norm(p1 - p2)

                    mx, my = (p1 + p2) / 2
                    draw_color.append(
                        drawsvg.Text(
                            f"{int(length)}",
                            5,
                            mx,
                            my,
                            fill='black',
                            text_anchor='middle',
                            alignment_baseline='middle'
                        )
                    )

        if save_svg:
            return draw_color
        else:
            out_png = f'outputs/{ext}/{tmp_count + i}c_{ext}.png'
            png_bytes = cairosvg.svg2png(draw_color.asSvg())
            Image.open(io.BytesIO(png_bytes)).save(out_png)

    return graph_errors

# -----------------------------
# Layout create function (uses cached loader)
# -----------------------------
def create_layout(graphs, corners, room_type, metrics=False, ddim_steps=100, num_samples=1):
    """
    Uses cached model to generate sample(s), then uses save_samples to produce SVG/PNG outputs.
    Returns lists of PNG and SVG paths.
    """
    # validate checkpoint availability & load model
    model, diffusion, device = load_model_once()

    ID_COLOR = {
        1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE',
        5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69',
        10: '#1F849B', 11: '#727171', 13: '#785A67', 12: '#D3A2C7'
    }

    model_kwargs = function_test(graphs, corners, room_type)
    # convert to torch tensors and move to device
    for key in model_kwargs:
        arr = np.array([model_kwargs[key]])
        model_kwargs[key] = th.from_numpy(arr).to(device)

    sample_fn = diffusion.ddim_sample_loop

    png_paths = []
    svg_paths = []

    # ensure output dirs exist
    output_dir = Path("./generated_svgs")
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir('outputs/pred')

    for count in range(num_samples):
        # create sample
        sample = sample_fn(
            model,
            th.Size([1, 2, 100]),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
        # permute to shape [B, S, P, 2] for our save_samples
        sample = sample.permute([0, 1, 3, 2])

        pred = save_samples(sample, 'pred', model_kwargs, count, 14,
                            ID_COLOR=ID_COLOR, is_syn=True, save_svg=True, metrics=metrics)

        # Save SVG and PNG
        temp_svg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
        pred.saveSvg(temp_svg_file.name)

        png_file_path = f'./generated_svgs/output_{count}.png'
        Image.open(io.BytesIO(cairosvg.svg2png(pred.asSvg()))).save(png_file_path)

        persistent_svg_path = Path(f"./generated_svgs/output_{count}.svg")
        shutil.move(temp_svg_file.name, persistent_svg_path)

        svg_paths.append(str(persistent_svg_path))
        png_paths.append(png_file_path)

    return png_paths, svg_paths

# -----------------------------
# Parsers: improved simple parser + Gemini integration
# -----------------------------
def simple_prompt_parser(prompt, auto_add_entrance=False):
    """Improved simple parser:
    - Recognizes counts in numeric or word form (e.g., "2 bedrooms", "two bedrooms").
    - Adds Entrance only when auto_add_entrance True.
    """
    room_keywords = {
        'living room': 'Living Room',
        'kitchen': 'Kitchen',
        'bedroom': 'Bedroom',
        'bathroom': 'Bathroom',
        'bath': 'Bathroom',
        'entrance': 'Entrance',
        'entry': 'Entrance',
        'front door': 'Entrance',
        'dining room': 'Dining Room',
        'dining': 'Dining Room',
        'study room': 'Study Room',
        'study': 'Study Room',
        'storage': 'Storage',
        'balcony': 'Balcony'
    }

    result = {"rooms": [], "connections": []}
    prompt_lower = prompt.lower()
    tokens = re.findall(r'\w+|\d+', prompt_lower)

    # Count rooms: handle "2 bedroom", "two bedroom"
    room_counts = {}
    for i, token in enumerate(tokens):
        for key, room_type in room_keywords.items():
            # match multi-word keys as sequence - simple approach
            key_tokens = key.split()
            if tokens[i:i + len(key_tokens)] == key_tokens:
                count = extract_count_from_phrase(tokens, i)
                room_counts[room_type] = room_counts.get(room_type, 0) + count

    # Add rooms based on counts
    for room_type, count in room_counts.items():
        for _ in range(count):
            result["rooms"].append({"room_type": room_type, "num_corners": 4})

    # Auto add entrance/living room rules
    if auto_add_entrance and not any(room['room_type'] == 'Entrance' for room in result['rooms']):
        result["rooms"].append({"room_type": "Entrance", "num_corners": 4})
    if not any(room['room_type'] == 'Living Room' for room in result['rooms']):
        result["rooms"].append({"room_type": "Living Room", "num_corners": 4})

    # Build connections heuristically:
    # Map indexes and connect Living Room as hub
    living_room_idx = next((i for i, r in enumerate(result['rooms']) if r['room_type'] == 'Living Room'), None)
    entrance_idx = next((i for i, r in enumerate(result['rooms']) if r['room_type'] == 'Entrance'), None)

    if entrance_idx is not None and living_room_idx is not None:
        result["connections"].append({"room1": entrance_idx, "room2": living_room_idx})

    if living_room_idx is not None:
        for i, room in enumerate(result['rooms']):
            if room['room_type'] not in ['Living Room', 'Entrance']:
                result["connections"].append({"room1": living_room_idx, "room2": i})

    # If bedrooms exist but no bathroom, add one
    if any(r['room_type'] == 'Bedroom' for r in result['rooms']) and not any(r['room_type'] == 'Bathroom' for r in result['rooms']):
        result['rooms'].append({"room_type": "Bathroom", "num_corners": 4})
        # connect the bathroom to first bedroom
        b_idx = next((i for i, r in enumerate(result['rooms']) if r['room_type'] == 'Bedroom'), None)
        if b_idx is not None:
            result['connections'].append({"room1": b_idx, "room2": len(result['rooms']) - 1})

    return result

# Cache for Gemini responses
_GEMINI_CACHE = {}

def process_prompt_with_gemini(prompt, auto_add_entrance=True):
    """
    Use Gemini when available. A simple cache prevents repeated identical calls.
    Falls back to simple_prompt_parser on failure.
    """
    if not GEMINI_AVAILABLE:
        return simple_prompt_parser(prompt, auto_add_entrance=auto_add_entrance)

    # cache lookup
    key = (prompt, auto_add_entrance)
    if key in _GEMINI_CACHE:
        return _GEMINI_CACHE[key]

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
4.Dont add the entrance in the output at any scenario

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
        # extract JSON
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # postprocess to ensure indices consistent
            # optionally add entrance if requested and missing
            if auto_add_entrance and not any(room.get("room_type") == "Entrance" for room in result.get("rooms", [])):
                result["rooms"].append({"room_type": "Entrance", "num_corners": 4})
            # ensure connectivity and structural rules minimally apply
            _GEMINI_CACHE[key] = result
            return result
        else:
            # fallback
            return simple_prompt_parser(prompt, auto_add_entrance=auto_add_entrance)
    except Exception as e:
        print(f"Gemini error (falling back): {e}")
        return simple_prompt_parser(prompt, auto_add_entrance=auto_add_entrance)

# -----------------------------
# Main generator (Gradio handler)
# -----------------------------
def generate_from_prompt(prompt, metrics, ddim_steps, num_samples, auto_add_entrance):
    """Generate layout from text prompt and return PNGs, SVG download, color guide, tables."""
    try:
        # Step 1: parse with Gemini if available, else fallback
        result = process_prompt_with_gemini(prompt, auto_add_entrance=auto_add_entrance) if GEMINI_AVAILABLE else simple_prompt_parser(prompt, auto_add_entrance=auto_add_entrance)

        if not result["rooms"]:
            raise gr.Error("Could not identify any rooms in your description.")

        # Prepare room lists & corners
        room_list = []
        room_corners = []
        for room in result["rooms"]:
            rtype = room.get("room_type", "Unknown")
            room_list.append(ROOM_CLASS.get(rtype, ROOM_CLASS["Unknown"]))
            room_corners.append(room.get("num_corners", 4))

        edges = []
        existing_connections = set()

        # Build interior door nodes for connections
        for conn in result.get("connections", []):
            room1 = conn["room1"]
            room2 = conn["room2"]
            if room1 >= len(result["rooms"]) or room2 >= len(result["rooms"]):
                print(f"Warning: skipping invalid connection {room1} - {room2}")
                continue
            # Avoid duplicates (unordered)
            if (room1, room2) in existing_connections or (room2, room1) in existing_connections:
                continue
            door_idx = len(room_list)
            room_list.append(12)  # Interior door
            room_corners.append(4)
            edges.append([room1, 1, door_idx])  # room1 -> door
            edges.append([room2, 1, door_idx])  # room2 -> door
            existing_connections.add((room1, room2))

        # Add front door if Entrance exists
        entrance_indices = [i for i, room in enumerate(result['rooms']) if room['room_type'] == 'Entrance']
        if entrance_indices:
            entrance_idx = entrance_indices[0]
            front_door_idx = len(room_list)
            room_list.append(11)  # Front door
            room_corners.append(4)
            edges.append([front_door_idx, 1, entrance_idx])

        if np.sum(room_corners) > 99:
            raise gr.Error("Too many corners in the layout. Please simplify your design.")

        # Call layout generator
        png_paths, svg_paths = create_layout(edges, room_corners, room_list,
                                            metrics=metrics, ddim_steps=ddim_steps, num_samples=num_samples)

        png_color_guide = './color_guide.png' if os.path.exists('./color_guide.png') else None

        # Create summary tables
        rooms_summary = [{"id": i, "type": room["room_type"], "corners": room["num_corners"]} for i, room in enumerate(result["rooms"])]

        connections_summary = []
        for i, conn in enumerate(result.get("connections", [])):
            if conn["room1"] < len(result["rooms"]) and conn["room2"] < len(result["rooms"]):
                room1_type = result["rooms"][conn["room1"]]["room_type"]
                room2_type = result["rooms"][conn["room2"]]["room_type"]
                connections_summary.append({
                    "id": i,
                    "from": f"{conn['room1']} {room1_type}",
                    "to": f"{conn['room2']} {room2_type}"
                })

        rooms_df = pd.DataFrame(rooms_summary)
        connections_df = pd.DataFrame(connections_summary)

        # Gradio gallery expects list of images; return png paths
        return png_paths, (svg_paths[0] if svg_paths else None), png_color_guide, rooms_df, connections_df

    except Exception as e:
        import traceback
        print(f"Error details: {e}")
        print(traceback.format_exc())
        raise gr.Error(f"Error generating layout: {e}")

# ============================================================
# CHATBOT HANDLER (NEW)
# ============================================================

def chatbot_handler(message, history):
    if CHAT_STATE["base_prompt"] is None:
        CHAT_STATE["base_prompt"] = message
        reply = "Initial house description recorded. You can now refine it."
    else:
        CHAT_STATE["edits"].append(message)
        reply = "Modification added to the design."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""

# ============================================================
# CHATBOT → GENERATOR WRAPPER (NEW, FIXED)
# ============================================================

def generate_from_chatbot(metrics, ddim_steps, num_samples):
    """
    Wrapper that reuses your ORIGINAL generator but returns only 4 outputs
    to match Gradio's expected outputs
    """
    full_prompt = build_full_prompt()
    if not full_prompt:
        raise gr.Error("Please describe the house using the chatbot first.")

    # Get all 5 outputs from generate_from_prompt
    png_paths, svg_file, color_guide, rooms_df, connections_df = generate_from_prompt(
        full_prompt,
        metrics,
        ddim_steps,
        num_samples,
        auto_add_entrance=False
    )
    
    # Return only the 4 outputs that Gradio expects
    # We're dropping the color_guide output since it's not in the UI
    return png_paths, svg_file, rooms_df, connections_df

# ============================================================
# GRADIO UI (MODIFIED SAFELY)
# ============================================================

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🏠 House Plan Generator with Chatbot")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    height=350,
                    type="messages",
                    allow_tags=False
                )
                chat_input = gr.Textbox(
                    placeholder="Describe or modify your house (e.g., 2BHK, add balcony)"
                )
                send_btn = gr.Button("Send")
                reset_btn = gr.Button("Reset Design")

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated Layouts", columns=2)
                svg_download = gr.File(label="Download SVG")

        with gr.Row():
            metrics = gr.Checkbox(label="Show measurements", value=True)
            ddim_steps = gr.Slider(1, 1000, value=100, label="DDIM Steps")
            num_samples = gr.Slider(1, 4, value=1, step=1, label="Number of layouts")

        generate_btn = gr.Button("Generate Layout", variant="primary")

        with gr.Accordion("Layout Details", open=True):
            rooms_table = gr.DataFrame(label="Rooms")
            connections_table = gr.DataFrame(label="Connections")

        send_btn.click(chatbot_handler, [chat_input, chatbot], [chatbot, chat_input])
        reset_btn.click(lambda: (reset_chat(), []), outputs=[chatbot])

        generate_btn.click(
            generate_from_chatbot,
            inputs=[metrics, ddim_steps, num_samples],
            outputs=[gallery, svg_download, rooms_table, connections_table]
        )

    return demo

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    try:
        load_model_once()
    except Exception as e:
        print(f"Model preload warning: {e}")

    build_ui().launch(share=True)