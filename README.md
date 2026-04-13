# 🏠 2D Floor Plan Generator from Natural Language

> **Convert natural language descriptions into beautiful 2D floor plans using AI and diffusion models**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![Status](https://img.shields.io/badge/status-Active-success)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Room Type Legend](#room-type-legend)
- [Installation](#installation)
- [Setup & Configuration](#setup--configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**2D Floor Plan Generator** is an intelligent system that transforms natural language descriptions (e.g., "2 bedroom apartment with kitchen and living room") into geometrically valid 2D floor plans using:

- **Diffusion Models** for spatial layout generation
- **Google Gemini AI** for intelligent room parsing and connectivity
- **Gradio UI** for interactive user experience
- **PyTorch** for deep learning inference

Perfect for architects, interior designers, real estate professionals, and anyone needing quick floor plan prototyping!

---

## ✨ Features

### Core Functionality
- ✅ **Natural Language Processing** - Understand room descriptions in plain English
- ✅ **Intelligent Room Parsing** - Extract room types, counts, and requirements
- ✅ **Graph-Based Room Connectivity** - Ensure logical room connections
- ✅ **Physics-Based Collision Resolution** - Prevent room overlaps
- ✅ **Smart Topology Optimization** - Avoid "Living Room Gravity" patterns

### Output Formats
- 📊 **SVG Export** - Vector format for scalability
- 🖼️ **PNG Export** - Raster format for quick sharing
- 📐 **Measurements** - Optional dimension labels on floor plans

### User Interface
- 💬 **Interactive Chatbot** - Iterative design refinement
- 🎨 **Color-Coded Rooms** - Visual room type identification
- 📈 **Real-Time Preview** - Instant layout visualization
- 📋 **Data Tables** - Room and connection summaries

### Advanced Options
- 🔧 **Developer Mode** - Seed control for reproducible results
- ⚙️ **DDIM Step Control** - Adjust generation quality (100-1000 steps)
- 📦 **Batch Generation** - Generate multiple layouts at once
- 🧮 **Metrics Display** - Show door dimensions and measurements

---

## 🎨 Room Type Legend

| Room Type | Color | Hex Code |
|-----------|-------|----------|
| Living Room | 🔴 Red | #EE4D4D |
| Kitchen | 🟠 Orange | #C67C7B |
| Bedroom | 🟡 Yellow | #FFD274 |
| Bathroom | ⚫ Gray | #BEBEBE |
| Balcony | 🔵 Light Blue | #BFE3E8 |
| Entrance | 🟢 Green | #7BA779 |
| Dining Room | 💗 Pink | #E87A90 |
| Study Room | 🧡 Coral | #FF8C69 |
| Storage | 🌊 Teal | #1F849B |
| Front Door | ⚪ Light Gray | #727171 |
| Interior Door | 💜 Purple | #D3A2C7 |
| Unknown | 🟣 Mauve | #785A67 |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration, optional)
- 8GB+ RAM recommended
- 2GB+ disk space for model checkpoint

### Step 1: Clone Repository
```bash
git clone https://github.com/crajeshaFile/2D-Floor-Plan-Generator.git
cd 2D-Floor-Plan-Generator# Generating-2D-Floor-Plan-from-Natural-Language-Prompt-

Step 2: Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Install System Dependencies (Linux/Ubuntu)
bash
sudo apt-get install -y libopenmpi-dev
🔧 Setup & Configuration
1. Download Model Checkpoint
Download the pre-trained diffusion model:

bash
mkdir -p ckpt
# Download model250000.pt from the house_diffusion repository
# Place it in: ckpt/model250000.pt
Download Model

2. Configure Gemini API Key
Get your free API key from Google AI Studio:

Option A: Environment Variable

bash
export GEMINI_API_KEY="your-api-key-here"
Option B: Direct Configuration (in app files)

Python
GEMINI_API_KEY = "your-api-key-here"
3. Set Model Path (Optional)
bash
export MODEL_PATH="ckpt/model250000.pt"
🚀 Usage
Quick Start: Basic App
bash
python app.py
Advanced: With Chatbot & Developer Options
bash
python 3dapp.py
With Smart Topology Optimization
bash
python gapp.py
Access Web Interface
Open your browser to: http://localhost:7860

📝 Usage Examples
Example 1: Simple 2 Bedroom House
Input:

Code
"2 bedroom apartment with kitchen, bathroom and living room"
Output:

2 bedrooms connected to living room
Kitchen connected to living room
Bathroom automatically connected to bedroom
Entrance added automatically
Example 2: Luxury 3 BHK
Input:

Code
"Luxury 3 bedroom house with 2 bathrooms, open kitchen, dining area, study room, and balcony"
Output:

3 bedrooms, each with ensuite bathrooms
Open kitchen-dining area connected
Study room accessible from living room
Balcony connected to bedroom
Example 3: Studio Apartment
Input:

Code
"Studio apartment with kitchenette, bathroom, and living area"
Output:

Single open living/kitchen area
Separate bathroom
Minimal door nodes
📁 Project Structure
Code
2D-Floor-Plan-Generator/
├── app.py                      # Main production app
├── 3dapp.py                    # Advanced version with dropdowns
├── gapp.py                     # Smart topology version
├── capp.py                     # Chatbot-enabled version
├── app2.0.py                   # Enhanced Gemini version
├── testapp.py                  # Test/prototype
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies
├── README.md                   # This file
├── ckpt/
│   └── model250000.pt         # Diffusion model (download separately)
├── outputs/
│   └── pred/                  # Generated SVG/PNG outputs
└── generated_svgs/            # Output folder for layouts
🔌 API Reference
Main Functions
generate_from_prompt(prompt, metrics, ddim_steps, num_samples, auto_add_entrance)
Generates floor plans from a text description.

Parameters:

prompt (str): Natural language house description
metrics (bool): Show door measurements
ddim_steps (int): Diffusion steps (100-1000, higher = better quality)
num_samples (int): Number of layouts to generate
auto_add_entrance (bool): Automatically add entrance if missing
Returns:

Python
(png_paths, svg_paths, color_guide, rooms_df, connections_df)
process_prompt_with_gemini(prompt, auto_add_entrance)
Parses prompt using Google Gemini AI.

Returns:

Python
{
    "rooms": [
        {"room_type": "Living Room", "num_corners": 4},
        {"room_type": "Bedroom", "num_corners": 4}
    ],
    "connections": [
        {"room1": 0, "room2": 1}
    ]
}
🧠 How It Works
1. Natural Language Understanding
Code
User Input: "2 bedroom house with kitchen"
    ↓
Gemini AI Parser
    ↓
Structured JSON: {rooms: [...], connections: [...]}
2. Graph Optimization
Code
Initial Star Topology (all rooms → living room)
    ↓
Smart Rewriter (kitchen → dining → living)
    ↓
Optimal Tree Topology
3. Diffusion Model Sampling
Code
Room Graph + Constraints
    ↓
Diffusion Model (DDIM sampling)
    ↓
Point Cloud Layout
    ↓
Polygon Reconstruction
4. Collision Resolution
Code
Overlapping Rooms
    ↓
Physics Engine (iterative repulsion)
    ↓
Non-overlapping Layout
5. Rendering
Code
Polygons + Room Types + Colors
    ↓
SVG/PNG Conversion
    ↓
Final Floor Plan
⚙️ Configuration Options
DDIM Steps (Quality vs Speed)
Steps	Quality	Speed	Use Case
50	Poor	Very Fast	Testing
100	Good	Fast	Daily Use
250	Excellent	Medium	Production
500+	Best	Slow	High-Quality Output
Room Connectivity Rules
Living Room:

Central hub connecting to: Kitchen, Bedrooms, Study, Balcony, Dining
Bathrooms:

Connect to bedrooms preferentially
Fallback to living room if insufficient bathrooms
Kitchen:

Connects to: Dining room, Living room, Storage
Balcony:

Connects to: Living room or Bedroom
