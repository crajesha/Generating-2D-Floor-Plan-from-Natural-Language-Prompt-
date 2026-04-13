import gradio as gr
import pandas as pd
import numpy as np
import torch as th
from house_diffusion import dist_util
from house_diffusion.script_util import (
    create_model_and_diffusion,
)
from PIL import Image
import io
import drawSvg as drawsvg
import cairosvg
from tqdm import tqdm

import webcolors
import tempfile
from pathlib import Path
import shutil
import os


ROOM_CLASS = {
    'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4,
    'Balcony': 5, 'Entrance': 6, 'Dining Room': 7, 'Study Room': 8,
    'Storage': 10, 'Front Door': 11, 'Unknown': 13, 'Interior Door': 12
}

ROOM_CATEGORIES = {
    'Living Room': 1, 'Kitchen': 2, 'Bedroom': 3, 'Bathroom': 4,
    'Balcony': 5, 'Entrance': 6, 'Dining Room': 7, 'Study Room': 8,
    'Storage': 10, 'Front Door': 11, 'Other': 13
}


def save_samples(
        sample, ext, model_kwargs,
        tmp_count, num_room_types,
        # save_gif=False,
        save_gif=True,
        door_indices=[11, 12, 13], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False, metrics=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []

    print(sample.shape)

    if not save_gif:
        sample = sample[-1:]
    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))
            polys = []
            types = []
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'{prefix}src_key_padding_mask'][i][j] == 1:
                    continue
                point = point.cpu().data.numpy()
                if j == 0:
                    poly = []
                if j > 0 and (model_kwargs[f'{prefix}room_indices'][i, j] != model_kwargs[f'{prefix}room_indices'][
                    i, j - 1]).any():
                    polys.append(poly)
                    types.append(c)
                    poly = []
                pred_center = False
                if pred_center:
                    point = point / 2 + 1
                    point = point * resolution // 2
                else:
                    point = point / 2 + 0.5
                    point = point * resolution
                poly.append((point[0], point[1]))
                c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j - 1].cpu().numpy())
            polys.append(poly)
            types.append(c)
            for poly, c in zip(polys, types):
                if c in door_indices or c == 0:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                  fill_opacity=1.0, stroke='black', stroke_width=1))

            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])

                # TODO --------------------------------------------------------------------------------------
                # https://github.com/sakmalh/house_diffusion
                line_lengths = [np.linalg.norm(np.array(poly[i]) - np.array(poly[(i + 1) % len(poly)])) for i in
                                range(len(poly))]

                if metrics:
                    text_size = 5
                    for z, length in enumerate(line_lengths):
                        # Calculate the mid-point of the line segment
                        midpoint = ((poly[z][0] + poly[(z + 1) % len(poly)][0]) / 2,
                                    (poly[z][1] + poly[(z + 1) % len(poly)][1]) / 2)

                        # Calculate x and y differences
                        x_diff = poly[z][0] - poly[(z + 1) % len(poly)][0]
                        y_diff = poly[z][1] - poly[(z + 1) % len(poly)][1]

                        # Determine text position adjustments based on differences
                        if int(y_diff) != 0:
                            if y_diff > 0:
                                text_x = midpoint[0] + text_size
                                text_y = midpoint[1]

                                draw_color.append(drawsvg.Line(
                                    text_x, text_y + text_size,  # Start point at the text label
                                            poly[z][0] + text_size, poly[z][1],  # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))

                                draw_color.append(drawsvg.Line(
                                    text_x, text_y - text_size,  # Start point at the text label
                                            poly[(z + 1) % len(poly)][0] + text_size, poly[(z + 1) % len(poly)][1],
                                    # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))
                            else:
                                text_x = midpoint[0] - text_size
                                text_y = midpoint[1]

                                draw_color.append(drawsvg.Line(
                                    text_x, text_y - text_size,  # Start point at the text label
                                            poly[z][0] - text_size, poly[z][1],  # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))

                                draw_color.append(drawsvg.Line(
                                    text_x, text_y + text_size,  # Start point at the text label
                                            poly[(z + 1) % len(poly)][0] - text_size, poly[(z + 1) % len(poly)][1],
                                    # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))
                        else:
                            if x_diff > 0:
                                text_x = midpoint[0]
                                text_y = midpoint[1] - text_size

                                draw_color.append(drawsvg.Line(
                                    text_x + text_size, text_y,  # Start point at the text label
                                    poly[z][0], poly[z][1] - text_size,  # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))

                                draw_color.append(drawsvg.Line(
                                    text_x - text_size, text_y,  # Start point at the text label
                                    poly[(z + 1) % len(poly)][0], poly[(z + 1) % len(poly)][1] - text_size,
                                    # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))
                            else:
                                text_x = midpoint[0]
                                text_y = midpoint[1] + text_size

                                draw_color.append(drawsvg.Line(
                                    text_x - text_size, text_y,  # Start point at the text label
                                    poly[z][0], poly[z][1] + text_size,  # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))

                                draw_color.append(drawsvg.Line(
                                    text_x + text_size, text_y,  # Start point at the text label
                                    poly[(z + 1) % len(poly)][0], poly[(z + 1) % len(poly)][1] + text_size,
                                    # End point at the polygon endpoint
                                    stroke='black',
                                    stroke_width=1
                                ))

                        # Add the text label to the SVG
                        draw_color.append(
                            drawsvg.Text(
                                f'{int(abs(length))}',  # Format the length to two decimal places
                                text_size,
                                text_x, text_y,
                                fill='black',
                                text_anchor='middle',
                                alignment_baseline='middle'
                            )
                        )

                draw_color.append(
                    drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type],
                                  fill_opacity=1.0, stroke='black', stroke_width=1))

            if k == sample.shape[0] - 1 or True:
                if save_svg:
                    # draw_color.saveSvg(f'outputs/{ext}/{tmp_count + i}c_{k}_{ext}.svg')
                    return draw_color
                else:
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color.asSvg()))).save(
                        f'outputs/{ext}/{tmp_count + i}c_{ext}.png')

        # if save_gif:
        #     imageio.mimwrite(f'outputs/gif/{tmp_count + i}.gif', images, fps=10, loop=1)
        #     imageio.mimwrite(f'outputs/gif/{tmp_count + i}_v2.gif', images2, fps=10, loop=1)
        #     imageio.mimwrite(f'outputs/gif/{tmp_count + i}_v3.gif', images3, fps=10, loop=1)
    return graph_errors


def function_test(org_graphs, corners, room_type):
    get_one_hot = lambda x, z: np.eye(z)[x]
    max_num_points = 100

    house = []
    corner_bounds = []
    num_points = 0

    for i, room in enumerate(room_type):
        # Adding conditions
        num_room_corners = corners[i]
        rtype = np.repeat(np.array([get_one_hot(room, 25)]), num_room_corners, 0)
        room_index = np.repeat(np.array([get_one_hot(len(house) + 1, 32)]), num_room_corners, 0)
        corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
        # Src_key_padding_mask
        padding_mask = np.repeat(1, num_room_corners)
        padding_mask = np.expand_dims(padding_mask, 1)
        # Generating corner bounds for attention masks
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
        if not is_connected:
            door_mask[corner_bounds[i][0]:corner_bounds[i][1],
            corner_bounds[living_room_index][0]:corner_bounds[living_room_index][1]] = 0

    syn_houses = house_layouts
    syn_door_masks = door_mask
    syn_self_masks = self_mask
    syn_gen_masks = gen_mask

    syn_graph = np.concatenate((org_graphs, np.zeros([200 - len(org_graphs), 3])), 0)

    cond = {
        'syn_door_mask': syn_door_masks,
        'syn_self_mask': syn_self_masks,
        'syn_gen_mask': syn_gen_masks,
        'syn_room_types': syn_houses[:, 2:2 + 25],
        'syn_corner_indices': syn_houses[:, 2 + 25:2 + 57],
        'syn_room_indices': syn_houses[:, 2 + 57:2 + 89],
        'syn_src_key_padding_mask': 1 - syn_houses[:, 2 + 89],
        'syn_connections': syn_houses[:, 2 + 90:2 + 92],
        'syn_graph': syn_graph,
    }

    return cond


def create_layout(graphs, corners, room_type, metrics=False, use_ddim=True, ddim_steps=100, num_samples=4):
    model_path = "ckpt/model250000.pt"
    steps = f"ddim{ddim_steps}"
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
        "timestep_respacing": steps,
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
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
                13: '#785A67', 12: '#D3A2C7'}
    num_room_types = 14
    sample_fn = (diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop)
    print(graphs, corners, room_type)
    model_kwargs = function_test(graphs, corners, room_type)
    for key in model_kwargs:
        # model_kwargs[key] = th.from_numpy(np.array([model_kwargs[key]])).cuda()
        model_kwargs[key] = th.from_numpy(np.array([model_kwargs[key]])).cpu()

    png_paths = []
    svg_paths = []
    for count in range(num_samples):
        sample = sample_fn(
            model,
            th.Size([1, 2, 100]),
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )

        sample = sample.permute([0, 1, 3, 2])

        pred = save_samples(sample, 'pred', model_kwargs, count, num_room_types, ID_COLOR=ID_COLOR,
                            is_syn=True, draw_graph=False, save_svg=True, save_gif=False, metrics=metrics)

        temp_svg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".svg")
        pred.saveSvg(temp_svg_file.name)
        png_file_name = temp_svg_file.name.split(".")[0].split("/")[-1]
        png_file_path = f'./generated_svgs/{png_file_name}.png'
        # print(temp_svg_file.name)
        # print(png_file_name)
        # print(png_file_path)

        Image.open(io.BytesIO(cairosvg.svg2png(pred.asSvg()))).save(png_file_path)

        output_dir = Path("./generated_svgs")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = temp_svg_file.name.split("/")[-1]
        persistent_path = Path(f"{output_dir}/{file_name}")
        shutil.move(temp_svg_file.name, persistent_path)
        os.chmod(persistent_path, 0o644)

        svg_paths.append(str(persistent_path))
        png_paths.append(png_file_path)
        # print(str(persistent_path))

    return png_paths, svg_paths


rooms_data = []
edges_data = []


def generate_layout(metrics: bool, ddim_steps: int, num_samples: int):
    room_list = []
    room_corners = []
    living_room = 0
    front_door = False
    entrance = -1

    print(rooms_data)
    print(edges_data)

    for i, room in enumerate(rooms_data):
        room_list.append(ROOM_CLASS[room['room_type']])
        if room['num_corners'] != 0:
            room_corners.append(int(room['num_corners']))
        else:
            room_corners.append(4)

        if room['room_type'] == "Living Room":
            living_room = i

        elif room['room_type'] == "Entrance":
            entrance = i

        elif room['room_type'] == "Front Door":
            front_door = True

    edges = []
    for edge in edges_data:
        source_id = int(edge['room1_id'].split()[0])
        target_id = int(edge['room2_id'].split()[0])
        edges.append([source_id, 1, target_id])

        index = len(room_list)
        room_list.append(12)
        room_corners.append(4)
        edges.append([source_id, 1, index])
        edges.append([target_id, 1, index])

    if not front_door:
        room_list.append(11)
        room_corners.append(4)
        if entrance == -1:
            edges.append([len(room_list) - 1, 1, living_room])
        else:
            edges.append([len(room_list) - 1, 1, entrance])

    if np.sum(room_corners) > 99:
        return {"Error": "Number of Corners exceeded"}

    print(room_list, room_corners, edges)
    png_paths, svg_paths = create_layout(edges, room_corners, room_list, metrics=metrics, ddim_steps=ddim_steps,
                             num_samples=num_samples)

    png_color_guide = './color_guide.png'

    rooms_data.clear()
    edges_data.clear()

    rooms_df = pd.DataFrame(columns=["room_id", "room_type", "num_corners"])
    edges_df = pd.DataFrame(columns=["edge_id", "room1_id", "room2_id"])

    return png_paths, svg_paths, png_color_guide, rooms_df, edges_df


with gr.Blocks() as demo:
    gr.Markdown("## House Layout Generator")

    with gr.Row():
        room_type = gr.Dropdown(label="Room Type", choices=list(ROOM_CATEGORIES.keys()), value="Living Room")
        num_corners = gr.Number(label="Number of Corners", value=4)
        add_room_button = gr.Button("Add Room")

    with gr.Row():
        room1_id = gr.Dropdown(label="Room 1", choices=[], value=None)
        room2_id = gr.Dropdown(label="Room 2", choices=[], value=None)
        add_edge_button = gr.Button("Add Edge")

    rooms_table = gr.DataFrame(label="Rooms Table")
    edges_table = gr.DataFrame(label="Edges Table")

    metrics_toggle = gr.Checkbox(label="Include metrics", value=True)
    ddim_input = gr.Number(label="DDIM steps", value=100)
    num_sample = gr.Number(label="Number of samples", value=4)

    png_gallery = gr.Gallery(label="Layout PNG Outputs", columns=4)
    svg_files = gr.File(label="Layout SVG Outputs (higher quality)")
    png_color_guide = gr.Image(label="Color Guide")

    def add_room(room_type, num_corners):
        room_id = len(rooms_data)
        rooms_data.append({
            "room_id": room_id,
            "room_type": room_type,
            "num_corners": num_corners
        })
        return update_rooms_and_edges()


    def add_edge(room1_id, room2_id):
        edge_id = len(edges_data)
        edges_data.append({
            "edge_id": edge_id,
            "room1_id": room1_id,
            "room2_id": room2_id
        })
        return update_rooms_and_edges()


    def update_rooms_and_edges():
        rooms_df = pd.DataFrame(rooms_data, columns=["room_id", "room_type", "num_corners"])
        edges_df = pd.DataFrame(edges_data, columns=["edge_id", "room1_id", "room2_id"])
        room_options = [f"{room['room_id']} {room['room_type']}" for room in rooms_data]
        return rooms_df, edges_df, gr.update(choices=room_options, value=None), gr.update(choices=room_options,
                                                                                          value=None)


    generate_button = gr.Button("Generate Layout")
    # generate_button.click(generate_layout, inputs=[metrics_toggle, ddim_input, num_sample], outputs=[png_gallery, svg_files, png_color_guide])
    generate_button.click(
        generate_layout,
        inputs=[metrics_toggle, ddim_input, num_sample],
        outputs=[
            png_gallery,
            svg_files,
            png_color_guide,
            rooms_table,
            edges_table
        ]
    )
    add_room_button.click(add_room, inputs=[room_type, num_corners],
                          outputs=[rooms_table, edges_table, room1_id, room2_id])
    add_edge_button.click(add_edge, inputs=[room1_id, room2_id], outputs=[rooms_table, edges_table, room1_id, room2_id])


#demo.launch()
# global demo
demo.launch(share=True)