import utils
from video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse

import streamlit as st
import tempfile
import cv2

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0
        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
            cnt += 1
            return total_loss
        optimizer.step(closure)

    return dump_path

def run_gui():
    st.set_page_config(page_title="Neural Style Transfer", layout="wide")
    st.title("🎨 Neural Style Transfer - Echo Style")

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

    model = st.selectbox("Model", ["vgg19", "vgg16"])
    optimizer = st.selectbox("Optimizer", ["adam", "lbfgs"])
    init_method = st.selectbox("Initialization Method", ["random", "content", "style"])

    height = st.slider("Image Height", 256, 800, 400, step=64)
    cw = st.number_input("Content Weight", value=1e5)
    sw = st.number_input("Style Weight", value=3e4)
    tvw = st.number_input("Total Variation Weight", value=1.0)

    if content_file and style_file and st.button("Generate Stylized Image"):
        with tempfile.TemporaryDirectory() as td:
            cpath = os.path.join(td, "content.jpg")
            spath = os.path.join(td, "style.jpg")
            with open(cpath, "wb") as f: f.write(content_file.read())
            with open(spath, "wb") as f: f.write(style_file.read())

            st.image(cpath, caption="Content Image", use_column_width=True)
            st.image(spath, caption="Style Image", use_column_width=True)

            config = {
                "content_img_name": "content.jpg",
                "style_img_name": "style.jpg",
                "content_images_dir": td,
                "style_images_dir": td,
                "output_img_dir": td,
                "img_format": (4, ".jpg"),
                "height": height,
                "content_weight": cw,
                "style_weight": sw,
                "tv_weight": tvw,
                "optimizer": optimizer,
                "model": model,
                "init_method": init_method,
                "saving_freq": -1
            }

            st.info("Running style transfer… this may take a few minutes ⏳")
            torch.cuda.empty_cache()
            out_dir = neural_style_transfer(config)

            files = sorted(os.listdir(out_dir))
            last = files[-1]
            img = cv2.imread(os.path.join(out_dir, last))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption="🎉 Stylized Output", use_column_width=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run Streamlit GUI")
    args, _ = parser.parse_known_args()
    if args.gui:
        run_gui()
    else:
        print("Please run with --gui to launch the interface")