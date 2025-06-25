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
import cv2 as cv

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
 st.set_page_config(page_title="Neural Style Transfer App", layout="centered")
st.set_page_config(page_title="Neural Style Transfer App", layout="centered")
st.title("🎨 Neural Style Transfer")
st.markdown("Stylize your images using deep learning with VGG19.")

st.sidebar.header("🪄 Settings")
image_height = st.sidebar.slider("Image Height", min_value=128, max_value=720, value=400, step=8)
content_weight = st.sidebar.slider("Content Weight", 1e3, 1e6, value=1e5, step=1e3, format="%.0f")
style_weight = st.sidebar.slider("Style Weight", 1e3, 1e6, value=3e4, step=1e3, format="%.0f")
tv_weight = st.sidebar.slider("Total Variation Weight", 0.0, 10.0, value=1.0, step=0.1)
optimizer = st.sidebar.selectbox("Optimizer", options=["lbfgs", "adam"], index=0)
init_method = st.sidebar.selectbox("Init Method", options=["random", "content", "style"], index=1)

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content")
with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style")

run_button = st.button("✨ Stylize")

if run_button:
    if not content_file or not style_file:
        st.warning("Please upload both content and style images.")
    else:
        with st.spinner("Processing image with Neural Style Transfer..."):
            os.makedirs("data/content-images", exist_ok=True)
            os.makedirs("data/style-images", exist_ok=True)

            content_img_path = f"data/content-images/{content_file.name}"
            style_img_path = f"data/style-images/{style_file.name}"

            with open(content_img_path, "wb") as f:
                f.write(content_file.read())
            with open(style_img_path, "wb") as f:
                f.write(style_file.read())

            config = {
                "content_img_name": content_file.name,
                "style_img_name": style_file.name,
                "height": image_height,
                "content_weight": content_weight,
                "style_weight": style_weight,
                "tv_weight": tv_weight,
                "optimizer": optimizer,
                "model": "vgg19",
                "init_method": init_method,
                "saving_freq": -1,
                "content_images_dir": "data/content-images",
                "style_images_dir": "data/style-images",
                "output_img_dir": "data/output-images",
                "img_format": (4, ".jpg"),
            }

            result_path = neural_style_transfer(config)
            final_image_path = os.path.join(result_path, "final.jpg")

            if os.path.exists(final_image_path):
                output_image = cv.imread(final_image_path)[..., ::-1]  # BGR to RGB
                st.image(output_image, caption="Stylized Image", use_column_width=True)
                with open(final_image_path, "rb") as img_file:
                    st.download_button("💾 Download Image", data=img_file, file_name="stylized_output.jpg", mime="image/jpeg")
            else:
                st.error("Stylized image could not be generated. Please check your code or inputs.")