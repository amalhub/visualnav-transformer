import matplotlib.pyplot as plt
import os
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import matplotlib.pyplot as plt
import yaml

# ROS
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray
from utils import msg_to_pil, to_numpy, transform_images, load_model

from vint_train.training.train_utils import get_action
import torch
from PIL import Image as PILImage
import numpy as np
import argparse
import yaml
import time

# UTILS
from topic_names import (IMAGE_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC)


# CONSTANTS
TOPOMAP_IMAGES_DIR = "../topomaps/images"
MODEL_WEIGHTS_PATH = "../model_weights"
ROBOT_CONFIG_PATH ="../config/robot.yaml"
MODEL_CONFIG_PATH = "../config/models.yaml"
with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
RATE = robot_config["frame_rate"] 

# GLOBALS
context_queue = []
context_size = None  
subgoal = []

# Load the model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class NavigationNode(Node):
    def __init__(self, args):
        super().__init__('exploration')
        self.args = args
        
        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 
            IMAGE_TOPIC, 
            self.callback_obs, 
            10)
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, 
            WAYPOINT_TOPIC, 
            10)
        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray, 
            SAMPLED_ACTIONS_TOPIC, 
            10)
        self.goal_pub = self.create_publisher(
            Bool, 
            "/topoplan/reached_goal", 
            10)

        # Create timer for main loop
        self.timer = self.create_timer(1.0/RATE, self.timer_callback)
        
        self.setup_model()
        self.setup_topomap()
        
        self.closest_node = 0
        self.goal_node = len(self.topomap) - 1 if args.goal_node == -1 else args.goal_node
        self.reached_goal = False

    def callback_obs(self, msg):
        global context_queue, context_size
        obs_img = msg_to_pil(msg)
        if context_size is not None:
            if len(context_queue) < context_size + 1:
                context_queue.append(obs_img)
            else:
                context_queue.pop(0)
                context_queue.append(obs_img)

    def setup_model(self):
        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            model_paths = yaml.safe_load(f)

        model_config_path = model_paths[self.args.model]["config_path"]
        with open(model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        global context_size
        context_size = self.model_params["context_size"]

        # load model weights
        ckpth_path = model_paths[self.args.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            print(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        
        self.model = load_model(ckpth_path, self.model_params, device)
        self.model = self.model.to(device)
        self.model.eval()

        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]  # Store this for later use
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.num_diffusion_iters,
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

    def setup_topomap(self):
        topomap_filenames = sorted(os.listdir(os.path.join(
            TOPOMAP_IMAGES_DIR, self.args.dir)), key=lambda x: int(x.split(".")[0]))
        topomap_dir = f"{TOPOMAP_IMAGES_DIR}/{self.args.dir}"
        num_nodes = len(os.listdir(topomap_dir))
        self.topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_dir, topomap_filenames[i])
            self.topomap.append(PILImage.open(image_path))

    def timer_callback(self):
        # Move main navigation loop logic here
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > self.model_params["context_size"]:
            if self.model_params["model_type"] == "nomad":
                obs_images = transform_images(context_queue, self.model_params["image_size"], center_crop=False)
                obs_images = torch.split(obs_images, 3, dim=1)
                obs_images = torch.cat(obs_images, dim=1) 
                obs_images = obs_images.to(device)
                mask = torch.zeros(1).long().to(device)  

                start = max(self.closest_node - self.args.radius, 0)
                end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                goal_image = [transform_images(g_img, self.model_params["image_size"], center_crop=False).to(device) for g_img in self.topomap[start:end + 1]]
                goal_image = torch.concat(goal_image, dim=0)

                obsgoal_cond = self.model('vision_encoder', obs_img=obs_images.repeat(len(goal_image), 1, 1, 1), goal_img=goal_image, input_goal_mask=mask.repeat(len(goal_image)))
                dists = self.model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                self.closest_node = min_idx + start
                print("closest node:", self.closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < self.args.close_threshold), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                # infer action
                with torch.no_grad():
                    # encoder vision features
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.args.num_samples, 1, 1)
                    
                    # initialize action from Gaussian noise
                    noisy_action = torch.randn(
                        (self.args.num_samples, self.model_params["len_traj_pred"], 2), device=device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)  # Use the stored value
                    
                    start_time = time.time()
                    for k in self.noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    print("time elapsed:", time.time() - start_time)

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                # Convert numpy arrays to list of floats
                actions_data = np.concatenate((np.array([0.0]), naction.flatten()))
                sampled_actions_msg.data = [float(x) for x in actions_data]
                print("published sampled actions")
                self.sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[0] 
                chosen_waypoint = naction[self.args.waypoint]
            else:
                start = max(self.closest_node - self.args.radius, 0)
                end = min(self.closest_node + self.args.radius + 1, self.goal_node)
                distances = []
                waypoints = []
                batch_obs_imgs = []
                batch_goal_data = []
                for i, sg_img in enumerate(topomap[start: end + 1]):
                    transf_obs_img = transform_images(context_queue, model_params["image_size"])
                    goal_data = transform_images(sg_img, model_params["image_size"])
                    batch_obs_imgs.append(transf_obs_img)
                    batch_goal_data.append(goal_data)
                    
                # predict distances and waypoints
                batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(device)
                batch_goal_data = torch.cat(batch_goal_data, dim=0).to(device)

                distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
                distances = to_numpy(distances)
                waypoints = to_numpy(waypoints)
                # look for closest node
                min_dist_idx = np.argmin(distances)
                # chose subgoal and output waypoints
                if distances[min_dist_idx] > args.close_threshold:
                    chosen_waypoint = waypoints[min_dist_idx][args.waypoint]
                    self.closest_node = start + min_dist_idx
                else:
                    chosen_waypoint = waypoints[min(
                        min_dist_idx + 1, len(waypoints) - 1)][args.waypoint]
                    self.closest_node = min(start + min_dist_idx + 1, goal_node)
        # RECOVERY MODE
        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)  
        waypoint_msg = Float32MultiArray()
        # Convert numpy array to list of floats
        waypoint_msg.data = [float(x) for x in chosen_waypoint]
        self.waypoint_pub.publish(waypoint_msg)

        # Update reached_goal status first
        self.reached_goal = bool(self.closest_node == self.goal_node)
        
        # Create and publish goal message with proper boolean type
        goal_msg = Bool()
        goal_msg.data = self.reached_goal
        self.goal_pub.publish(goal_msg)
        
        if self.reached_goal:
            print("Reached goal! Stopping...")


def main(args=None):
    rclpy.init(args=args)
    
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad",
        type=str,
        help="model name (only nomad is supported) (hint: check ../config/models.yaml) (default: nomad)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="""goal node index in the topomap (if -1, then the goal node is 
        the last node in the topomap) (default: -1)""",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="""temporal distance within the next node in the topomap before 
        localizing to it (default: 3)""",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="""temporal number of locobal nodes to look at in the topopmap for
        localization (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parsed_args = parser.parse_args()
    
    navigation_node = NavigationNode(parsed_args)
    
    try:
        rclpy.spin(navigation_node)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

