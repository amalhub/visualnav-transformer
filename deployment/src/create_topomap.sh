#!/bin/bash

# Create a new tmux session
session_name="gnm_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run ros2 daemon in the first pane
tmux select-pane -t 0
tmux send-keys "ros2 topic echo /go2/camera" Enter

# Run the create_topomap.py script with command line args in the second pane
tmux select-pane -t 1
# tmux send-keys "conda_base" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "vint" Enter
tmux send-keys "python create_topomap.py $1" Enter

# Change the directory to ../topomaps/bags and run the ros2 bag play command in the third pane
tmux select-pane -t 2
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "ros2 bag play $2 --rate 1.5" # feel free to change the playback rate to change the edge length in the graph

# Attach to the tmux session
tmux -2 attach-session -t $session_name
