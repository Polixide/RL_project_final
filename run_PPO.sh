#!/bin/bash
source rl_env/bin/activate
export WEBOTS_HEADLESS=1

sudo pkill -f webots
sudo pkill -f webots-bin
sudo pkill -f Xvfb
sleep 2

#Starting 10 webots instances
for i in {0..9}
do
  PORT=$((10000 + $i))
  echo "Starting Webots instance on port $PORT"
  PORT=$PORT xvfb-run -a webots --mode=fast --no-rendering --batch --minimize worlds/new_Highway.wbt &
done


# Wait until all the instances are ready
sleep 30
sudo lsof -i -P -n | grep LISTEN

# Start the training file
python3 train_PPO.py

