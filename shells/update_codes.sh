# !/bin/bash

# push TinyVLM
echo "--------------------------"
echo "update TinyLLM codes..."
echo "--------------------------"
# Checkout the specified branch
git checkout main
echo "Successfully checked out main."

git add .
git commit -m "init project"

# Pull the latest changes from the remote config repository
git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the config."

# push utils
echo "--------------------------"
echo "update utils codes..."
echo "--------------------------"
cd utils

git add .
git commit -m "update"
git pull
echo "Successfully pulled the latest changes."

git push
echo "Successfully checked out master and updated the config."
