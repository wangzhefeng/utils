#!/bin/bash

# cd project directory
cd E:/projects/utils

# activate virtual environment
source .venv/Scripts/activate

# update codes
git add .
git commit -m "update"
git pull
git push

# enter project directory
code .
