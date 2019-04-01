#!/usr/bin/bash
echo "delete *.pyc"
find . -name "*.pyc" -delete
echo "delete __pycache__"
find . -name "__pycache__" -delete
