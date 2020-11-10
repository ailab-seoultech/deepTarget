#!/bin/bash

echo "Downloading data.tar.gz..."
wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1G9PWJC7wgvEecQWV7WaQaZw-Wq4z-A9u' -O data.tar.gz

echo "Extracting data.tar.gz..."
tar -xzf data.tar.gz
rm data.tar.gz
