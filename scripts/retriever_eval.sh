#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python src/retriever.py retrieve \
  -c configs/datasets.yaml \
  --ids-path assets/data/test_acl_2024.json

wait
echo "Retriever Eval Finish..."
