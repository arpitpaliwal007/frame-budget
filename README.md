# Frame Budget

A query-aware video frame selection project.

## Overview

The pipeline uses LongCLIP to select video frames that are relevant to a question. The selected frames are passed to Qwen2-VL-7B for video question answering.

## Result

Evaluated Qwen2-VL-7B on MLVU and obtained **59.4% accuracy**.

## Main ideas

- Use the question to rank video frames instead of sampling frames uniformly.
- Keep the most relevant frames within a fixed frame budget.
- Evaluate the selected frames with Qwen2-VL-7B on MLVU.
