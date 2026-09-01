# Frame Budget

A query-aware video frame selection project.

## Overview

The pipeline uses LongCLIP to select video frames that are relevant to a question. The selected frames are passed to Qwen2-VL-7B for video question answering.

## Problem

Video language models cannot always process every frame in a long video. Uniform sampling can miss the short moment that answers a question, while passing too many frames increases computation and context length.

## Approach

Each frame is compared with the input question using LongCLIP. The frames with the highest similarity scores are kept within a fixed frame budget. Qwen2-VL-7B then receives only those selected frames and answers the question.

## Result

Evaluated Qwen2-VL-7B on MLVU and obtained **59.4% accuracy**.

## Main ideas

- Use the question to rank video frames instead of sampling frames uniformly.
- Keep the most relevant frames within a fixed frame budget.
- Evaluate the selected frames with Qwen2-VL-7B on MLVU.
