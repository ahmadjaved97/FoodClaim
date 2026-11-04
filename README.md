This repository contains code used for the FoodClaim Dataset.

## Qwen scripts

1. [qwen3_classify_human.py](image_generation/qwen_scripts/qwen3_classify_human.py): Classifies human presence using its built-in prompt; hands alone count as no human.
2. [qwen3_description_generation.py](image_generation/qwen_scripts/qwen3_description_generation.py): Generates image descriptions using Qwen vision-language models.
3. [qwen3_generate_manipulation_instruction.py](image_generation/qwen_scripts/qwen3_generate_manipulation_instruction.py): Generates Flux manipulation instructions from images, descriptions, and target manipulations using Qwen3.5-VL.
4. [qwen3_generate_manipulated_food_descriptions.py](image_generation/qwen_scripts/qwen3_generate_manipulated_food_descriptions.py): Rewrites food descriptions to reflect supplied manipulations using a Qwen text model.
5. [qwen3_prompt_summarizer.py](image_generation/qwen_scripts/qwen3_prompt_summarizer.py): Shortens CSV prompts with Qwen3.5-VL and checks their lengths against the FLUX token limit.
6. [prompt_description_generation.json](image_generation/qwen_scripts/prompt_description_generation.json): Defines prompts for image descriptions.
7. [prompt_manipulation_instruction_generation.json](image_generation/qwen_scripts/prompt_manipulation_instruction_generation.json): Defines prompts for manipulation instructions.

## FLUX scripts

1. [flux1_image_generation.py](image_generation/flux_scripts/flux1_image_generation.py): Generates food images from CSV descriptions using FLUX.1-dev or FLUX.1-schnell.
2. [flux2_klein_image_generation.py](image_generation/flux_scripts/flux2_klein_image_generation.py): Generates food images from CSV descriptions using FLUX.2-klein models.
3. [flux2_klein_image_manipulation.py](image_generation/flux_scripts/flux2_klein_image_manipulation.py): Edits source food images using FLUX.2-klein and manipulation instructions from a CSV.

The code for FLUX.2-dev image generation and manipulation is available in the [flux2-distributed-inference repository](https://github.com/ahmadjaved97/flux2-distributed-inference).

## Qwen image scripts

1. [qwen_image_edit_manipulation.py](image_generation/qwen_image_scripts/qwen_image_edit_manipulation.py): Edits source food images using Qwen-Image-Edit-2511 and manipulation instructions from a CSV.

## OpenRouter scripts

1. [open_router_image_generation.py](image_generation/open_router_scripts/open_router_image_generation.py): Generates or edits food images from CSV prompts through OpenRouter, with generation, editing, and automatic mode selection.

## Related repositories

The code used for training the models is available in the [fake-food-detection-training repository](https://github.com/ahmadjaved97/fake-food-detection-training).

The code for UniversalFakeDetect is available in the [UniversalFakeDetect repository](https://github.com/ahmadjaved97/UniversalFakeDetect).
