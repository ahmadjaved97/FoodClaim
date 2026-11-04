This repository contains code used for the FoodClaim Dataset.

## Qwen scripts

1. [qwen3_classify_human.py](image_generation/qwen_scripts/qwen3_classify_human.py): Classifies whether an image contains a human being according to the prompt defined in the script. Images showing only hands are classified as having no human present.
2. [qwen3_description_generation.py](image_generation/qwen_scripts/qwen3_description_generation.py): Generates image descriptions using Qwen vision-language models.
3. [qwen3_generate_manipulation_instruction.py](image_generation/qwen_scripts/qwen3_generate_manipulation_instruction.py): Generates Flux inpainting/manipulation instructions using Qwen3.5-VL, based on the image, its description, and the target manipulation supplied in the input CSV.
4. [qwen3_generate_manipulated_food_descriptions.py](image_generation/qwen_scripts/qwen3_generate_manipulated_food_descriptions.py): Uses a Qwen text model to rewrite existing food descriptions to include the supplied manipulation instructions, without loading images. Saves the results in a `manipulated_description` CSV column.
5. [prompt_description_generation.json](image_generation/qwen_scripts/prompt_description_generation.json): Defines the prompts used to generate image descriptions.
6. [prompt_manipulation_instruction_generation.json](image_generation/qwen_scripts/prompt_manipulation_instruction_generation.json): Defines the prompts used to generate manipulation instructions.

## Related repositories

The code used for training the models is available in the [fake-food-detection-training repository](https://github.com/ahmadjaved97/fake-food-detection-training).

The code for UniversalFakeDetect is available in the [UniversalFakeDetect repository](https://github.com/ahmadjaved97/UniversalFakeDetect).
