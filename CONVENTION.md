# Roadmap

1. `base_models`: various base models to start continued pretraining from. The folder is structured as hierarchical as possible:

   `[ModelFamilyName]/[Implementation]/[ModelName]`

   For example, `OLMo/olmo/OLMo-1B-final`. In this folder, we use huggingface-compatible checkpoints --- `hf` and OLMo-compatible checkpoints --- `olmo`.

2. `checkpoints`: where individual member save their trained checkpoints

3. `data`: raw text data

4. `processed_data`: "processed" here incldues: train/valid split, tokenization, `pickle`, etc.

5. `OLMo`: main branch. **!!!IMPORTANT!!!:** DO NOT edit this folder. Each memeber should clone from `git@github.com:leo-liuzy/OLMo.git`.
