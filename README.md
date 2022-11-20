# Stable Diffusion With Style Tokens

This repo is entirly based on [Stable Diffusion](https://github.com/CompVis/stable-diffusion).
Please refer to Stable Diffusion for setting up.

This fork made the following changes:

- adding configs/latent-diffusion/txt2img-1p4b-eval_prompt.yaml
- adding BERTEmbedderWithPrompt in ldm/modules/encoders/modules.py
- adding TransformerWrapperWithPrompt in ldm/modules/x_transformer.py
- adding ldm/data/coco_gen.py
- modifying main.py to take pretrained model

First, please obtain text2img-large/model.ckpt following the guide in [Latent Diffusion](https://github.com/CompVis/latent-diffusion)

Next, please refer to scripts/gen_coco_cap.py and ldm/data/coco_gen.py for generating the training dataset, you need to change the address in gen_coco_cap.py and coco_gen.py accordingly. You also need to download the captions_val2014.json file from [MS COCO](https://cocodataset.org/#download)
```
python scripts/gen_coco_cap.py --prompt " " --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50 --ckpt models/ldm/text2img-large/model.ckpt --config configs/latent-diffusion/txt2img-1p4B-eval.yaml
```

Then, run the following code to train style tokens:
```
CUDA_VISIBLE_DEVICES="0" python main.py --base configs/latent-diffusion/txt2img-1p4B-eval_prompt.yaml -t --gpus 0, --ckpt models/ldm/text2img-large/model.ckpt
```

To generate images using the original models, refer to [Latent Diffusion](https://github.com/CompVis/latent-diffusion).
To generate images using the style tokens, run the following:

```
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse." --ddim_eta 0.0 --n_samples 4 --n_iter 4 --scale 5.0  --ddim_steps 50 --ckpt logs/<Trained .ckpt path> --config configs/latent-diffusion/txt2img-1p4B-eval_prompt.yaml
```

