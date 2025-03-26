import os
import time
import json
import torch
import hashlib
import argparse
import numpy as np

from Crypto.Random import get_random_bytes

from watermark import Gaussian_Shading_chacha
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from swa import key_channel_enhance, FY_shuffle, exact_k, FY_inverse_shuffle
from utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_number", type=int, default=1000)
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--dataset_path", type=str, default="../fid_outputs/coco")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--sub_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_inversion_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--user_number", type=int, default=100000)
    parser.add_argument("--fpr", type=float, default=0.00001)
    parser.add_argument("--ch_factor", type=int, default=1)
    parser.add_argument("--hw_factor", type=int, default=8)
    parser.add_argument("--ch_num", type=int, default=3)
    parser.add_argument("--gen_seed", type=int, default=0)
    parser.add_argument("--swa_M", type=int, default=8)
    parser.add_argument("--swa_R", type=int, default=64)

    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)

    return parser.parse_args()

def main():
    args = args_parser()
    gs = Gaussian_Shading_chacha(
        ch_factor=args.ch_factor, 
        hw_factor=args.hw_factor, 
        ch_num=args.ch_num, 
        fpr=args.fpr, 
        user_number=args.user_number
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16
    ).to("cuda")

    dataset, prompt_key = get_dataset(args)

    if args.sub_path is None:
        sub_path = hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]
    else:
        sub_path = args.sub_path
    image_path = os.path.join(args.output_path, sub_path)
    os.makedirs(image_path, exist_ok=True)

    ksr = 0
    bit_acc = 0
    for i in range(args.test_number):
        seed = i + args.gen_seed

        prompt = dataset[i][prompt_key]
        text_embeddings = pipe.get_text_embedding(prompt)

        key_channel, k = key_channel_enhance(args.swa_M, args.swa_R)
        init_latents_w = gs.create_watermark_and_return_w()
        shuffled_init_latents_w = FY_shuffle(init_latents_w, k)

        init_latents = torch.cat([shuffled_init_latents_w, key_channel.unsqueeze(0).cuda()], dim=1).half().cuda()
        outputs = pipe(
            prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=512,
            width=512,
            latents=init_latents,
        )
        image_w = outputs.images[0]
        image_w.save(os.path.join(image_path, f"{i:05d}.png"))

        image_w_distortion = image_distortion(image_w, seed, args)

        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to("cuda")
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        reversed_key_channel = reversed_latents_w[:, 3, :, :]
        reversed_k = exact_k(reversed_key_channel, args.swa_M, args.swa_R)
        k_check = reversed_k == k

        reversed_latents_w = FY_inverse_shuffle(reversed_latents_w[0, :3, :, :], reversed_k)

        reversed_latents_w = reversed_latents_w.to(text_embeddings.dtype).to("cuda")
        acc_metric = gs.eval_watermark(reversed_latents_w)
        print(k_check, acc_metric)
        ksr += k_check
        bit_acc += acc_metric
    print(ksr / args.test_number, bit_acc / args.test_number)
    with open(os.path.join(image_path, "result.json"), "w") as f:
        json.dump({"ksr": ksr / args.test_number, "bit_acc": bit_acc / args.test_number}, f)

if __name__ == "__main__":
    main()




