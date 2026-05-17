from __future__ import annotations

import glob
import os

from PIL import Image
from tqdm.auto import tqdm

from wmbench.attacks.base import Attack
from wmbench.pipeline.fsutil import atomic_image_save
from wmbench.pipeline.resume import is_done, mark_done


def run_attack_stage(
    attacks: dict[str, Attack],
    watermarked_dir: str,
    out_root: str,
    *,
    resume: bool = False,
    diffusion_attack_batch_size: int = 1,
) -> None:
    diffusion_attack_batch_size = max(1, int(diffusion_attack_batch_size))
    wm_paths = sorted(
        p
        for p in glob.glob(os.path.join(watermarked_dir, "*"))
        if os.path.isfile(p)
        and p.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        and ".wmbench_meta" not in p
    )
    for attack_name, attack in attacks.items():
        for strength in attack.strengths:
            stren_tag = str(strength).replace(os.sep, "_")
            out_dir = os.path.join(out_root, attack_name, stren_tag)
            done_flag = os.path.join(out_dir, ".done")

            def dest_for(p: str) -> str:
                return os.path.join(out_dir, os.path.basename(p))

            all_exist = bool(wm_paths) and all(os.path.isfile(dest_for(p)) for p in wm_paths)
            if all_exist:
                mark_done(done_flag)
                continue

            if resume and is_done(done_flag):
                pass
            elif not resume and is_done(done_flag):
                os.remove(done_flag)

            os.makedirs(out_dir, exist_ok=True)
            pending = [src for src in wm_paths if not os.path.isfile(dest_for(src))]
            with tqdm(total=len(pending), desc=f"attack/{attack_name}/{stren_tag}") as pbar:
                i = 0
                while i < len(pending):
                    batch_src = pending[i : i + diffusion_attack_batch_size]
                    try:
                        images: list[Image.Image] = []
                        for src in batch_src:
                            with Image.open(src) as im:
                                images.append(im.convert("RGB"))
                        attacked_images = attack.apply_batch(images, strength)
                        if len(attacked_images) != len(batch_src):
                            raise RuntimeError(
                                f"attack batch output length mismatch: expected {len(batch_src)}, "
                                f"got {len(attacked_images)}"
                            )
                        for src, attacked in zip(batch_src, attacked_images):
                            atomic_image_save(attacked, dest_for(src))
                    except Exception as e:
                        err_path = os.path.join(out_root, "attack_errors.log")
                        os.makedirs(out_root, exist_ok=True)
                        with open(err_path, "a", encoding="utf-8") as ef:
                            for src in batch_src:
                                ef.write(f"{attack_name}\t{strength}\t{src}\t{e!r}\n")
                        raise
                    i += len(batch_src)
                    pbar.update(len(batch_src))
            mark_done(done_flag)
