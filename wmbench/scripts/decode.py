from __future__ import annotations

import os
import warnings

import click
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import torch
import torch.multiprocessing as mp
from tqdm.auto import tqdm

try:
    import dotenv

    dotenv.load_dotenv(override=False)
except ModuleNotFoundError:  # pragma: no cover
    pass

from wmbench.dev import (
    LIMIT,
    SUBSET_LIMIT,
    WATERMARK_METHODS,
    DCT_WATERMARK_LEN,
    DCT_WATERMARK_ALPHA,
    DATASET_NAMES,
    check_file_existence,
    existence_operation,
    existence_to_indices,
    parse_image_dir_path,
    save_json,
    load_json,
    encode_array_to_string,
)
from wmbench.utils.dct_utils import extract_bits

warnings.filterwarnings("ignore")


def _selected_decode_modes() -> list[str]:
    raw = os.environ.get("WAVES_DECODE_MODES", "").strip()
    if not raw:
        return list(WATERMARK_METHODS.keys())
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    invalid = sorted(set(requested).difference(WATERMARK_METHODS.keys()))
    if invalid:
        raise ValueError(f"Invalid WAVES_DECODE_MODES values: {invalid}")
    deduped: list[str] = []
    seen: set[str] = set()
    for mode in requested:
        if mode not in seen:
            seen.add(mode)
            deduped.append(mode)
    return deduped


def _parse_image_path_any(path: str, quiet: bool = True):
    """Parse image directory path under DATA_DIR or RESULT_DIR."""
    try:
        return parse_image_dir_path(path, quiet=quiet)
    except ValueError:
        result_root = os.environ.get("RESULT_DIR")
        if not result_root:
            # Heuristic parse for result paths when RESULT_DIR is not set.
            normalized_path = os.path.normpath(str(path))
            dataset_name = os.path.basename(os.path.dirname(normalized_path))
            dirname = os.path.basename(normalized_path)
            if dataset_name in DATASET_NAMES:
                if dirname == "real":
                    return dataset_name, None, None, "real"
                parts = dirname.split("-")
                if len(parts) == 3:
                    attack_name, attack_strength, source_name = parts
                    try:
                        attack_strength = float(attack_strength)
                    except ValueError as exc:
                        raise ValueError("Attack strength must be a number") from exc
                    return dataset_name, attack_name, attack_strength, source_name
            raise
        normalized_root = os.path.normpath(result_root)
        normalized_path = os.path.normpath(str(path))
        if os.path.commonpath([normalized_root, normalized_path]) != os.path.commonpath([normalized_root]):
            raise

        dataset_name = os.path.basename(os.path.dirname(normalized_path))
        dirname = os.path.basename(normalized_path)
        if dirname in {"real", "images"}:
            return dataset_name, None, None, "real"
        parts = dirname.split("-")
        if len(parts) == 3:
            attack_name, attack_strength, source_name = parts
            try:
                attack_strength = float(attack_strength)
            except ValueError as exc:
                raise ValueError("Attack strength must be a number") from exc
            return dataset_name, attack_name, attack_strength, source_name
        raise


def _result_root_for_path(path: str) -> str:
    result_root = os.environ.get("RESULT_DIR")
    if result_root:
        return result_root
    # Assume path is .../<result_root>/<dataset>/<dir>
    normalized_path = os.path.normpath(str(path))
    dataset_dir = os.path.dirname(normalized_path)
    return os.path.dirname(dataset_dir)


def _result_stem(path: str) -> tuple[str, str]:
    dataset_name, attack_name, attack_strength, source_name = _parse_image_path_any(path, quiet=True)
    if attack_name is None or attack_strength is None:
        return dataset_name, source_name
    return dataset_name, f"{attack_name}-{attack_strength}-{source_name}"


def get_indices(mode: str, path: str, quiet: bool, subset: bool, limit: int, subset_limit: int) -> list[int]:
    dataset_name, stem = _result_stem(path)
    result_root = _result_root_for_path(path)
    json_path = os.path.join(result_root, dataset_name, f"{stem}-decode.json")
    if os.path.exists(json_path) and (data := load_json(json_path)) is not None:
        decoded_existences = [data[str(i)][mode] is not None for i in range(limit)]
        if (not subset and sum(decoded_existences) == limit) or (
            subset and sum(decoded_existences[:subset_limit]) == subset_limit
        ):
            return []

    image_existences = check_file_existence(path, name_pattern="{}.png", limit=limit)
    reversed_latents_existences = check_file_existence(path, name_pattern="{}_reversed.pkl", limit=limit)
    if not quiet:
        print(f"Found {sum(image_existences)} images, and {sum(reversed_latents_existences)} reversed latents")

    if mode == "tree_ring":
        existences = reversed_latents_existences
    elif mode in ["stable_sig", "stegastamp", "dct"]:
        existences = image_existences
    else:
        raise ValueError(f"Unknown decode mode {mode!r}")

    if not os.path.exists(json_path):
        indices = existence_to_indices(existences, limit=limit if not subset else subset_limit)
    else:
        indices = existence_to_indices(
            existence_operation(existences, decoded_existences, op="difference"),
            limit=limit if not subset else subset_limit,
        )
    return indices


def _onnx_session(model_path: str, gpu: int) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.log_severity_level = 3

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider"],
            provider_options=[{"device_id": str(gpu)}],
            sess_options=session_options,
        )
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"], sess_options=session_options)


def init_model(mode: str, gpu: int):
    if mode == "tree_ring":
        size = 64
        radius = 10
        channel = 3
        mask = torch.zeros((1, 4, size, size), dtype=torch.bool)
        x0 = y0 = size // 2
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        mask[:, channel] = torch.tensor(((x - x0) ** 2 + (y - y0) ** 2) <= radius**2)
        return mask

    if mode == "stable_sig":
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir:
            raise RuntimeError("MODEL_DIR is not set")
        return _onnx_session(os.path.join(model_dir, "stable_signature.onnx"), gpu)

    if mode == "stegastamp":
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir:
            raise RuntimeError("MODEL_DIR is not set")
        return _onnx_session(os.path.join(model_dir, "stega_stamp.onnx"), gpu)

    if mode == "dct":
        return {"length": DCT_WATERMARK_LEN, "alpha": DCT_WATERMARK_ALPHA}

    raise ValueError(f"Unknown decode mode {mode!r}")


def load_files(mode: str, path: str, indices: list[int]):
    if mode == "tree_ring":
        return torch.cat(
            [torch.load(os.path.join(path, f"{idx}_reversed.pkl"), map_location="cpu") for idx in indices],
            dim=0,
        )

    if mode == "stable_sig":
        return np.stack(
            [
                (
                    (
                        np.array(Image.open(os.path.join(path, f"{idx}.png")), dtype=np.float32) / 255.0
                        - [0.485, 0.456, 0.406]
                    )
                    / [0.229, 0.224, 0.225]
                )
                .transpose((2, 0, 1))
                .astype(np.float32)
                for idx in indices
            ],
            axis=0,
        )

    if mode == "stegastamp":
        return np.stack(
            [
                np.array(ImageOps.fit(Image.open(os.path.join(path, f"{idx}.png")), (400, 400)), dtype=np.float32)
                / 255.0
                for idx in indices
            ],
            axis=0,
        )

    if mode == "dct":
        return np.stack(
            [np.array(Image.open(os.path.join(path, f"{idx}.png")).convert("L"), dtype=np.float64) for idx in indices],
            axis=0,
        )

    raise ValueError(f"Unknown decode mode {mode!r}")


def decode(mode: str, model, gpu: int, inputs):
    if mode == "tree_ring":
        device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
        fft_latents = torch.fft.fftshift(torch.fft.fft2(inputs.to(device)), dim=(-1, -2))
        messages = torch.stack(
            [fft_latents[i].unsqueeze(0)[model].flatten() for i in range(fft_latents.shape[0])],
            dim=0,
        )
        return torch.concatenate([messages.real, messages.imag], dim=1).cpu().numpy()

    if mode == "stable_sig":
        outputs = model.run(None, {"image": inputs})
        return (outputs[0] > 0).astype(bool)

    if mode == "stegastamp":
        outputs = model.run(None, {"image": inputs, "secret": np.zeros((inputs.shape[0], 100), dtype=np.float32)})
        return outputs[2].astype(bool)

    if mode == "dct":
        raise RuntimeError("DCT decode uses the per-folder path decoder, not batched ONNX decode")

    raise ValueError(f"Unknown decode mode {mode!r}")


def _get_real_dir_for_path(path: str) -> str:
    dataset_name, _, _, _ = _parse_image_path_any(path, quiet=True)
    structured = os.path.join(os.environ.get("DATA_DIR"), "main", dataset_name, "real")
    if os.path.isdir(structured):
        return structured
    flat_images = os.path.join(os.environ.get("DATA_DIR"), "images")
    if os.path.isdir(flat_images):
        return flat_images
    return os.path.normpath(os.environ.get("DATA_DIR"))


def process_dct(indices: list[int], path: str, quiet: bool):
    real_dir = _get_real_dir_for_path(path)
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Missing real image directory required for DCT decoding: {real_dir}")

    results: dict[int, str] = {}
    iterator = tqdm(indices, desc="Decoding DCT messages", unit="img") if not quiet else indices
    for idx in iterator:
        original = np.array(Image.open(os.path.join(real_dir, f"{idx}.png")).convert("L"), dtype=np.float64)
        candidate_pil = Image.open(os.path.join(path, f"{idx}.png")).convert("L")
        if candidate_pil.size != (original.shape[1], original.shape[0]):
            candidate_pil = candidate_pil.resize((original.shape[1], original.shape[0]), Image.Resampling.BICUBIC)
        candidate = np.array(candidate_pil, dtype=np.float64)
        decoded_bits = extract_bits(
            original_image=original,
            candidate_image=candidate,
            length=DCT_WATERMARK_LEN,
            alpha=DCT_WATERMARK_ALPHA,
        )
        results[idx] = encode_array_to_string(decoded_bits)
    return results


def worker(mode: str, gpu: int, path: str, indices: list[int], lock, counter, results):
    model = init_model(mode, gpu)
    batch_size = {"tree_ring": 32, "stable_sig": 4, "stegastamp": 4}[mode]
    for it in range(0, len(indices), batch_size):
        batch_indices = indices[it : min(it + batch_size, len(indices))]
        inputs = load_files(mode, path, batch_indices)
        messages = decode(mode, model, gpu, inputs)
        with lock:
            counter.value += inputs.shape[0]
            for idx, message in zip(batch_indices, messages):
                results[idx] = encode_array_to_string(message)


def process(mode: str, indices: list[int], path: str, quiet: bool):
    if mode == "dct":
        return process_dct(indices, path, quiet)

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0 and mode in {"stable_sig", "stegastamp"}:
        raise RuntimeError("No GPUs available for ONNX CUDA decode")

    if not quiet:
        print(f"Using {max(num_gpus, 1)} device(s) for processing")

    num_workers = {
        "tree_ring": max(num_gpus, 1),
        "stable_sig": max(num_gpus, 1),
        "stegastamp": max(num_gpus, 1) * 2,
    }[mode]

    chunk_size = len(indices) // num_workers if num_workers > 0 else len(indices)
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()

        processes = []
        for rank in range(num_workers):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_workers - 1 else (rank + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    mode,
                    rank % max(num_gpus, 1),
                    path,
                    indices[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(indices), desc="Decoding images or reversed latents", unit="file") as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break

        for p in processes:
            p.join()

        return dict(results)


def report(mode: str, path: str, results: dict[int, str], quiet: bool, limit: int) -> None:
    dataset_name, stem = _result_stem(path)
    result_root = _result_root_for_path(path)
    json_path = os.path.join(result_root, dataset_name, f"{stem}-decode.json")

    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {
                _mode: results.get(idx) if mode == _mode else None for _mode in WATERMARK_METHODS.keys()
            }
    else:
        for idx, message in results.items():
            data[str(idx)][mode] = message

    save_json(data, json_path)
    if not quiet:
        print(f"Decoded messages saved to {json_path}")


def _collect_image_dirs(root: str, quiet: bool) -> list[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Missing directory: {root}")
    paths: list[str] = []
    for name in os.listdir(root):
        candidate = os.path.join(root, name)
        if not os.path.isdir(candidate):
            continue
        try:
            _parse_image_path_any(candidate, quiet=True)
        except Exception:
            continue
        paths.append(candidate)
    if not paths and not quiet:
        print(f"No valid image directories found under {root}")
    return sorted(paths)


def single_mode(mode: str, path: str, quiet: bool, subset: bool, limit: int, subset_limit: int) -> None:
    if not quiet:
        print(f"Decoding {mode} messages")
    indices = get_indices(mode, path, quiet, subset, limit, subset_limit)
    if len(indices) == 0:
        if not quiet:
            print("All messages requested already decoded")
        return
    results = process(mode, indices, path, quiet)
    report(mode, path, results, quiet, limit)


@click.command()
@click.option("--path", "-p", type=str, default=os.getcwd(), help="Path to image directory")
@click.option("--all", "-a", is_flag=True, default=False, help="Decode all image directories under --path")
@click.option("--include-clean", is_flag=True, default=False, help="Also decode the clean images folder as source 'real'.")
@click.option("--dry", "-d", is_flag=True, default=False, help="Dry run")
@click.option("--subset", "-s", is_flag=True, default=False, help="Run on subset only")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Quiet mode")
def main(
    path: str,
    all: bool,
    include_clean: bool,
    dry: bool,
    subset: bool,
    quiet: bool,
    limit: int = LIMIT,
    subset_limit: int = SUBSET_LIMIT,
):
    if dry:
        return

    decode_modes = _selected_decode_modes()
    paths = _collect_image_dirs(path, quiet) if all else [path]

    for target in paths:
        _, attack_name, _, source_name = _parse_image_path_any(target, quiet=quiet)

        if attack_name is None and source_name == "real" and not include_clean:
            continue

        if source_name == "real":
            for mode in decode_modes:
                single_mode(mode, target, quiet, subset, limit, subset_limit)
            continue

        for mode in decode_modes:
            if source_name.endswith(mode):
                single_mode(mode, target, quiet, subset, limit, subset_limit)
                break
        else:
            raise ValueError(f"Invalid source name {source_name} encountered")


if __name__ == "__main__":
    main()
