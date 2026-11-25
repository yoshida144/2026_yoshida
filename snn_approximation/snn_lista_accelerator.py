"""SNN-assisted penultimate update for a frozen LISTA classifier."""
import argparse
import hashlib
import json
import re
from itertools import count
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import ML_CSC_Image_Classification
from snn_classifier import lif_step


SYSTEM_DIR = Path(__file__).resolve().parent
ROOT = SYSTEM_DIR.parent
CONFIG_DIR = SYSTEM_DIR / "json"
DATA_DIR = ROOT / "dataset"


class SpikingUpdate(nn.Module):
    """Predict a correction from the current code and the last LISTA displacement."""

    def __init__(self, channels):
        super().__init__()
        self.input = nn.Conv2d(2 * channels, 16, 1)
        self.output = nn.Conv2d(16, channels, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.register_buffer("code_scale", torch.tensor(1.0))

    def forward(self, previous, current):
        delta = current - previous
        normalized = torch.cat((current, delta), dim=1) / self.code_scale
        drive = self.input(normalized)
        first, membrane = lif_step(drive, 0.0)
        second, _ = lif_step(drive, membrane)
        correction = self.output((first + second) * 0.5) * self.code_scale
        return (current + self.scale * delta + correction).relu()


class AcceleratedLISTA(nn.Module):
    """Replace selected LISTA updates and retain exact refinement in each layer."""

    def __init__(self, model, skipped_updates=1, approximate_layers=None):
        super().__init__()
        self.model = model.requires_grad_(False).eval()
        layers = model.feature_extractor.layers
        self.approximate_layers = tuple(approximate_layers or [len(layers)])
        self.predictors = nn.ModuleDict({
            str(number): SpikingUpdate(layers[number - 1].out_channels)
            for number in self.approximate_layers
        })
        self.skipped_updates = skipped_updates

    def run_codes(self, images, shortcuts=None):
        layers = self.model.feature_extractor.layers
        for index, layer in enumerate(layers):
            shortcut = shortcuts[str(index + 1)] if shortcuts is not None and str(index + 1) in shortcuts else None
            images, _ = layer(images, return_reconstruction=False,
                                      predict_penultimate=shortcut, skipped_updates=self.skipped_updates)
        return images

    def logits(self, code):
        return self.model.final_classifier(code.amax(dim=(2, 3)))

    def forward(self, images):
        return self.logits(self.run_codes(images, self.predictors))


@torch.no_grad()
def gather_traces(model, loader, device):
    samples = {name: [] for name in model.predictors}
    for images, _ in loader:
        features = images.to(device, memory_format=torch.channels_last)
        for number, layer in enumerate(model.model.feature_extractor.layers, start=1):
            name = str(number)
            trace = [] if name in samples else None
            features, _ = layer(features, return_reconstruction=False, trace=trace)
            if trace is not None:
                index = len(trace) - 2 - model.skipped_updates
                current = trace[index]
                previous = trace[index - 1] if index else torch.zeros_like(current)
                samples[name].append((previous.cpu(), current.cpu(), trace[-2].cpu()))
    return samples


def fit(predictor, samples, epochs, device, *, learning_rate, seed, skipped_updates, layer):
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)
    scale = sum(target.square().mean().item() for _, _, target in samples) / len(samples)
    predictor.code_scale.fill_(max(scale ** 0.5, 0.001))
    if skipped_updates == 2:
        cross = sum((current * target).sum(dim=(0, 2, 3))
                    for _, current, target in samples)
        energy = sum(current.square().sum(dim=(0, 2, 3)) for _, current, _ in samples)
        with torch.no_grad():
            gain = cross / energy.clamp_min(1e-8)
            predictor.scale.copy_((gain - 1.0).reshape(1, -1, 1, 1))
    shuffle = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        losses = []
        for index in torch.randperm(len(samples), generator=shuffle).tolist():
            previous, current, target = (x.to(device, memory_format=torch.channels_last) for x in samples[index])
            optimizer.zero_grad(set_to_none=True)
            predicted = predictor(previous, current)
            loss = F.mse_loss(predicted / predictor.code_scale, target / predictor.code_scale)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Layer {layer} predictor epoch {epoch + 1}/{epochs}: scaled MSE {sum(losses)/len(losses):.5f}")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(model, loader, device, *, convergence=False, progress=False):
    model.eval()
    first_images, _ = next(iter(loader))
    first_images = first_images.to(device, memory_format=torch.channels_last)
    model(first_images)
    model.logits(model.run_codes(first_images))
    results = {"original": [], "fast": [], "short": [], "extrapolation": []}
    times = {"original": 0.0, "fast": 0.0}
    labels = []
    errors = {"short": [], "fast": [], "extrapolation": []}

    def timed(name, operation):
        synchronize(device)
        start = perf_counter()
        result = operation()
        synchronize(device)
        times[name] += perf_counter() - start
        return result

    for index, (images, target) in enumerate(loader):
        images = images.to(device, memory_format=torch.channels_last)
        operations = {
            "original": lambda: model.run_codes(images),
            "fast": lambda: model.run_codes(images, model.predictors),
        }
        order = ("original", "fast") if index % 2 == 0 else ("fast", "original")
        codes = {}
        logits = {}
        for name in order:
            def forward():
                code = operations[name]()
                return code, model.logits(code)
            codes[name], logits[name] = timed(name, forward)
        original, fast = logits["original"], logits["fast"]
        results["original"].append(original.argmax(1).cpu())
        results["fast"].append(fast.argmax(1).cpu())
        labels.append(target)
        if convergence:
            shortcuts = {
                "short": lambda prev, curr: curr,
                "extrapolation": lambda prev, curr: (curr + model.skipped_updates * (curr - prev)).relu(),
            }
            for name, shortcut in shortcuts.items():
                codes[name] = model.run_codes(images, dict.fromkeys(model.predictors, shortcut))
                results[name].append(model.logits(codes[name]).argmax(1).cpu())
            reference = codes["original"]
            norm = reference.flatten(1).norm(dim=1).clamp_min(1e-8)
            for name in errors:
                errors[name].append((codes[name] - reference).flatten(1).norm(dim=1).div(norm).cpu())
        if progress and (index + 1) % 64 == 0:
            print(f"Evaluated {(index + 1) * loader.batch_size}/{len(loader.dataset)} images")
    return ({name: torch.cat(values) for name, values in results.items() if values},
            torch.cat(labels), times,
            {name: torch.cat(values).mean().item() for name, values in errors.items() if values})


def reserve_output_paths(metrics_dir, weights_dir):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    used = []
    for directory, pattern in ((metrics_dir, r"metrics([0-9]*)[.](json|pending)"),
                               (weights_dir, r"snn_update([0-9]*)[.]pth")):
        if directory.is_dir():
            for path in directory.iterdir():
                match = re.fullmatch(pattern, path.name)
                if match:
                    used.append(int(match.group(1) or 1))
    for number in count(max(used, default=0) + 1):
        suffix = "" if number == 1 else str(number)
        metrics_path = metrics_dir / f"metrics{suffix}.json"
        weights_path = weights_dir / f"snn_update{suffix}.pth"
        reservation = metrics_path.with_suffix(".pending")
        try:
            reservation.touch(exist_ok=False)
        except FileExistsError:
            continue
        if not metrics_path.exists() and not weights_path.exists():
            return metrics_path, weights_path, reservation
        reservation.unlink()


def main():
    common = json.loads((CONFIG_DIR / "model_config.json").read_text(encoding="utf-8"))
    training_config = json.loads((CONFIG_DIR / "snn_training_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((CONFIG_DIR / "evaluation_config.json").read_text(encoding="utf-8"))
    seed = common["seed"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save results and newly trained weights to a custom directory")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a trained SNN update without refitting it")
    parser.add_argument("--fit-images", type=int, default=training_config["fit_images"])
    parser.add_argument("--test-images", type=int, default=evaluation["test_images"])
    parser.add_argument("--skipped-updates", type=int, choices=(1, 2), default=common["skipped_updates"])
    parser.add_argument("--approximate-layers", nargs="+", type=int, metavar="LAYER")
    parser.add_argument("--skip-ablation", action=argparse.BooleanOptionalAction, default=evaluation["skip_ablation"])
    parser.add_argument("--epochs", type=int, default=training_config["epochs"])
    parser.add_argument("--threads", type=int, default=common["threads"])
    args = parser.parse_args()
    if min(args.fit_images, args.test_images, args.epochs, args.threads) < 1:
        parser.error("Image, epoch and thread counts must be positive")
    if args.skipped_updates not in (1, 2):
        parser.error("skipped_updates must be 1 or 2")
    if min(training_config["batch_size"], evaluation["batch_size"]) < 1:
        parser.error("Batch sizes must be positive")
    if not training_config["learning_rate"] > 0:
        parser.error("Learning rate must be positive")
    if common["target_dataset"] != "CIFAR10":
        parser.error("This accelerator supports CIFAR10")
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = ROOT / common["teacher_directory"]
    config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    info = json.loads((DATA_DIR / "datasets_info.json").read_text(encoding="utf-8"))["CIFAR10"]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(info["data_mean"], info["data_std"])])
    training = datasets.CIFAR10(DATA_DIR, train=True, transform=transform)
    test = datasets.CIFAR10(DATA_DIR, train=False, transform=transform)
    if args.fit_images > len(training) or args.test_images > len(test):
        parser.error("Selection exceeds the official split size")
    generator = torch.Generator().manual_seed(seed)
    train_ids = torch.randperm(len(training), generator=generator).tolist()
    test_ids = torch.randperm(len(test), generator=generator).tolist()[:args.test_images]
    fit_loader = DataLoader(Subset(training, train_ids[:args.fit_images]), batch_size=training_config["batch_size"])
    test_loader = DataLoader(Subset(test, test_ids), batch_size=evaluation["batch_size"])
    teacher = ML_CSC_Image_Classification(
        info["input_channels"], info["num_classes"], config["layer_specs"], method=config["method"]
    )
    teacher.load_state_dict(torch.load(base / "model.pth", map_location="cpu", weights_only=True))
    teacher_hash = hashlib.sha256((base / "model.pth").read_bytes()).hexdigest()
    layers = teacher.feature_extractor.layers
    weights = None
    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if weights["teacher_sha256"] != teacher_hash:
            parser.error("The accelerator was trained with different LISTA weights")
        if weights.get("skipped_updates", 1) != args.skipped_updates:
            parser.error("The accelerator checkpoint uses a different skipped-update count")
        saved_layers = tuple(weights.get("approximate_layers", [len(layers)]))
        if args.approximate_layers is not None and tuple(sorted(args.approximate_layers)) != saved_layers:
            parser.error("The requested layers do not match the checkpoint")
        approximate_layers = saved_layers
    else:
        approximate_layers = tuple(sorted(args.approximate_layers or common.get("approximate_layers", [len(layers)])))
    if (not approximate_layers or len(set(approximate_layers)) != len(approximate_layers)
            or any(not 1 <= number <= len(layers) for number in approximate_layers)):
        parser.error("Approximate layers must be distinct, valid 1-based layer numbers")
    if any(args.skipped_updates > layers[number - 1].T - 2 for number in approximate_layers):
        parser.error("Each approximated layer needs at least one exact update before and after the shortcut")
    model = AcceleratedLISTA(teacher, args.skipped_updates, approximate_layers).to(
        device=device, memory_format=torch.channels_last
    )
    model.eval()
    probe, _ = next(iter(fit_loader))
    probe = probe.to(device, memory_format=torch.channels_last)
    with torch.inference_mode():
        torch.testing.assert_close(model.logits(model.run_codes(probe)), teacher(probe, return_reconstructions=False)[0])
    if weights is not None:
        if "predictors" in weights:
            model.predictors.load_state_dict(weights["predictors"])
        else:
            model.predictors[str(len(layers))].load_state_dict(weights["predictor"])
    else:
        print(f"Collecting {args.fit_images} frozen LISTA traces on {device}")
        samples = gather_traces(model, fit_loader, device)
        for layer, predictor in model.predictors.items():
            fit(predictor, samples.pop(layer), args.epochs, device,
                learning_rate=training_config["learning_rate"], seed=seed,
                skipped_updates=args.skipped_updates, layer=layer)
    report = {"device": device.type, "threads": args.threads, "seed": seed,
              "fit_images": args.fit_images,
              "test_images": args.test_images, "epochs": args.epochs,
              "test_batch_size": evaluation["batch_size"],
              "original_updates": sum(layer.T for layer in teacher.feature_extractor.layers),
              "approximate_layers": list(approximate_layers),
              "skipped_updates_per_layer": args.skipped_updates,
              "fast_updates": sum(layer.T for layer in layers) - args.skipped_updates * len(approximate_layers)}
    metrics_dir = args.output or ROOT / "result" / "json"
    weights_dir = args.output or ROOT / "models" / "snn_approximation"
    metrics_path, weights_path, reservation = reserve_output_paths(metrics_dir, weights_dir)
    if args.checkpoint is None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with weights_path.open("xb") as checkpoint:
            torch.save({"predictors": model.predictors.state_dict(),
                        "approximate_layers": list(approximate_layers),
                        "teacher_sha256": teacher_hash, "skipped_updates": args.skipped_updates}, checkpoint)
        print(f"Saved SNN weights: {weights_path}")
    report["snn_checkpoint"] = str((args.checkpoint or weights_path).resolve())
    print(f"Benchmarking {args.test_images} test images")
    preds, labels, times, errors = benchmark(
        model, test_loader, device, convergence=not args.skip_ablation, progress=True
    )
    delta = (preds["fast"] == labels).float() - (preds["original"] == labels).float()
    mean_delta = delta.mean().item()
    half_width = 1.96 * delta.std(unbiased=len(delta) > 1).item() / len(delta) ** 0.5
    report["fast_accuracy_delta"] = mean_delta
    report["fast_accuracy_delta_approx_95_ci"] = [mean_delta - half_width, mean_delta + half_width]
    report["accuracy"] = {name: (values == labels).float().mean().item() for name, values in preds.items()}
    report["forward_seconds"] = times
    report["fast_speedup"] = times["original"] / times["fast"]
    report["relative_code_l2_error"] = errors
    metrics_json = json.dumps(report, indent=2, allow_nan=False)
    with metrics_path.open("x", encoding="utf-8") as metrics_file:
        metrics_file.write(metrics_json + "\n")
    reservation.unlink()
    print(metrics_json)
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
