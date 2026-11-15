import argparse
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.distilbert_model import (
    build_distilbert_model,
    compute_training_class_weights,
    create_label_mapping,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id, max_length):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        self.labels = [
            label2id[label]
            for label in labels
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if (
        state.get("cuda") is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_name(path.name + ".tmp")

    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_split(task_dir, split_name):
    file_path = task_dir / f"{split_name}.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{file_path} must contain columns: {required_columns}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return df


def evaluate_model(model, dataloader, device, class_names):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels")

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(**batch)

            predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.numpy().tolist()
            )

    class_ids = list(range(len(class_names)))

    return {
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            all_labels,
            all_predictions,
            labels=class_ids,
        ),
    }


def save_best_outputs(
    model,
    tokenizer,
    evaluation,
    task_name,
    class_names,
    label2id,
    id2label,
    epoch,
    config,
    train_size,
    validation_size,
    device,
    output_dir,
):
    output_dir = Path(output_dir)

    model_dir = output_dir / "models" / f"{task_name}_distilbert"
    report_dir = output_dir / "metrics"

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(
        model_dir / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {
                    str(key): value
                    for key, value in id2label.items()
                },
            },
            file,
            indent=4,
        )

    metrics = {
        "task": task_name,
        "model": "DistilBERT",
        "pretrained_model": config["model_name"],
        "training_samples": train_size,
        "validation_samples": validation_size,
        "number_of_classes": len(class_names),
        "best_epoch": epoch,
        "accuracy": evaluation["accuracy"],
        "macro_f1": evaluation["macro_f1"],
        "weighted_f1": evaluation["weighted_f1"],
        "max_length": config["max_length"],
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config[
            "gradient_accumulation_steps"
        ],
        "learning_rate": config["learning_rate"],
        "device": str(device),
    }

    with open(
        report_dir / f"{task_name}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        report_dir / f"{task_name}_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation["classification_report"],
            file,
            indent=4,
        )

    matrix_df = pd.DataFrame(
        evaluation["confusion_matrix"],
        index=class_names,
        columns=class_names,
    )

    matrix_df.to_csv(
        report_dir / f"{task_name}_confusion_matrix.csv"
    )


def train_task(
    task_name,
    config,
    checkpoint_root,
    output_dir,
    resume=True,
    checkpoint_every=300,
):
    print("\n" + "=" * 70)
    print(f"DISTILBERT TRAINING: {task_name.upper()}")
    print("=" * 70)

    set_seed(config["random_state"])

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cpu":
        torch.set_num_threads(config["cpu_threads"])
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    task_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / task_name
    )

    train_df = load_split(task_dir, "train")
    validation_df = load_split(task_dir, "validation")

    train_texts = train_df["text"]
    train_labels = train_df["label"]

    validation_texts = validation_df["text"]
    validation_labels = validation_df["label"]

    label2id, id2label = create_label_mapping(
        train_labels.tolist()
    )

    if set(validation_labels.unique()) != set(label2id):
        raise ValueError(
            f"{task_name}: training and validation labels differ."
        )

    class_names = [
        id2label[index]
        for index in range(len(id2label))
    ]

    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Classes: {len(class_names)}")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )

    print("Tokenizing datasets...")

    train_dataset = ComplaintDataset(
        train_texts,
        train_labels,
        tokenizer,
        label2id,
        config["max_length"],
    )

    validation_dataset = ComplaintDataset(
        validation_texts,
        validation_labels,
        tokenizer,
        label2id,
        config["max_length"],
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=config["num_workers"],
    )

    model = build_distilbert_model(
        config["model_name"],
        label2id,
        id2label,
    )

    model.to(device)

    class_weights = compute_training_class_weights(
        train_labels.tolist(),
        label2id,
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    accumulation_steps = config[
        "gradient_accumulation_steps"
    ]

    steps_per_epoch = math.ceil(
        math.ceil(len(train_dataset) / config["batch_size"])
        / accumulation_steps
    )

    total_steps = steps_per_epoch * config["epochs"]

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            total_steps * config["warmup_ratio"]
        ),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    checkpoint_dir = (
        Path(checkpoint_root) / task_name
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "latest.pt"

    start_epoch = 0
    resume_batch = 0
    global_step = 0
    best_macro_f1 = -1.0
    best_metrics = None
    history = []
    resume_rng_state = None
    
    if resume and latest_checkpoint.exists():
        print("\nResuming from checkpoint...")

        checkpoint = torch.load(
            latest_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"]
        resume_batch = checkpoint["batch_in_epoch"]
        global_step = checkpoint["global_step"]
        best_macro_f1 = checkpoint["best_macro_f1"]
        best_metrics = checkpoint.get("best_metrics")
        history = checkpoint["history"]

        resume_rng_state = checkpoint["rng_state"]

        print(
            f"Resuming at epoch {start_epoch + 1}, "
            f"after batch {resume_batch}"
        )

        del checkpoint

    else:
        print("\nStarting a new training run.")

    if start_epoch >= config["epochs"]:
        print("Training already completed. Loading saved metrics.")

        metrics_path = (
            Path(output_dir)
            / "metrics"
            / f"{task_name}_metrics.json"
        )

        if not metrics_path.exists():
            raise FileNotFoundError(
                "Training checkpoint indicates completion, "
                "but the best-model metrics file is missing."
            )

        with open(metrics_path, "r", encoding="utf-8") as file:
            return json.load(file)

    model.train()

    for epoch in range(start_epoch, config["epochs"]):
        print(
            f"\nEpoch {epoch + 1}/{config['epochs']}"
        )

        generator = torch.Generator()
        generator.manual_seed(
            config["random_state"] + epoch
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            generator=generator,
            collate_fn=collator,
            num_workers=config["num_workers"],
            pin_memory=use_amp,
        )

        batches_to_skip = (
            resume_batch if epoch == start_epoch else 0
        )

        iterator = iter(train_loader)

        for _ in range(batches_to_skip):
            next(iterator)

        if (
            epoch == start_epoch
            and batches_to_skip > 0
            and resume_rng_state is not None
        ):
            restore_rng_state(resume_rng_state)
            resume_rng_state = None

        model.train()
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        processed_batches = batches_to_skip

        for batch_index, batch in enumerate(
            iterator,
            start=batches_to_skip,
        ):
            labels = batch.pop("labels").to(device)

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                outputs = model(**batch)

                loss = loss_function(
                    outputs.logits,
                    labels,
                )

            total_loss += loss.item()
            scaler.scale(
                loss / accumulation_steps
            ).backward()

            processed_batches = batch_index + 1

            should_update = (
                processed_batches % accumulation_steps == 0
                or processed_batches == len(train_loader)
            )

            if should_update:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["max_grad_norm"],
                )

                old_scale = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                if scaler.get_scale() >= old_scale:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % checkpoint_every == 0:
                    print(
                        f"Step {global_step} | "
                        f"Epoch {epoch + 1} | "
                        f"Batch {processed_batches} | "
                        f"Loss: {loss.item():.4f}"
                    )

                    atomic_save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "epoch": epoch,
                            "batch_in_epoch": processed_batches,
                            "global_step": global_step,
                            "best_macro_f1": best_macro_f1,
                            "best_metrics": best_metrics,
                            "history": history,
                            "rng_state": save_rng_state(),
                        },
                        latest_checkpoint,
                    )

        average_loss = total_loss / max(
            1,
            len(train_loader) - batches_to_skip,
        )

        print(f"Training loss: {average_loss:.4f}")
        print("Running validation...")

        evaluation = evaluate_model(
            model,
            validation_loader,
            device,
            class_names,
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "training_loss": average_loss,
            "validation_accuracy": evaluation["accuracy"],
            "validation_macro_f1": evaluation["macro_f1"],
            "validation_weighted_f1": evaluation["weighted_f1"],
        }

        history.append(epoch_metrics)

        print(
            f"Accuracy: {evaluation['accuracy']:.4f}"
        )
        print(
            f"Macro F1: {evaluation['macro_f1']:.4f}"
        )
        print(
            f"Weighted F1: {evaluation['weighted_f1']:.4f}"
        )

        if evaluation["macro_f1"] > best_macro_f1:
            best_macro_f1 = evaluation["macro_f1"]

            best_metrics = {
                "accuracy": evaluation["accuracy"],
                "macro_f1": evaluation["macro_f1"],
                "weighted_f1": evaluation["weighted_f1"],
                "best_epoch": epoch + 1,
            }

            print("Saving new best model...")

            save_best_outputs(
                model,
                tokenizer,
                evaluation,
                task_name,
                class_names,
                label2id,
                id2label,
                epoch + 1,
                config,
                len(train_df),
                len(validation_df),
                device,
                output_dir,
            )

        # Save a resumable checkpoint at the end of each epoch.
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch + 1,
                "batch_in_epoch": 0,
                "global_step": global_step,
                "best_macro_f1": best_macro_f1,
                "best_metrics": best_metrics,
                "history": history,
                "rng_state": save_rng_state(),
            },
            latest_checkpoint,
        )

        resume_batch = 0

        print(
            f"Checkpoint saved: {latest_checkpoint}"
        )

        del train_loader
        gc.collect()

    history_path = (
        Path(output_dir)
        / "metrics"
        / f"{task_name}_training_history.json"
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("\nTraining completed.")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")

    del model
    del tokenizer
    del train_dataset
    del validation_dataset
    del validation_loader

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics_path = (
        Path(output_dir)
        / "metrics"
        / f"{task_name}_metrics.json"
    )

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["product", "issue"],
        required=True,
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=str(PROJECT_ROOT / "checkpoints"),
    )

    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "distilbert_output"),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")

    config = load_config()["distilbert"]

    results = train_task(
        task_name=args.task,
        config=config,
        checkpoint_root=args.checkpoint_dir,
        output_dir=args.output_dir,
        resume=args.resume and not args.no_resume,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n" + "=" * 70)
    print(f"{args.task.upper()} TRAINING RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()