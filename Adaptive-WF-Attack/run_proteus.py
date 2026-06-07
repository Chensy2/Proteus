import argparse
import importlib
import json
import os


def load_model(module_name, class_name, num_classes, kwargs_json):
    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    try:
        return model_cls(num_classes=num_classes, **kwargs)
    except TypeError:
        try:
            return model_cls(num_classes, **kwargs)
        except TypeError:
            return model_cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run Proteus adaptation with an external PyTorch attack model."
    )
    parser.add_argument("--train", required=True, help="Source train split: npz/json/jsonl/directory.")
    parser.add_argument("--target", required=True, help="Target split: npz/json/jsonl/directory.")
    parser.add_argument("--feature", default="DIR", choices=["DIR", "DT", "DT2", "Origin"])
    parser.add_argument("--view", default="dir", choices=["dir", "ps", "iat"],
                        help="JSON sequence view when --train/--target are JSON inputs.")
    parser.add_argument("--seq_len", type=int, default=5000)
    parser.add_argument("--label_key", default=None)
    parser.add_argument("--sequence_key", default=None)

    parser.add_argument("--model_module", required=True,
                        help="Import path for the external model module, e.g. my_repo.models.df.")
    parser.add_argument("--model_class", required=True, help="Class name in --model_module.")
    parser.add_argument("--model_kwargs", default="{}", help="JSON object passed to the model constructor.")
    parser.add_argument("--checkpoint", required=True, help="Source-trained model checkpoint.")
    parser.add_argument("--checkpoint_key", default=None,
                        help="Optional key for state_dict inside checkpoint dict.")
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--history_json", default=None)

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gmm_threshold", type=float, default=0.6)
    parser.add_argument("--pseudo_refresh_epochs", type=int, default=5)
    parser.add_argument("--mmd_weight", type=float, default=1.0)
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_weight", type=float, default=1.0)
    parser.add_argument("--source_ce_weight", type=float, default=1.0)
    args = parser.parse_args()

    import torch
    from proteus_core import ProteusConfig, adapt_model, load_traffic_split

    train_X, train_y, label_to_id = load_traffic_split(
        args.train,
        args.feature,
        args.seq_len,
        args.view,
        None,
        args.label_key,
        args.sequence_key,
    )
    target_X, target_y, _ = load_traffic_split(
        args.target,
        args.feature,
        args.seq_len,
        args.view,
        label_to_id,
        args.label_key,
        args.sequence_key,
    )
    num_classes = int(max(train_y.max().item(), target_y.max().item()) + 1)

    model = load_model(args.model_module, args.model_class, num_classes, args.model_kwargs)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.checkpoint_key:
        checkpoint = checkpoint[args.checkpoint_key]
    model.load_state_dict(checkpoint, strict=False)

    config = ProteusConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        gmm_threshold=args.gmm_threshold,
        pseudo_refresh_epochs=args.pseudo_refresh_epochs,
        mmd_weight=args.mmd_weight,
        entropy_weight=args.entropy_weight,
        pseudo_weight=args.pseudo_weight,
        source_ce_weight=args.source_ce_weight,
    )
    model, history = adapt_model(
        model,
        train_X,
        train_y,
        target_X,
        target_y,
        device=args.device,
        config=config,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_checkpoint)), exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    if args.history_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_json)), exist_ok=True)
        with open(args.history_json, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)


if __name__ == "__main__":
    main()
