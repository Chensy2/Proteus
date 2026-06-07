import argparse
import json
import os
import random
from collections import Counter

np = None
train_test_split = None


DEFAULT_PS_KEYS = (
    "ps",
    "packet_size",
    "packet_sizes",
    "packet_length",
    "packet_lengths",
    "length",
    "lengths",
    "flow",
    "sequence",
    "x",
    "X",
)
DEFAULT_IAT_KEYS = (
    "iat",
    "iats",
    "time",
    "times",
    "arrive_time_delta",
    "arrival_time_delta",
    "inter_arrival_time",
    "inter_arrival_times",
)
DEFAULT_LABEL_KEYS = ("label", "y", "class", "class_name", "site", "website", "target")
CONTAINER_KEYS = ("samples", "data", "records", "traces", "flows", "items")


def _read_json_file(path):
    with open(path, "r", encoding="utf-8") as fp:
        text = fp.read().strip()
    if not text:
        return []
    if path.lower().endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _looks_like_sample(obj):
    if not isinstance(obj, dict):
        return False
    keys = set(obj)
    return bool(keys.intersection(DEFAULT_PS_KEYS + DEFAULT_IAT_KEYS))


def _iter_json_object(obj, inherited_label=None):
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_json_object(item, inherited_label)
        return

    if not isinstance(obj, dict):
        return

    if _looks_like_sample(obj):
        item = dict(obj)
        if inherited_label is not None and not any(k in item for k in DEFAULT_LABEL_KEYS):
            item["label"] = inherited_label
        yield item
        return

    for key in CONTAINER_KEYS:
        if key in obj:
            yield from _iter_json_object(obj[key], inherited_label)
            return

    # Support {"class_a": [samples...], "class_b": [samples...]} layouts.
    for key, value in obj.items():
        if isinstance(value, (list, dict)):
            yield from _iter_json_object(value, key)


def load_records(path):
    records = []
    path = os.path.abspath(path)
    if os.path.isfile(path):
        records.extend(_iter_json_object(_read_json_file(path)))
    elif os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            json_files = [
                name for name in sorted(files)
                if name.lower().endswith((".json", ".jsonl"))
            ]
            if not json_files:
                continue
            dir_label = os.path.basename(root)
            for name in json_files:
                file_path = os.path.join(root, name)
                records.extend(_iter_json_object(_read_json_file(file_path), dir_label))
    else:
        raise FileNotFoundError(path)
    return records


def _first_existing(sample, keys):
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _extract_label(sample, label_key=None):
    if label_key:
        if label_key not in sample:
            raise KeyError("label key '{}' not found in sample".format(label_key))
        return sample[label_key]
    label = _first_existing(sample, DEFAULT_LABEL_KEYS)
    if label is None:
        raise KeyError("No label found. Pass --label-key or use a class directory layout.")
    return label


def _extract_sequence(sample, view, sequence_key=None):
    if sequence_key:
        if sequence_key not in sample:
            raise KeyError("sequence key '{}' not found in sample".format(sequence_key))
        seq = sample[sequence_key]
    elif view in ("ps", "dir"):
        seq = _first_existing(sample, DEFAULT_PS_KEYS)
    elif view == "iat":
        seq = _first_existing(sample, DEFAULT_IAT_KEYS)
    else:
        raise ValueError("Unsupported view: {}".format(view))

    if seq is None:
        raise KeyError("No sequence found for view '{}'. Pass --sequence-key.".format(view))
    if isinstance(seq, dict):
        for key in ("values", "seq", "sequence", "data"):
            if key in seq:
                seq = seq[key]
                break
    seq = [float(x) for x in seq if x is not None]
    if not seq:
        raise ValueError("Empty sequence for view '{}'.".format(view))
    return seq


def _align_sequence(seq, seq_len):
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= seq_len:
        return seq[:seq_len]
    out = np.zeros((seq_len,), dtype=np.float32)
    out[:len(seq)] = seq
    return out


def build_arrays(records, label_to_id, view, seq_len, label_key=None, sequence_key=None):
    X = []
    y = []
    skipped = 0
    for sample in records:
        try:
            label = str(_extract_label(sample, label_key))
            if label not in label_to_id:
                raise KeyError("Unknown label '{}'.".format(label))
            seq = _extract_sequence(sample, view, sequence_key)
            X.append(_align_sequence(seq, seq_len))
            y.append(label_to_id[label])
        except (KeyError, TypeError, ValueError) as exc:
            skipped += 1
            if skipped <= 5:
                print("[Skip] {}".format(exc))
    if not X:
        raise ValueError("No usable samples were found.")
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64), skipped


def make_label_mapping(source_records, target_records, label_key=None):
    source_labels = [str(_extract_label(sample, label_key)) for sample in source_records]
    target_labels = [str(_extract_label(sample, label_key)) for sample in target_records]
    source_set = set(source_labels)
    target_set = set(target_labels)
    unknown_target = sorted(target_set - source_set)
    if unknown_target:
        raise ValueError(
            "Target contains labels absent from source: {}. "
            "Proteus closed-world baselines require source and target to share classes.".format(
                unknown_target[:10]
            )
        )
    labels = sorted(source_set)
    return {label: idx for idx, label in enumerate(labels)}


def resolve_output_dir(out_dataset):
    if os.path.isabs(out_dataset) or os.sep in out_dataset or (os.altsep and os.altsep in out_dataset):
        return os.path.abspath(out_dataset)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "datasets", out_dataset)


def save_npz(path, X, y):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y)
    print("[Save] {} X={} y={}".format(path, X.shape, y.shape))


def split_source(X, y, valid_ratio, seed, stratify):
    if valid_ratio <= 0:
        return X, y, np.empty((0, X.shape[1]), dtype=X.dtype), np.empty((0,), dtype=y.dtype)
    stratify_y = y if stratify else None
    if stratify:
        counts = Counter(y.tolist())
        if min(counts.values()) < 2:
            print("[Warn] Some classes have <2 samples; disabling stratified source split.")
            stratify_y = None
    return train_test_split(
        X,
        y,
        test_size=valid_ratio,
        random_state=seed,
        stratify=stratify_y,
    )


def main():
    global np
    global train_test_split

    parser = argparse.ArgumentParser(
        description="Convert source/target JSON traffic traces to Proteus/WFlib npz files."
    )
    parser.add_argument("--source", required=True, help="Source-domain JSON/JSONL file or class-directory root.")
    parser.add_argument("--target", required=True, help="Target-domain JSON/JSONL file or class-directory root.")
    parser.add_argument("--out_dataset", required=True, help="Dataset name under ./datasets or an explicit output path.")
    parser.add_argument("--source_valid", default=None, help="Optional source-domain validation JSON/JSONL or directory.")
    parser.add_argument("--view", choices=["dir", "ps", "iat"], default="dir",
                        help="Sequence view to store in X. For DF+DIR baseline, use dir with signed ps.")
    parser.add_argument("--seq_len", type=int, default=5000, help="Pad/truncate sequences to this length.")
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Source validation ratio when --source_valid is not provided.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--label_key", default=None)
    parser.add_argument("--sequence_key", default=None)
    parser.add_argument("--no_stratify", action="store_true")
    args = parser.parse_args()

    import numpy as _np
    from sklearn.model_selection import train_test_split as _train_test_split
    np = _np
    train_test_split = _train_test_split

    random.seed(args.seed)
    np.random.seed(args.seed)

    source_records = load_records(args.source)
    target_records = load_records(args.target)
    valid_records = load_records(args.source_valid) if args.source_valid else None

    print("[Load] source records={}".format(len(source_records)))
    if valid_records is not None:
        print("[Load] source valid records={}".format(len(valid_records)))
    print("[Load] target records={}".format(len(target_records)))

    mapping_source = source_records + (valid_records or [])
    label_to_id = make_label_mapping(mapping_source, target_records, args.label_key)
    out_dir = resolve_output_dir(args.out_dataset)
    os.makedirs(out_dir, exist_ok=True)

    X_source, y_source, skipped_source = build_arrays(
        source_records, label_to_id, args.view, args.seq_len, args.label_key, args.sequence_key
    )
    X_target, y_target, skipped_target = build_arrays(
        target_records, label_to_id, args.view, args.seq_len, args.label_key, args.sequence_key
    )

    if valid_records is None:
        X_train, X_valid, y_train, y_valid = split_source(
            X_source,
            y_source,
            args.valid_ratio,
            args.seed,
            not args.no_stratify,
        )
    else:
        X_train, y_train = X_source, y_source
        X_valid, y_valid, skipped_valid = build_arrays(
            valid_records, label_to_id, args.view, args.seq_len, args.label_key, args.sequence_key
        )
        skipped_source += skipped_valid

    save_npz(os.path.join(out_dir, "train.npz"), X_train, y_train)
    save_npz(os.path.join(out_dir, "valid.npz"), X_valid, y_valid)
    save_npz(os.path.join(out_dir, "test.npz"), X_target, y_target)

    with open(os.path.join(out_dir, "label_mapping.json"), "w", encoding="utf-8") as fp:
        json.dump(label_to_id, fp, indent=2, ensure_ascii=False, sort_keys=True)
    manifest = {
        "source": os.path.abspath(args.source),
        "source_valid": os.path.abspath(args.source_valid) if args.source_valid else None,
        "target": os.path.abspath(args.target),
        "view": args.view,
        "seq_len": args.seq_len,
        "valid_ratio": args.valid_ratio if args.source_valid is None else None,
        "seed": args.seed,
        "num_classes": len(label_to_id),
        "skipped_source": skipped_source,
        "skipped_target": skipped_target,
        "files": {
            "train": "train.npz",
            "valid": "valid.npz",
            "test": "test.npz",
            "label_mapping": "label_mapping.json",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
    print("[Done] dataset written to {}".format(out_dir))


if __name__ == "__main__":
    main()
