import os
import random
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def display(value):
    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    elif isinstance(value, pd.Series):
        print(value.to_string())
    else:
        print(value)


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    figure_path = OUTPUT_DIR / filename
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {figure_path}")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


DATA_ROOT = Path("datasets")
DATASET_NAME = "imdb"  

CLASSIFIERS = [
    "RandomForestClassifier",
    "MLPClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
]


MAX_SAMPLES = None

EPOCHS = 20
BATCH_SIZE = 256
TEST_SIZE = 0.25
VALID_SIZE = 0.05

CHECKPOINT_DIR = Path("checkpoints")

DEFAULT_CONFIG = dict(
    num_experts=4,
    top_k=2,
    embedding_dim=16,
    hidden_dim=64,
    learning_rate=1e-3,
    aux_loss_weight=0.01,
    epochs=EPOCHS,
)

print("Dataset:", DATASET_NAME)
print("Default config:", DEFAULT_CONFIG)

def _candidate_noise_strings(value):
    if isinstance(value, str):
        return [value]
    return sorted({
        str(value),
        f"{value:.2f}".rstrip("0").rstrip("."),
        f"{value:.1f}",
        f"{value:.2f}",
    }, key=len)


def clean_paths(dataset_name=DATASET_NAME):
    x_path = DATA_ROOT / "clean_data" / f"{dataset_name}_X.csv"
    y_path = DATA_ROOT / "clean_data" / f"{dataset_name}_y.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Nu găsesc fișierele curate pentru {dataset_name}. Caut: {x_path} și {y_path}"
        )
    return x_path, y_path


def noisy_path(dataset_name, noise_level, classifier):
    folder = DATA_ROOT / "noisy_data" / "classification" / dataset_name
    for noise_str in _candidate_noise_strings(noise_level):
        p = folder / f"{dataset_name}_y_{noise_str}_{classifier}.csv"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Nu găsesc noisy label pentru dataset={dataset_name}, noise={noise_level}, classifier={classifier}. Folder: {folder}"
    )


def available_noise_levels(dataset_name=DATASET_NAME):
    folder = DATA_ROOT / "noisy_data" / "classification" / dataset_name
    if not folder.exists():
        raise FileNotFoundError(f"Nu există folderul: {folder}")
    levels = set()
    prefix = f"{dataset_name}_y_"
    for p in folder.glob(f"{dataset_name}_y_*_*.csv"):
        rest = p.name[len(prefix):-4]
        for clf in CLASSIFIERS:
            suffix = f"_{clf}"
            if rest.endswith(suffix):
                lvl = rest[:-len(suffix)]
                try:
                    levels.add(float(lvl))
                except ValueError:
                    levels.add(lvl)
    return sorted(levels, key=lambda x: float(x))


def load_dataset_with_noisy_labels(dataset_name=DATASET_NAME, noise_level=0.2, classifiers=CLASSIFIERS, max_samples=MAX_SAMPLES):
    x_path, y_path = clean_paths(dataset_name)
    X = pd.read_csv(x_path, header=None).to_numpy(dtype=np.float32)
    y = pd.read_csv(y_path, header=None).to_numpy().reshape(-1).astype(int)
    
    noisy = []
    for clf in classifiers:
        yp = noisy_path(dataset_name, noise_level, clf)
        noisy.append(pd.read_csv(yp, header=None).to_numpy().reshape(-1).astype(int))
    Yw = np.vstack(noisy).T.astype(int)  # shape: [n_samples, n_sources]
    
    # sampling reproducibil pentru rulare rapidă
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y, Yw = X[idx], y[idx], Yw[idx]
    
    return X, y, Yw


def load_dataset_with_mixed_noisy_labels(dataset_name=DATASET_NAME, source_specs=None, max_samples=MAX_SAMPLES):
    if source_specs is None:
        source_specs = [(0.2, clf) for clf in CLASSIFIERS]
    x_path, y_path = clean_paths(dataset_name)
    X = pd.read_csv(x_path, header=None).to_numpy(dtype=np.float32)
    y = pd.read_csv(y_path, header=None).to_numpy().reshape(-1).astype(int)
    noisy = []
    for noise_level, clf in source_specs:
        yp = noisy_path(dataset_name, noise_level, clf)
        noisy.append(pd.read_csv(yp, header=None).to_numpy().reshape(-1).astype(int))
    Yw = np.vstack(noisy).T.astype(int)
    
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X, y, Yw = X[idx], y[idx], Yw[idx]
    return X, y, Yw

print("Noise levels disponibile:", available_noise_levels(DATASET_NAME))
X_test_load, y_test_load, Yw_test_load = load_dataset_with_noisy_labels(DATASET_NAME, available_noise_levels(DATASET_NAME)[0])
print("X:", X_test_load.shape, "y:", y_test_load.shape, "Yw:", Yw_test_load.shape)
print("Primele noisy labels:\n", Yw_test_load[:5])

def make_splits(X, y, Yw, test_size=TEST_SIZE, valid_size=VALID_SIZE):

    idx = np.arange(len(X))
    train_valid_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y if len(np.unique(y)) == 2 else None,
    )

    rel_valid = valid_size / (1 - test_size)
    y_tv = y[train_valid_idx]
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=rel_valid,
        random_state=RANDOM_SEED,
        stratify=y_tv if len(np.unique(y_tv)) == 2 else None,
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
    X_valid = scaler.transform(X[valid_idx]).astype(np.float32)
    X_test = scaler.transform(X[test_idx]).astype(np.float32)
    
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y[train_idx].astype(int),
        "y_valid": y[valid_idx].astype(int),
        "y_test": y[test_idx].astype(int),
        "Yw_train": Yw[train_idx].astype(int),
        "Yw_valid": Yw[valid_idx].astype(int),
        "Yw_test": Yw[test_idx].astype(int),
    }

class NoisyLabelDataset(Dataset):
    def __init__(self, X, y_clean, Yw):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_clean = torch.tensor(y_clean, dtype=torch.long)
        self.Yw = torch.tensor(Yw, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y_clean[idx], self.Yw[idx]

class ADMoELayer(nn.Module):

    def __init__(self, input_dim, output_dim=2, num_experts=4, top_k=2, expert_hidden_dim=32):
        super().__init__()
        assert 1 <= top_k <= num_experts, "top_k trebuie să fie între 1 și num_experts"
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ELU(),
                nn.Linear(expert_hidden_dim, output_dim),
            ) for _ in range(num_experts)
        ])
    
    def forward(self, h):
        gate_logits = self.gate(h)                         
        gate_probs = torch.softmax(gate_logits, dim=1)    
        
        top_vals, top_idx = torch.topk(gate_probs, self.top_k, dim=1)
        sparse_gate = torch.zeros_like(gate_probs)
        sparse_gate.scatter_(1, top_idx, top_vals)
        sparse_gate = sparse_gate / (sparse_gate.sum(dim=1, keepdim=True) + 1e-8)
        
        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * sparse_gate.unsqueeze(-1), dim=1)
        
        usage = sparse_gate.mean(dim=0)
        target = torch.full_like(usage, 1.0 / self.num_experts)
        aux_loss = torch.mean((usage - target) ** 2) * self.num_experts
        return output, aux_loss, sparse_gate

class ADMoEMLP(nn.Module):
    def __init__(self, n_features, n_sources=4, embedding_dim=16, hidden_dim=64,
                 num_experts=4, top_k=2):
        super().__init__()
        self.n_sources = n_sources
        self.embedding_dim = embedding_dim
        
        self.label_embedding = nn.Embedding(n_sources * 2, embedding_dim)
        
        input_dim = n_features + embedding_dim
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.admoe = ADMoELayer(
            input_dim=hidden_dim,
            output_dim=2,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=max(8, hidden_dim // 2),
        )
    
    def noisy_embedding(self, Yw):
        batch_size, n_sources = Yw.shape
        source_ids = torch.arange(n_sources, device=Yw.device).unsqueeze(0).repeat(batch_size, 1)
        emb_ids = source_ids * 2 + Yw.long()
        emb = self.label_embedding(emb_ids)     
        return emb.mean(dim=1)                  
    
    def forward(self, X, Yw):
        emb = self.noisy_embedding(Yw)
        x_aug = torch.cat([X, emb], dim=1)
        h = self.base(x_aug)
        logits, aux_loss, gates = self.admoe(h)
        return logits, aux_loss, gates

def majority_vote_target(Yw):
    return (Yw.float().mean(dim=1) >= 0.5).long()

def _format_checkpoint_value(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}".replace(".", "p")
    return str(value).replace("/", "-").replace(" ", "_")

def checkpoint_path(dataset_name, model_name, config, noise_level=None, suffix="pt"):
    checkpoint_root = CHECKPOINT_DIR / dataset_name / model_name
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    parts = [
        f"dataset-{dataset_name}",
        f"model-{model_name}",
    ]
    if noise_level is not None:
        parts.append(f"noise-{_format_checkpoint_value(noise_level)}")
    for key in ["num_experts", "top_k", "embedding_dim", "hidden_dim", "learning_rate", "aux_loss_weight", "epochs"]:
        if key in config:
            parts.append(f"{key}-{_format_checkpoint_value(config[key])}")
    filename = "__".join(parts) + f".{suffix}"
    return checkpoint_root / filename

def build_admoe_model_from_config(n_features, n_sources, config):
    return ADMoEMLP(
        n_features=n_features,
        n_sources=n_sources,
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_experts=config["num_experts"],
        top_k=config["top_k"],
    ).to(DEVICE)

def load_checkpoint_artifact(dataset_name, model_name, config, *, noise_level=None):
    checkpoint_file = checkpoint_path(dataset_name, model_name, config, noise_level=noise_level)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Nu găsesc checkpoint-ul necesar: {checkpoint_file}. "
            f"Verifică dataset-ul și hiperparametrii din config."
        )
    payload = torch.load(checkpoint_file, map_location=DEVICE)
    return checkpoint_file, payload

def evaluate_model(model, loader):
    model.eval()
    ys, scores, gates_all = [], [], []
    with torch.no_grad():
        for Xb, y_clean, Ywb in loader:
            Xb = Xb.to(DEVICE)
            Ywb = Ywb.to(DEVICE)
            logits, _, gates = model(Xb, Ywb)
            prob_anomaly = torch.softmax(logits, dim=1)[:, 1]
            ys.append(y_clean.numpy())
            scores.append(prob_anomaly.cpu().numpy())
            gates_all.append(gates.cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(scores)
    gates_all = np.concatenate(gates_all, axis=0)
    roc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    gate_usage = gates_all.mean(axis=0)
    return roc, ap, gate_usage

def train_one_admoe_run(data, config, verbose=False, checkpoint_name=None, dataset_name=DATASET_NAME, noise_level=None):
    """În varianta curentă, funcția nu mai antrenează: încarcă checkpoint-ul .pt."""
    model_name = checkpoint_name or "ADMoEMLP"
    model = build_admoe_model_from_config(
        n_features=data["X_train"].shape[1],
        n_sources=data["Yw_train"].shape[1],
        config=config,
    )

    checkpoint_file, payload = load_checkpoint_artifact(
        dataset_name=dataset_name,
        model_name=model_name,
        config=config,
        noise_level=noise_level,
    )

    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict)

    train_ds = NoisyLabelDataset(data["X_train"], data["y_train"], data["Yw_train"] )
    valid_ds = NoisyLabelDataset(data["X_valid"], data["y_valid"], data["Yw_valid"] )
    test_ds = NoisyLabelDataset(data["X_test"], data["y_test"], data["Yw_test"] )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=2048, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False)

    train_auc, train_ap, train_gate_usage = evaluate_model(model, train_loader)
    valid_auc, valid_ap, valid_gate_usage = evaluate_model(model, valid_loader)
    test_auc, test_ap, gate_usage = evaluate_model(model, test_loader)

    return {
        "roc_auc": test_auc,
        "average_precision": test_ap,
        "gate_usage": gate_usage,
        "model": model,
        "checkpoint_file": str(checkpoint_file),
        "loaded_state": payload,
        "train_metrics": {
            "roc_auc": train_auc,
            "average_precision": train_ap,
            "gate_usage": train_gate_usage,
        },
        "valid_metrics": {
            "roc_auc": valid_auc,
            "average_precision": valid_ap,
            "gate_usage": valid_gate_usage,
        },
    }

def run_experiment(dataset_name=DATASET_NAME, noise_level=0.2, config=None, mixed_specs=None, verbose=False):
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    if mixed_specs is None:
        X, y, Yw = load_dataset_with_noisy_labels(dataset_name, noise_level, max_samples=MAX_SAMPLES)
    else:
        X, y, Yw = load_dataset_with_mixed_noisy_labels(dataset_name, mixed_specs, max_samples=MAX_SAMPLES)

    data = make_splits(X, y, Yw)

    result = train_one_admoe_run(
        data,
        cfg,
        verbose=verbose,
        checkpoint_name="ADMoEMLP",
        dataset_name=dataset_name,
        noise_level=noise_level if mixed_specs is None else "mixed",
    )

    return result

NOISE_LEVELS_TO_RUN = [0.01, 0.1, 0.2, 0.5]

noise_results = []
print("\n=== Starting experiment: noise sensitivity ===")
for lvl in NOISE_LEVELS_TO_RUN:
    print(f"Running noise level={lvl} ...")
    res = run_experiment(DATASET_NAME, noise_level=lvl, config=DEFAULT_CONFIG)
    noise_results.append({
        "noise_level": float(lvl),
        "roc_auc": res["roc_auc"],
        "average_precision": res["average_precision"],
    })

noise_df = pd.DataFrame(noise_results).sort_values("noise_level")
display(noise_df)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(noise_df["noise_level"], noise_df["roc_auc"], marker="o", label="ROC-AUC")
ax.plot(noise_df["noise_level"], noise_df["average_precision"], marker="s", label="Average Precision")
ax.set_xlabel("Noisy label quality / clean-label fraction used by generators")
ax.set_ylabel("Performance")
ax.set_title(f"Impactul nivelului de noisy labels – {DATASET_NAME}")
ax.grid(True, alpha=0.3)
ax.legend()
save_figure(fig, f"{DATASET_NAME}__noise_sensitivity.png")


EXPERT_VALUES = [2, 4, 8, 16]
TOP_K_VALUES = [1, 2, 4]
FIXED_NOISE_LEVEL = 0.2

expert_grid_results = []
print("\n=== Starting experiment: experts × top_k grid ===")
for m in EXPERT_VALUES:
    for k in TOP_K_VALUES:
        if k > m:
            continue
        print(f"Running num_experts={m}, top_k={k} ...")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({"num_experts": m, "top_k": k})
        res = run_experiment(DATASET_NAME, noise_level=FIXED_NOISE_LEVEL, config=cfg)
        expert_grid_results.append({
            "num_experts": m,
            "top_k": k,
            "roc_auc": res["roc_auc"],
            "average_precision": res["average_precision"],
            **{f"gate_usage_E{i+1}": val for i, val in enumerate(res["gate_usage"])}
        })

expert_grid_df = pd.DataFrame(expert_grid_results)
display(expert_grid_df)

pivot = expert_grid_df.pivot(index="num_experts", columns="top_k", values="roc_auc")
fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(pivot.values, aspect="auto")
fig.colorbar(im, ax=ax, label="ROC-AUC")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("top_k activat")
ax.set_ylabel("num_experts")
ax.set_title(f"ROC-AUC pentru num_experts × top_k, noise={FIXED_NOISE_LEVEL}")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center")
save_figure(fig, f"{DATASET_NAME}__experts_topk_grid__noise-{str(FIXED_NOISE_LEVEL).replace('.', 'p')}.png")


SENSITIVITY_SPACE = {
    "embedding_dim": [4, 8, 16, 32],
    "learning_rate": [1e-4, 5e-4, 1e-3, 3e-3],
    "aux_loss_weight": [0.0, 0.001, 0.01, 0.05],
}

sensitivity_results = []
print("\n=== Starting experiment: parameter sensitivity ===")
for param, values in SENSITIVITY_SPACE.items():
    for value in values:
        print(f"Running {param}={value} ...")
        cfg = DEFAULT_CONFIG.copy()
        cfg.update({param: value})
        res = run_experiment(DATASET_NAME, noise_level=FIXED_NOISE_LEVEL, config=cfg)
        sensitivity_results.append({
            "parameter": param,
            "value": value,
            "roc_auc": res["roc_auc"],
            "average_precision": res["average_precision"],
        })

sens_df = pd.DataFrame(sensitivity_results)
display(sens_df)

for param in SENSITIVITY_SPACE:
    sub = sens_df[sens_df["parameter"] == param].copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sub["value"].astype(float), sub["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(sub["value"].astype(float), sub["average_precision"], marker="s", label="Average Precision")
    if param == "learning_rate" or param == "aux_loss_weight":
        ax.set_xscale("log") if sub["value"].astype(float).min() > 0 else None
    ax.set_xlabel(param)
    ax.set_ylabel("Performance")
    ax.set_title(f"Impactul hiperparametrului {param}, noise={FIXED_NOISE_LEVEL}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, f"{DATASET_NAME}__sensitivity__{param}.png")


out_dir = Path("admoe_hyperparameter_results")
out_dir.mkdir(exist_ok=True)

if "noise_df" in globals():
    noise_df.to_csv(out_dir / f"{DATASET_NAME}_noise_sensitivity.csv", index=False)
if "expert_grid_df" in globals():
    expert_grid_df.to_csv(out_dir / f"{DATASET_NAME}_experts_topk_grid.csv", index=False)
if "sens_df" in globals():
    sens_df.to_csv(out_dir / f"{DATASET_NAME}_other_hyperparameters.csv", index=False)

print("Rezultate salvate în:", out_dir.resolve())
