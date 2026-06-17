import os
import argparse
import numpy as np
import pandas as pd
import torch
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score
from torch import nn, optim

from moe import MoE
from loaders import PyODDataset


RANDOM_SEED = 42

EPOCHS = 100
N_NOISY_RULES = 4
EMBEDDING_DIM = 30
NUM_EXPERTS = 4
TOP_K = 2
LR = 0.0001
BATCH_SIZE = 256
CHECKPOINT_EVERY = 10

DATASETS = ["27_mnist", "imdb"]
NOISE_LEVEL = 0.2

# 0 = DecisionTree, 1 = LightGBM, 2 = MLP, 3 = RandomForest
SOURCE_IDX = 0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Calea către checkpoint-ul .pt de la care se reia antrenarea"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Numărul total de epoci"
    )

    return parser.parse_args()


class NeuralNetwork(nn.Module):
    def __init__(self, n_features, hidden_neuron=64, n_layers=1,
                 num_experts=4, k=2, device="cpu"):
        super().__init__()

        layers = [
            nn.Linear(n_features, hidden_neuron),
            nn.ELU()
        ]

        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_neuron, hidden_neuron))
            layers.append(nn.ELU())

        self.simple_nn = nn.Sequential(*layers)

        self.moe = MoE(
            input_size=hidden_neuron,
            output_size=2,
            num_experts=num_experts,
            hidden_size=int(hidden_neuron * 0.5),
            noisy_gating=True,
            k=k,
            device=device
        )

    def forward(self, x):
        h = self.simple_nn(x)
        logits, aux_loss = self.moe(h)
        return logits, aux_loss


def get_embedding(n_noisy_rules, batch_len, batch_y_noisy, noisy_embedding, device):
    batch_mask = (torch.arange(n_noisy_rules) + 1).repeat(batch_len, 1).to(device)
    batch_y_noisy_masked = batch_y_noisy * batch_mask
    flat = batch_y_noisy_masked.flatten()

    nonzero_pos = torch.nonzero(flat).ravel()

    emb_full = torch.zeros(batch_len * n_noisy_rules, EMBEDDING_DIM).to(device)

    if len(nonzero_pos) > 0:
        non_zero_idx = (flat[nonzero_pos] - 1).long()
        non_zero_idx = torch.clamp(non_zero_idx, min=0, max=n_noisy_rules - 1)
        emb_full[nonzero_pos, :] = noisy_embedding(non_zero_idx)

    emb_full = emb_full.reshape(batch_len, n_noisy_rules, EMBEDDING_DIM)

    return torch.mean(emb_full, dim=1)


def save_checkpoint(path, epoch, model, noisy_embedding, optimizer,
                    valid_roc, test_roc, dataset, noise_level, source_idx):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "embedding_state_dict": noisy_embedding.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_auc": valid_roc,
            "test_auc": test_roc,
            "dataset": dataset,
            "noise_level": noise_level,
            "source_idx": source_idx + 1,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "embedding_dim": EMBEDDING_DIM,
        },
        path
    )


def load_data(dataset_name, noise_level):
    X = pd.read_csv(
        os.path.join("datasets", "clean_data", f"{dataset_name}_X.csv"),
        header=None
    ).to_numpy()

    y = pd.read_csv(
        os.path.join("datasets", "clean_data", f"{dataset_name}_y.csv"),
        header=None
    ).to_numpy()

    y1 = pd.read_csv(
        os.path.join(
            "datasets", "noisy_data", "classification", dataset_name,
            f"{dataset_name}_y_{noise_level}_DecisionTreeClassifier.csv"
        ),
        header=None
    ).to_numpy()

    y2 = pd.read_csv(
        os.path.join(
            "datasets", "noisy_data", "classification", dataset_name,
            f"{dataset_name}_y_{noise_level}_LGBMClassifier.csv"
        ),
        header=None
    ).to_numpy()

    y3 = pd.read_csv(
        os.path.join(
            "datasets", "noisy_data", "classification", dataset_name,
            f"{dataset_name}_y_{noise_level}_MLPClassifier.csv"
        ),
        header=None
    ).to_numpy()

    y4 = pd.read_csv(
        os.path.join(
            "datasets", "noisy_data", "classification", dataset_name,
            f"{dataset_name}_y_{noise_level}_RandomForestClassifier.csv"
        ),
        header=None
    ).to_numpy()

    return X, y, y1, y2, y3, y4


def train_one_experiment(dataset_name, noise_level, source_idx, epochs,
                         resume_checkpoint=None):
    random_state = np.random.RandomState(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, y1, y2, y3, y4 = load_data(dataset_name, noise_level)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    train_perc = 0.7
    test_perc = 0.25

    indices = np.arange(len(y))
    random_state.shuffle(indices)

    train_idx = indices[:int(n_samples * train_perc)]
    test_idx = indices[int(n_samples * train_perc):int(n_samples * (train_perc + test_perc))]
    valid_idx = indices[int(n_samples * (train_perc + test_perc)):]

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    X_valid = X[valid_idx, :]

    y_train = y[train_idx]
    y_test = y[test_idx]
    y_valid = y[valid_idx]

    y1_train, y2_train, y3_train, y4_train = y1[train_idx], y2[train_idx], y3[train_idx], y4[train_idx]
    y1_test, y2_test, y3_test, y4_test = y1[test_idx], y2[test_idx], y3[test_idx], y4[test_idx]
    y1_valid, y2_valid, y3_valid, y4_valid = y1[valid_idx], y2[valid_idx], y3[valid_idx], y4[valid_idx]

    X_train, scaler = standardizer(X_train, keep_scalar=True)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_valid = X_valid.astype(np.float32)

    y_valid_noisy = torch.from_numpy(
        np.concatenate([y1_valid, y2_valid, y3_valid, y4_valid], axis=1)
    ).long().to(device)

    y_test_noisy = torch.from_numpy(
        np.concatenate([y1_test, y2_test, y3_test, y4_test], axis=1)
    ).long().to(device)

    train_set = PyODDataset(
        X=X_train,
        y=y_train,
        y1=y1_train,
        y2=y2_train,
        y3=y3_train,
        y4=y4_train
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = NeuralNetwork(
        n_features=n_features + EMBEDDING_DIM,
        hidden_neuron=64,
        n_layers=1,
        num_experts=NUM_EXPERTS,
        k=TOP_K,
        device=device
    ).to(device)

    noisy_embedding = nn.Embedding(N_NOISY_RULES, EMBEDDING_DIM).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(noisy_embedding.parameters()),
        lr=LR
    )

    criterion = nn.CrossEntropyLoss()

    ckpt_dir = os.path.join(
        "checkpoints",
        dataset_name,
        f"noise_{noise_level}",
        f"source_{source_idx + 1}"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(ckpt_dir, "best_checkpoint.pt")

    best_valid = -1
    best_test = -1
    start_epoch = 0

    checkpoint_log = []

    if resume_checkpoint is not None:
        if not os.path.exists(resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint-ul nu există: {resume_checkpoint}")

        checkpoint = torch.load(resume_checkpoint, map_location=device)

        dataset_name = checkpoint["dataset"]
        noise_level = checkpoint["noise_level"]
        source_idx = checkpoint["source_idx"] - 1

        model.load_state_dict(checkpoint["model_state_dict"])
        noisy_embedding.load_state_dict(checkpoint["embedding_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_valid = checkpoint.get("valid_auc", -1)
        best_test = checkpoint.get("test_auc", -1)

        print("\nRESUME ACTIV")
        print(f"Checkpoint: {resume_checkpoint}")
        print(f"Dataset: {dataset_name}")
        print(f"Noise level: {noise_level}")
        print(f"Source: {source_idx + 1}")
        print(f"Se continuă de la epoca: {start_epoch}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch_x = batch[0].float().to(device)

            batch_y_noisy = torch.cat(
                (batch[2], batch[3], batch[4], batch[5]),
                dim=1
            ).long().to(device)

            emb = get_embedding(
                N_NOISY_RULES,
                len(batch_x),
                batch_y_noisy,
                noisy_embedding,
                device
            )

            batch_x_comb = torch.cat([batch_x, emb], dim=1)

            noisy_idx = source_idx

            optimizer.zero_grad()

            preds, aux_loss = model(batch_x_comb)

            loss = aux_loss + criterion(
                preds,
                batch_y_noisy[:, noisy_idx].ravel().long()
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()

        with torch.no_grad():
            valid_emb = get_embedding(
                N_NOISY_RULES,
                len(y_valid_noisy),
                y_valid_noisy,
                noisy_embedding,
                device
            )

            valid_x_comb = torch.cat(
                [torch.tensor(X_valid).float().to(device), valid_emb],
                dim=1
            )

            valid_scores, _ = model(valid_x_comb)

            valid_roc = roc_auc_score(
                y_valid,
                valid_scores[:, 1].cpu().numpy()
            )

            test_emb = get_embedding(
                N_NOISY_RULES,
                len(y_test_noisy),
                y_test_noisy,
                noisy_embedding,
                device
            )

            test_x_comb = torch.cat(
                [torch.tensor(X_test).float().to(device), test_emb],
                dim=1
            )

            test_scores, _ = model(test_x_comb)

            test_roc = roc_auc_score(
                y_test,
                test_scores[:, 1].cpu().numpy()
            )

        periodic_checkpoint = ""

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            periodic_checkpoint = os.path.join(
                ckpt_dir,
                f"epoch_{epoch + 1:03d}.pt"
            )

            save_checkpoint(
                periodic_checkpoint,
                epoch + 1,
                model,
                noisy_embedding,
                optimizer,
                valid_roc,
                test_roc,
                dataset_name,
                noise_level,
                source_idx
            )

        if valid_roc > best_valid:
            best_valid = valid_roc
            best_test = test_roc

            save_checkpoint(
                best_checkpoint_path,
                epoch + 1,
                model,
                noisy_embedding,
                optimizer,
                valid_roc,
                test_roc,
                dataset_name,
                noise_level,
                source_idx
            )

        checkpoint_log.append({
            "Dataset": dataset_name,
            "Noise_Level": noise_level,
            "Source": source_idx + 1,
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Valid_ROC": valid_roc,
            "Test_ROC": test_roc,
            "Best_Valid_ROC": best_valid,
            "Best_Test_ROC": best_test,
            "Periodic_Checkpoint": periodic_checkpoint,
            "Best_Checkpoint": best_checkpoint_path
        })

        print(
            f"dataset={dataset_name} "
            f"noise={noise_level} "
            f"source={source_idx + 1} "
            f"epoch={epoch + 1}/{epochs} "
            f"loss={train_loss:.4f} "
            f"valid_roc={valid_roc:.4f} "
            f"test_roc={test_roc:.4f} "
            f"best_valid={best_valid:.4f} "
            f"best_test={best_test:.4f}"
        )

    pd.DataFrame(checkpoint_log).to_csv(
        os.path.join(ckpt_dir, "checkpoint_log.csv"),
        index=False
    )

    return best_test, best_valid, best_checkpoint_path


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    results = []

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")

        dataset_list = [ckpt["dataset"]]
        noise_level = ckpt["noise_level"]
        source_idx = ckpt["source_idx"] - 1

        print("\nRESUME MODE")
        print(f"Se rulează doar datasetul: {dataset_list[0]}")
        print(f"Noise level: {noise_level}")
        print(f"Source: {source_idx + 1}\n")

    else:
        dataset_list = DATASETS
        noise_level = NOISE_LEVEL
        source_idx = SOURCE_IDX

    for dataset_name in dataset_list:
        best_test, best_valid, best_ckpt = train_one_experiment(
            dataset_name=dataset_name,
            noise_level=noise_level,
            source_idx=source_idx,
            epochs=args.epochs,
            resume_checkpoint=args.resume_checkpoint
        )

        results.append({
            "Dataset": dataset_name,
            "Noise_Level": noise_level,
            "Source": source_idx + 1,
            "Best_Valid_ROC": best_valid,
            "Best_Test_ROC": best_test,
            "Best_Checkpoint": best_ckpt
        })

    pd.DataFrame(results).to_csv("single_source_results.csv", index=False)

    print("\nRezultate salvate în:")
    print("single_source_results.csv")