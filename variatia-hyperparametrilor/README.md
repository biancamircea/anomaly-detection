# ADMoE Hyperparameter Evaluation Fast

Acest proiect conține un script pentru evaluarea rapidă a hiperparametrilor ADMoE și pentru rularea experimentelor pe checkpoint-uri deja antrenate.

## Ce face scriptul

Fișierul principal este `ADMoE_Hyperparameter_Evaluation_Fast.py` și:

- încarcă dataset-ul curat din `datasets/clean_data/`;
- încarcă noisy labels din `datasets/noisy_data/classification/<dataset>/`;
- construiește modelul ADMoE-MLP direct în script;
- încarcă checkpoint-ul `.pt` potrivit din `checkpoints/`;
- evaluează modelul pe train / valid / test;
- rulează experimente pentru:
  - nivelul de noisy labels;
  - `num_experts` și `top_k`;
  - `embedding_dim`, `learning_rate`, `aux_loss_weight`;
- salvează figurile în `output/`;
- salvează tabelele cu rezultate în `admoe_hyperparameter_results/`.

## Cerințe

- Python `3.11`
- Dependențele din `requirements.txt`


## Instalare

```bash
pip install -r requirements.txt
```

## Rulare

```bash
python ADMoE_Hyperparameter_Evaluation_Fast.py
```
