================================================================================
ADMoE – Detectarea anomaliilor cu Mixture-of-Experts din noisy labels
================================================================================

1. INSTALARE
--------------------------------------------------------------------------------

Se recomandă utilizarea unui mediu virtual.

Instalare cu pip:

    pip install -r requirements.txt

Instalare cu uv:

    uv pip install -r requirements.txt


================================================================================
2. RULAREA EXPERIMENTULUI PRINCIPAL
--------------------------------------------------------------------------------

Scriptul principal este:

    main.py

Rulare normală:

    python main.py

Acest script:
- încarcă seturile de date curate;
- încarcă noisy labels;
- împarte datele în train / validation / test;
- antrenează ADMoE-MLP;
- rulează baseline-urile activate;
- calculează metricile;
- salvează rezultatele;
- salvează checkpoint-uri.


================================================================================
3. RULARE DIN CHECKPOINT
--------------------------------------------------------------------------------

Pentru reluarea antrenării dintr-un checkpoint pentru un anumit set de date se folosește argumentul:

    --resume_checkpoint

Exemplu:

    python main.py --resume_checkpoint checkpoints/imdb/noise_0.2/source_1/best_checkpoint.pt

Calea către checkpoint se ia din folderul:

    checkpoints/

Structura checkpoint-urilor este:

    checkpoints/<dataset>/noise_<nivel_zgomot>/source_<sursa>/best_checkpoint.pt

Exemplu:

    checkpoints/imdb/noise_0.2/source_1/best_checkpoint.pt

Checkpoint-ul recomandat este:

    best_checkpoint.pt

Acesta este salvat automat când modelul obține cel mai bun ROC-AUC pe setul de validare.


================================================================================
4. CHECKPOINT-URI SALVATE
--------------------------------------------------------------------------------

În timpul antrenării se salvează:

- un checkpoint la fiecare 10 epoci;
- best_checkpoint.pt atunci când Validation ROC-AUC se îmbunătățește.

Exemplu structură:

    checkpoints/
        imdb/
            noise_0.2/
                source_1/
                    epoch_010.pt
                    epoch_020.pt
                    epoch_030.pt
                    epoch_040.pt
                    best_checkpoint.pt
                    checkpoint_log.csv


================================================================================
5. FIȘIERE GENERATE
--------------------------------------------------------------------------------

După rulare, se generează metricile cu ajutorul functiei " plot_metrics.py " prin rularea :
  
  python plot_metrics.py 

Astfel, se obtin urmatoarele fisiere:

    metrics_complete.csv
    metrics_summary.csv
    metrics_partial.csv
    false_positive_cases.csv
    false_negative_cases.csv
    checkpoints/

Descriere:

    metrics_complete.csv
        Conține toate metricile pentru fiecare dataset, noise level, model și source.

    metrics_summary.csv
        Conține media metricilor pe fiecare dataset, model și noise level.

    metrics_partial.csv
        Salvează rezultate intermediare în timpul rulării.

    false_positive_cases.csv
        Conține cazurile cu cele mai multe False Positive.

    false_negative_cases.csv
        Conține cazurile cu cele mai multe False Negative.

    checkpoints/
        Conține checkpoint-urile salvate în timpul antrenării.


================================================================================
6. RULAREA MODELELOR INDIVIDUALE
--------------------------------------------------------------------------------

ADMoE cu MLP:

    python demo_admoe_mlp.py

ADMoE cu DeepSAD:

    python demo_admoe_deepsad.py

Baseline LightGBM:

    python exp_baseline_lightgbm.py

Baseline PReNet:

    python exp_baseline_prenet.py

Baseline DevNet:

    python exp_baseline_devnet.py

Baseline XGBOD:

    python exp_baseline_xgbod.py


================================================================================
7. STRUCTURA PROIECTULUI
--------------------------------------------------------------------------------

    main.py
        Scriptul principal.

    moe.py
        Implementarea Mixture-of-Experts:
        - Gating Network
        - Expert Networks
        - Top-k selection
        - Weighted aggregation
        - Auxiliary loss

    mlp.py
        Implementarea MLP.

    loaders.py
        Încărcarea datelor.

    myutils.py
        Funcții auxiliare.

    datasets/
        Seturi de date curate și noisy labels.

    baseline/
        Implementări baseline.

    checkpoints/
        Checkpoint-uri salvate.

    plots/
        Grafice generate.


================================================================================
8. OBSERVAȚII
--------------------------------------------------------------------------------

Rulare completă:

    python main.py

Reluare din checkpoint:

    python main.py --resume_checkpoint checkpoints/imdb/noise_0.2/source_1/best_checkpoint.pt

Dacă nu se dorește reluarea antrenării, nu se folosește argumentul
--resume_checkpoint.