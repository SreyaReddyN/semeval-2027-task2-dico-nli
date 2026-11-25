# DiCo-NLI: DeBERTa-v3-base baseline and reversal-aware training

Refactored from `DiCo_NLI_DeBERTa_v3_Base_Baseline.ipynb`. Contains two **independent** experiments: a standard 4-label NLI baseline, and a fresh reversal-aware model with consistency loss (weight 0.5). Both use 10 epochs maximum, early stopping (patience 2), weighted F1 selection, learning rate 2e-5, max sequence length 128, and seed 42.

## Setup (macOS / VS Code)

```bash
cd dico_nli_script
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data
```

Put these original labeled CSVs inside `data/`:
- `dico_nli_train_track1_participant_labeled.csv`
- `dico_nli_dev_track1_participant_labeled.csv`

In VS Code, select the `.venv` interpreter (Python: Select Interpreter).

## Commands

```bash
python train_dico_nli.py --help
python train_dico_nli.py --mode validate
python train_dico_nli.py --mode baseline
python train_dico_nli.py --mode reversal
python train_dico_nli.py --mode both
```

**Cautious M4 smoke test:** `python train_dico_nli.py --mode baseline --epochs 1 --batch-size 2 --eval-batch-size 2` (this still processes the **entire** dataset for one epoch). For reversal training try `--reverse-batch-size 1 --eval-batch-size 2`. Set `--device cpu` if MPS encounters an unsupported operation. A 16 GB machine may need smaller batches; memory availability depends on dataset size and other apps.

## Output files

`outputs/baseline/` and/or `outputs/reversal/` each contain:
- `best_model/` — model weights, tokenizer and config (not intended for Git)
- `training_history.csv` — epoch-by-epoch losses and validation metrics
- `dev_predictions.csv` — instance_id, pair_id, gold label, predicted label, confidence
- `classification_report.txt` — per-class precision, recall, F1
- `metrics.csv` — final development metrics

Terminal prints dataset checks, progress bars, epoch metrics, early stopping, and a classification report. No notebook-style tables are automatically displayed; open CSVs in VS Code or pandas.

## Notes

- Model weights are downloaded from Hugging Face on first training run; internet access required.
- `.gitignore` excludes dataset CSVs and trained weights. Check your dataset's license before sharing it.
- `--mode validate` does not download a model or start training.
- `--mode both` trains the baseline **then starts a fresh reversal-aware model**, matching the notebook's two distinct experiments. They are not sequential fine-tuning stages.
- The notebook's Google Colab uploads/Drive mounts, repeated inspection cells, gradient sanity tests, and Colab-only paths were removed.
- The original notebook's standalone checkpoint optimizer-state save was omitted; the best model is saved in Hugging Face `save_pretrained` format, not a resumable optimizer checkpoint.
- The script has been syntax-checked but not fully trained against the original CSVs; actual metrics and runtime are not verified.
