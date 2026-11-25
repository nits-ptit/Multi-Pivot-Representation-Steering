# Multi-Pivot-Representation-Steering

Official implementation of **Multi-Pivot Prompting (MPP)** & **Multi-Pivot Representation Steering (MPRS)** from *Many-to-Many Machine Translation Representation Steering through Multi-Pivot Prompting*.

## Setup & Dataset

```bash
cd "Multi-Pivot Prompting"
pip install -r requirements.txt
# Optional for BLEURT-20:
pip install git+https://github.com/google-research/bleurt.git

# Download FLORES-200 dataset (extracts to ./flores200_dataset/{dev,devtest})
wget --trust-server-names https://tinyurl.com/flores200dataset -O flores200_dataset.tar.gz
tar -xzf flores200_dataset.tar.gz && rm flores200_dataset.tar.gz
```

## Multi-Pivot Prompting (MPP)

Run all commands below from inside `Multi-Pivot Prompting/`:

### Stage 1: Pivot Selection
Evaluate direct translation on FLORES-200 `dev` with COMET, build the quality matrix $Q(s,t)$, and select the top-5 pivots via $S(s,p,t) = 0.3\,Q(s,p) + 0.7\,Q(p,t)$:
```bash
python3 -m mpp.prepare_flores --flores_dir flores200_dataset --split dev --output_path outputs/dev_direct.json
python3 -m eval.run_eval --model_path /path/to/model --infer_data_path outputs/dev_direct.json --metrics comet --comet_path Unbabel/wmt22-comet-da
python3 -m mpp.build_quality_matrix --dev_direct_path outputs/dev_direct.json --output_csv_path outputs/parallel_data.csv
python3 -m mpp.pivot_selection --quality_matrix_csv outputs/parallel_data.csv --output_path outputs/pivot_selection_top_5.json --top_k 5 --w_sp 0.3 --w_pt 0.7
```

### Stage 2: Candidate Generation & QE Filtering
Generate direct ($s \to t$) and two-hop pivot ($s \to p_i \to t$) candidates on `devtest`, score them with **COMETKiwi-DA**, and retain the top-3 candidates beating direct translation:
```bash
python3 -m mpp.prepare_flores --flores_dir flores200_dataset --split devtest --output_path outputs/devtest_direct.json
python3 -m eval.run_eval --model_path /path/to/model --infer_data_path outputs/devtest_direct.json --metrics comet,bleurt --comet_path Unbabel/wmt22-comet-da --bleurt_path /path/to/BLEURT-20

python3 -m mpp.construct_pivot_data --pivot_selection_path outputs/pivot_selection_top_5.json --flores_dir flores200_dataset --split devtest --direct_mt_path outputs/devtest_direct.json --output_path outputs/devtest_pivots.json
python3 -m eval.run_eval --model_path /path/to/model --infer_data_path outputs/devtest_pivots.json --add_target_reference True

python3 -m mpp.comet_qe --input_path outputs/devtest_pivots.json --comet_qe_path Unbabel/wmt22-cometkiwi-da
python3 -m mpp.filter_pivots --input_path outputs/devtest_pivots.json --top_k 3
```

### Stage 3: Candidate Aggregation & Evaluation
Generate final translations conditioned on the filtered pivot references (falling back to direct translation when none survive) and evaluate with COMET and BLEURT:
```bash
python3 -m eval.run_eval --model_path /path/to/model --infer_data_path outputs/devtest_pivots.json --target_reference_prompt True --metrics comet,bleurt --comet_path Unbabel/wmt22-comet-da --bleurt_path /path/to/BLEURT-20
```
