This package contains data and training scripts.

### Data (see releases page to download)

We include training and evaluation data for all three datasets (ManySStuBs4J, TSSB-3M and MegaDiff) with
28 lines of pre and post-context (for a total of 56 context lines).
The data is released as gzipped JSONL files. Each record has the following fields:
* `old`: the buggy code *with* context
* `new`: the fixed code (buggy section only)  
* `train`: whether this is a training sample or not
* `change_count`: number of changed lines in the corresponding diff
* `hunk_count`: number of hunks in the corresponding diff
* `labels` (optional): labels if available in the original dataset
* `repo_name`: repository name in the format <owner_name/project_name> (GitHub)
* `sha`: commit hash of the fixing commit

To obtain different context sizes, simply remove the first/last n lines of `old`.

Please note that the code (bugs) in the dataset is released under specific licenses. Please refer to the original datasets or the corresponding code repositories for more details.

### Training

We use an adapted version of HuggingFace's transformer library training script (`train.py`).
The parameters we use are given in `train.sh`. A typical training run would look like

```sh
$ bash train.sh <input_file.jsonl.gz> <model_output_dir> --overwrite_output_dir --min_hunk_count 1 --max_hunk_count 1
```

### Prediction

For prediction we use `predict.py`. Typical use might look like

```sh
$ python predict.py <model_dir> <input_file.jsonl.gz> <output_file.jsonl.gz>
```

This will predict 5 (by default, can be changed) possible fixes for all non-training samples. Predictions
are saved to the `preds` column in the output file.

### Evaluation

We use exact-match for evaluation. In particular we used the following code to match correct fixes.

```py
def match_df(row):
    new = row['new']
    if new is None or pd.isna(new):
        return pd.NA

    normalized_new = re.sub(r"\s+", " ", new).strip()

    for pred in row['preds']:
        normalized_pred = re.sub(r"\s+", " ", pred).strip()

        if normalized_pred == normalized_new:
            return pred

    return pd.NA

    ...

    df['matching_pred'] = df.apply(match_df, axis=1)
    df['matching'] = df.matching_pred.notna()
```
