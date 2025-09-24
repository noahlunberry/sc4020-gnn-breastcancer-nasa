from pathlib import Path

path = Path(r"sc4020-gnn-breastcancer-nasa/cancer_gan_augmentation.py")
text = path.read_text()

first = text.find('"""')
second = text.find('"""', first + 3)
if first == -1 or second == -1:
    raise RuntimeError("Docstring bounds not found")
new_doc = """"""
Cancer-GAN trains a small CGAN on the Wisconsin Breast Cancer tabel.

Steps:
1) Load /mnt/data/breast-cancer.csv with columns like diagnosis or radius_mean.
2) Split the data into train and test sets with stratified labels.
3) Fit a small Conditional GAN on the malignant class M so the generator learn that shape.
4) Make extra malignant rows and join them with the train set to balance labels.
5) Fit a Logistic Regression model on the base train set. Fit the same model on the GAN boosted train set. Score both on the same test set.
6) Save the synthetic rows, a metrics csv, and a log file under /mnt/data/.

The script keep the workflow small for data mining lab work.
""""""
new_doc = new_doc.replace('\n', '\r\n')
text = text[:first] + new_doc + text[second+3:]

replacements = [
    ('# Torch (for the GAN)', '# Torch tools for the GAN stack'),
    ('# Scikit-learn (for preprocessing + evaluation + baseline classifier)', '# Scikit-learn pieces for prep work and metrics'),
    ('"""Append a line to the plain-text log file and print it for visibility."""', '"""Write each log line to the log file and print it."""'),
    ('# ----------------------------\r\n# 1) Load and inspect dataset\r\n# ----------------------------', '# Step 1: Load the dataset and inspect the colums\r\n'),
    ('# -------------------------------------\r\n# 2) Split into train/test (stratified)\r\n# -------------------------------------', '# Step 2: Make train and test splits with stratified labels\r\n'),
    ('# -----------------------------------\r\n# 3) Tiny Conditional GAN for tabular\r\n# -----------------------------------', '# Step 3: Set up the Conditional GAN for table data\r\n'),
    ('# ------------------------------------------------------\r\n# 4) Train/eval Logistic Regression on A vs B (baseline)\r\n# ------------------------------------------------------', '# Step 4: Train and score Logistic Regression on both setups\r\n'),
    ("    \"\"\"\n    Generator maps (noise, class_label) -> synthetic features.\n    We only use class_label=1 (malignant) during training/generation,\n    but keeping it \"conditional\" makes it easy to extend later.\n    \"\"\"", "    \"\"\"\n    Generator takes noise plus a class label and returns fake feature rows.\n    We fit it on the malignant label so the code stay ready for two labels.\n    \"\"\""),
    ("    \"\"\"\n    Discriminator maps (features, class_label) -> [0,1] real/fake probability.\n    \"\"\"", "    \"\"\"\n    Discriminator takes feature rows with a class label and returns a real versus fake score.\n    \"\"\""),
    ("    \"\"\"\n    Train the GAN using ONLY malignant (class==1) rows from the TRAIN set.\n    We fit on scaled numeric features.\n    \"\"\"", "    \"\"\"\n    Train the GAN with malignant class rows from the train set.\n    Work with scaled numeric features.\n    \"\"\""),
    ("    \"\"\"\n    Fit a simple, class-friendly classifier and report common metrics.\n    \"\"\"", "    \"\"\"\n    Fit Logistic Regression and report key metrics.\n    \"\"\""),
    ("# We expect standard Wisconsin columns with 'diagnosis' in {'M','B'} and an 'id' column.", "# Expect the Wisconsin schema with diagnosis and id feilds"),
    ("# Drop obvious non-feature column 'id'", "# Drop the id column becasue it is not a feature"),
    ("# Ensure diagnosis is present", "# Make sure the diagnosis column exist"),
    ("# y is one-hot (batch, label_dim)", "# y holds one hot labels shaped batch by label_dim"),
    ("    label_dim: int = 2   # two classes: 0=benign, 1=malignant", "    label_dim: int = 2   # two class values: 0 benign, 1 malignant"),
    ("    # Filter malignant samples", "    # Pick malignant samples from the train set"),
    ("    # Convert to tensors", "    # Turn arrays into tensors for torch"),
    ("    # Models", "    # Build the generator and the discriminator modules"),
    ("    # Optims and loss", "    # Set up Adam optimizers and BCE loss"),
    ("        # Mini-batch loop", "        # Loop over mini batches of malignant rows"),
    ("            # -----------------\r\n            # Train Discriminator\r\n            # -----------------", "            # Train the discriminator on real and fake rows"),
    ("            # Real", "            # Real batch pass"),
    ("            # Fake", "            # Fake batch pass"),
    ("            # For GAN training we still feed the correct label (1) as condition", "            # Feed label 1 as the condition during GAN training"),
    ("            # -----------------\r\n            # Train Generator\r\n            # -----------------", "            # Train the generator so fake rows fool the disc"),
    ("            # We want discriminator to think generated are real (label=1)", "            # Push the disc to score fake rows as label 1"),
    ('    # condition on malignant class = 1', '    # Set the condition tensor to malignant class 1'),
    ('    # reset log', '    # Reset the log file before we write anything'),
    ('    # Basic sanity about labels', '    # Check label counts for a sanity peek'),
    ('    # Split', '    # Split the dataset into train and test sets'),
    ('    # Scale features for GAN training (fit scaler on TRAIN ONLY)', '    # Fit the scaler on the train set and transform both splits'),
    ('    # Train GAN on malignant rows only', '    # Train the GAN on malignant rows from the train split'),
    ('    # Decide how many synthetic malignant samples to make:', '    # Decide how many malignant samples to synthesize'),
    ('    # Goal: balance classes in TRAIN set.', '    # Goal: balance the train labels'),
    ('    # Inverse scale back to original feature space', '    # Undo the scaling to recover the original feature space'),
    ('    # Build a DataFrame for synthetic malignant rows', '    # Build a DataFrame that stores the malignant synthetic rows'),
    ('    # Save ONLY the synthetic rows as a CSV so your teammate can merge as needed', '    # Save these synthetic rows as a CSV for the teem'),
    ("    # We'll map 1 back to 'M' to keep the file human-readable", '    # Map value 1 to M so the file stays readable'),
    ('    # Build two training sets for evaluation:', '    # Build two train sets for scoring tasks'),
    ('    # (A) Original', '    # Set A keeps the original train rows'),
    ('    # (B) GAN augmented (append synthetic malignant rows)', '    # Set B adds the malignant synthetic rowes'),
    ('    # Re-encode to 0/1 for classifier training', '    # Encode diagnosis into 0 and 1 for the classifier'),
    ('    # Evaluate Logistic Regression as a class-friendly downstream model', '    # Score Logistic Regression on the base and GAN sets')
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(f"Missing expected text: {old!r}")
    if '\n' in new:
        new = new.replace('\n', '\r\n')
    text = text.replace(old, new)

inline_replacements = [
    ('labels = y_mal_t[idx]  # should be all 1s', 'labels = y_mal_t[idx]  # labels stay at 1'),
    ('cfg.device = "cpu"  # keep CPU-friendly for class envs', 'cfg.device = "cpu"  # use cpu to match the class env'),
    ('    to_add = min(to_add, 1000)  # safety cap', '    to_add = min(to_add, 1000)  # Cap to stop runaway synth count'),
    ('    synth_df.insert(0, "diagnosis", 1)  # 1 = malignant', '    synth_df.insert(0, "diagnosis", 1)  # Value 1 marks malignant')
]

for old, new in inline_replacements:
    if old not in text:
        raise RuntimeError(f"Missing inline text: {old!r}")
    text = text.replace(old, new)

path.write_text(text)
