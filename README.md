# Remover

Go to the project root directory and run:

### Setup Python venv

```bash
python -m venv .venv

# Activate the virtual environment:
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Deactivate the virtual environment when done:
deactivate
```

### Get Requirements

```bash
pip install -r requirements.txt
```

### Run


```bash
python main.py <PATH_TO_YOUR_CSV_FILE>
```

Replace `<PATH_TO_YOUR_CSV_FILE>` with the path to your CSV file containing the data you want to open.

Your source file will not be modified. Output will be saved to the outputs folder in the project root directory.