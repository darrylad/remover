# Remover

Go to the project root directory in Terminal and run:

### Setup Python venv

```bash
# Create a virtual environment:
python -m venv .venv

# Activate the virtual environment:
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`
```

### Get Requirements

```bash
pip install -r requirements.txt
```
This will install the required packages into your virtual environment.

### Run


```bash
python main.py <PATH_TO_YOUR_CSV_FILE>
```

Replace `<PATH_TO_YOUR_CSV_FILE>` with the path to your CSV file containing the data you want to open.

Your source file will not be modified. Output will be saved to the outputs folder in the project root directory.

### Deactivate venv

```bash
# When done, deactivate the virtual environment:
deactivate
```

### Screenshots

<img width="1512" height="1178" alt="Selection on time-series plot" src="https://github.com/user-attachments/assets/5e2501a4-06df-4b19-9b7b-b08c9a14f8f9" />

<img width="1512" height="1176" alt="Welch Fourier Transform plot after removal" src="https://github.com/user-attachments/assets/f41e3fc2-965d-4804-9f52-c3d5cf8814c8" />

<img width="1512" height="1176" alt="Welch Fourier Transform plot before removal" src="https://github.com/user-attachments/assets/9b3aa4b0-a5d7-499e-8b68-6dbeac42df92" />

<img width="1512" height="1176" alt="Welch Fourier Transform logarithmic plot" src="https://github.com/user-attachments/assets/26ddc7c9-8ffe-4d4e-a47c-ac144d4ff33b" />
