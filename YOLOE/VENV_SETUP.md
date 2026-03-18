# Python Virtual Environment Setup

Run these commands from the YOLOE folder.

## Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Windows (cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify install

```powershell
python -c "import ultralytics, sentence_transformers, torch, psutil, cv2, PIL; print('OK')"
```

## Deactivate when done

```powershell
deactivate
```
