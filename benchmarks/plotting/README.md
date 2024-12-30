## plotting

If you use Nix, you can enter the developement environment:
```bash
nix develop
```

Create a virtual python environment:
```bash
python3 -m venv .venv
```

Enter the python edevelopement environment
```bash
source .venv/bin/activate
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Create a kernel for jupyter:
```bash
python -m ipykernel install --user --name=venv
```

Adjust yout python path:
```bash
export PYTHONPATH="$(pwd)"
```

Run jupyter notebook:
```bash
jupyter lab
```
