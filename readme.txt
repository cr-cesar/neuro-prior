# Para correr desde el main

Primero debes ir a tu root .../Debbuging Neuro

conda create -n debugging python=3.12 -y
conda activate debugging
which Python

pip install torch torchvision torchaudio

pip install -r requirements.txt
python main.py

### Línea de Comandos

```
# Ejecución básica
python main.py

# Con opciones personalizadas
python main.py --n_cases 50 --vae_epochs 30 --do_steps 200 --output_dir Data

# Sin visualizaciones (más rápido)
python main.py --skip_plots

# Con caché de HuggingFace personalizado
python main.py --cache_dir ~/hf-datasets-cache

```



# Para correr en Jupyter notebook

conda create -n debugging python=3.12 -y
conda activate debugging
which python
pip install -r requirements.txt
pip install jupyter
jupyter notebook


Nota: Una vez en el notebook cambiar estas dos líneas:

os.environ["HF_DATASETS_CACHE"] = "/Users/tu_usuario/hf-datasets-cache" (carpeta local donde se guardará la data)
ROOT = "/Users/tu_usuario/path/to/neuro_tabpfn" (ruta al proyecto)

Y en la celda de configuración cambiar N_CASES = 149 a un número menor (por ejemplo 50) para pruebas rápidas.