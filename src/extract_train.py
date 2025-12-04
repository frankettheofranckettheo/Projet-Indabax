"""
Script pour extraire les archives audio du train set
À exécuter AVANT le fine-tuning
"""

import os
import tarfile
from pathlib import Path
from tqdm import tqdm

DATA_DIR = "E:/projet_indabax/src/dataset"
EXTRACT_DIR = "./extracted_audio_train"

os.makedirs(EXTRACT_DIR, exist_ok=True)

def extract_all_tar_archives(tar_dir, extract_to):
    """Extrait toutes les archives tar.xz"""
    tar_files = list(Path(tar_dir).glob("*.tar.xz"))
    
    if not tar_files:
        print(f"⚠️ Aucune archive trouvée dans {tar_dir}")
        return False
    
    print(f"📦 {len(tar_files)} archives à extraire...")
    
    for tar_path in tqdm(tar_files, desc="Extraction"):
        try:
            with tarfile.open(tar_path, 'r:xz') as tar:
                tar.extractall(path=extract_to)
        except Exception as e:
            print(f"Erreur extraction {tar_path}: {e}")
    
    return True

# Extraction
train_audio_dir = os.path.join(DATA_DIR, "train_tarred/sharded_manifests_with_image/audio_shards")

# Vérifier si déjà extrait
audio_files = list(Path(EXTRACT_DIR).glob("*.webm"))

if len(audio_files) > 0:
    print(f"✅ {len(audio_files)} fichiers déjà extraits")
    print("   Supprimez le dossier pour ré-extraire")
else:
    print("🔄 Extraction des archives train...")
    extract_all_tar_archives(train_audio_dir, EXTRACT_DIR)
    
    # Vérification
    audio_files = list(Path(EXTRACT_DIR).glob("*.webm"))
    print(f"✅ {len(audio_files)} fichiers audio extraits")
    print(f"📁 Emplacement: {EXTRACT_DIR}")