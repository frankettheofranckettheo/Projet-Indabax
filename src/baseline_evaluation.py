import os
import torch
import json
import pandas as pd
import numpy as np
import tarfile
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from jiwer import wer
from tqdm import tqdm
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor, as_completed

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.units import cm
import matplotlib.pyplot as plt



# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_ID = "facebook/wav2vec2-large-xlsr-53" 
LOCAL_DATA_DIR = "E:/projet_indabax/src/dataset"
OUTPUT_DIR = "./baseline_results"
EXTRACT_DIR = "./extracted_audio_test"  # Dossier pour extraire tous les audios


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


# ==========================================
# 2. EXTRACTION ET CHARGEMENT DES DONNÉES
# ==========================================
print("📥 Chargement des manifests...")

def load_manifests(manifest_dir, require_text=True):
    """Charge tous les fichiers manifest JSON d'un dossier"""
    all_data = []
    manifest_files = sorted(Path(manifest_dir).glob("manifest_*.json"))
    
    print(f"Fichiers manifest trouvés: {len(manifest_files)}")

    for manifest_file in manifest_files:
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    if require_text:
                        # Pour le train : on garde seulement si text existe
                        if data.get('text') and isinstance(data.get('text'), str):
                            all_data.append(data)
                    else:
                        # Pour le test : on garde tout (même sans text)
                        all_data.append(data)
                        
                except json.JSONDecodeError as e:
                    print(f"Erreur JSON dans {manifest_file}: {e}")
                    continue    
    return all_data

def inspect_tar_contents(tar_path, max_files=5):
    """Inspecte le contenu d'une archive tar pour debug"""
    print(f"\n🔍 Inspection de {os.path.basename(tar_path)}:")
    try:
        with tarfile.open(tar_path, 'r:xz') as tar:
            members = tar.getmembers()
            print(f"   Nombre de fichiers: {len(members)}")
            print(f"   Premiers fichiers:")
            for i, member in enumerate(members[:max_files]):
                print(f"      - {member.name}")
            return [m.name for m in members]
    except Exception as e:
        print(f"   Erreur: {e}")
        return []


print("📥 Extraction des archives audio (une seule fois)...")

def extract_all_tar_archives(tar_dir, extract_to):
    """Extrait toutes les archives tar.xz d'un coup"""
    tar_files = list(Path(tar_dir).glob("*.tar.xz"))
    
    if not tar_files:
        print(f"⚠️ Aucune archive trouvée dans {tar_dir}")
        return False
    
    print(f"📦 {len(tar_files)} archives à extraire...")
    
    for tar_path in tqdm(tar_files, desc="Extraction archives"):
        try:
            with tarfile.open(tar_path, 'r:xz') as tar:
                tar.extractall(path=extract_to)
        except Exception as e:
            print(f"Erreur extraction {tar_path}: {e}")
    
    return True

# Extraction des archives test
test_audio_dir = os.path.join(LOCAL_DATA_DIR, "test_tarred/sharded_manifests_with_image/audio_shards")

# Vérifier si déjà extrait
audio_files_exist = len(list(Path(EXTRACT_DIR).glob("*.webm"))) > 0

if not audio_files_exist:
    print("🔄 Première extraction des archives...")
    extract_all_tar_archives(test_audio_dir, EXTRACT_DIR)
    print("✅ Extraction terminée!")
else:
    print("✅ Fichiers audio déjà extraits, réutilisation...")

# Chargement des données train (avec text requis) et test (sans text requis)
train_manifests = load_manifests(
    os.path.join(LOCAL_DATA_DIR, "train_tarred/sharded_manifests_with_image"),
    require_text=True
)
test_manifests = load_manifests(
    os.path.join(LOCAL_DATA_DIR, "test_tarred/sharded_manifests_with_image"),
    require_text=False  # On accepte les entrées sans text pour le test
)
val_manifests = load_manifests(
    os.path.join(LOCAL_DATA_DIR, "val_tarred/sharded_manifests_with_image"),
    require_text=True  # IMPORTANT: On veut les références
)

print(f"📊 Train: {len(train_manifests)} instances")
print(f"📊 Test: {len(test_manifests)} instances")
print(f"📊 Val: {len(val_manifests)} instances")

# Afficher quelques exemples de données test
if len(test_manifests) > 0:
    print("\n📋 Exemple de données test:")
    example = test_manifests[0]
    for key, value in example.items():
        print(f"   {key}: {value}")

if len(train_manifests) == 0:
    print("❌ Aucune donnée d'entraînement chargée. Vérifiez les manifests.")
    exit()

if len(test_manifests) == 0:
    print("❌ Aucune donnée de test chargée. Vérifiez les manifests.")
    exit()

if len(val_manifests) == 0:
    print("❌ Aucune donnée de validation chargée. Vérifiez les manifests.")
    exit()


# Inspection d'une archive test pour comprendre sa structure
test_audio_tar = os.path.join(
    LOCAL_DATA_DIR, 
    "test_tarred/sharded_manifests_with_image/audio_shards",
    "audio_0.tar.xz"
)
if os.path.exists(test_audio_tar):
    tar_contents = inspect_tar_contents(test_audio_tar)

# Inspection d'une archive validation pour comprendre sa structure
val_audio_tar = os.path.join(
    LOCAL_DATA_DIR, 
    "val_tarred/sharded_manifests_with_image/audio_shards",
    "audio_0.tar.xz"
)
if os.path.exists(val_audio_tar):
    tar_contents = inspect_tar_contents(val_audio_tar)

# ==========================================
# 3. CRÉATION DU VOCABULAIRE (TOKENIZER)
# ==========================================
print("\n🔤 Création du vocabulaire...")

# Extraction de tous les caractères uniques du train set
all_texts = [item['text'] for item in train_manifests]
print(f"📝 {len(all_texts)} textes valides pour le vocabulaire")

# Vérification qu'il y a bien des textes
if len(all_texts) == 0:
    print("❌ Aucun texte trouvé pour créer le vocabulaire.")
    exit()

vocab_list = list(set(" ".join(all_texts)))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

# Tokens spéciaux CTC
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# Sauvegarde du vocabulaire
vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding='utf-8') as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

print(f"✅ Vocabulaire créé avec {len(vocab_dict)} tokens")

# Création du Processor
tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path, 
    unk_token="[UNK]", 
    pad_token="[PAD]", 
    word_delimiter_token="|"
)
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, 
    sampling_rate=16000, 
    padding_value=0.0, 
    do_normalize=True, 
    return_attention_mask=True
)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# ==========================================
# 4. CHARGEMENT DU MODÈLE DE BASE
# ==========================================
print(f"\n🤖 Chargement du modèle {MODEL_ID}...")

model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)

# Sauvegarde de la configuration
model.config.save_pretrained(os.path.join(OUTPUT_DIR, "base_model_config"))
print(f"✅ Configuration du modèle sauvegardée")

# GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
model.to(device)
model.eval()


# ====================================
# 4.1 DIAGNOSTIC
# ====================================

# Ajoutez ce code de diagnostic juste après l'extraction des archives

print("\n🔍 DIAGNOSTIC:")
print(f"📁 Dossier d'extraction: {EXTRACT_DIR}")
print(f"   Existe? {os.path.exists(EXTRACT_DIR)}")

# Liste les fichiers extraits
extracted_files = list(Path(EXTRACT_DIR).rglob("*"))
audio_files = [f for f in extracted_files if f.suffix in ['.webm', '.wav', '.mp3']]

print(f"📦 Fichiers extraits totaux: {len(extracted_files)}")
print(f"🎵 Fichiers audio trouvés: {len(audio_files)}")

if len(audio_files) > 0:
    print(f"\n📝 Exemples de fichiers extraits:")
    for f in audio_files[:5]:
        print(f"   - {f}")
    
    print(f"\n📋 Structure des chemins:")
    example_file = audio_files[0]
    print(f"   Chemin complet: {example_file}")
    print(f"   Chemin relatif: {example_file.relative_to(EXTRACT_DIR)}")
else:
    print("⚠️ Aucun fichier audio trouvé!")
    print("\n📂 Contenu du dossier d'extraction:")
    for item in Path(EXTRACT_DIR).iterdir():
        print(f"   - {item.name} ({'dir' if item.is_dir() else 'file'})")

# Vérifier le premier manifest
if len(val_manifests) > 0:
    print(f"\n📋 Premier manifest:")
    first_item = val_manifests[0]
    print(f"   audio_filepath: {first_item['audio_filepath']}")
    
    # Tester le chemin construit
    val_path = os.path.join(EXTRACT_DIR, first_item['audio_filepath'])
    print(f"   Chemin construit: {val_path}")
    print(f"   Existe? {os.path.exists(val_path)}")


# ==========================================
# 5. INFÉRENCE ET CALCUL WER
# ==========================================

print("\n Fonction de chargement audio ameliore")
def load_audio_pydub(audio_path, sr=16000):
    """Charge un fichier audio avec pydub (supporte webm)"""
    try:
        # Charge avec pydub
        audio = AudioSegment.from_file(audio_path)
        
        # Convertit en mono si stéréo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample à 16kHz
        audio = audio.set_frame_rate(sr)
        
        # Convertit en numpy array float32
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        
        # Normalise entre -1 et 1
        if audio.sample_width == 2:  # 16-bit
            samples = samples / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            samples = samples / 2147483648.0
        else:
            samples = samples / (2 ** (8 * audio.sample_width - 1))
        
        return samples
    except Exception as e:
        print(f"Erreur chargement {audio_path}: {e}")
        return None

BATCH_SIZE = 4

predictions = []
references = []
audio_filepaths = []  # Pour garder trace des fichiers traités
errors = {"file_not_found": 0, "load_failed": 0}

# Préparer les batchs
batch_speeches = []
batch_refs = []
batch_files = []



print("\n🚀 Démarrage de l'inférence sur le val set...")

# Limitation a 150 echantillons pour le test set
TARGET_SIZE = 150
if len(val_manifests) > TARGET_SIZE:
    print(f"⚠️ Mode rapide : Conservation des {TARGET_SIZE} premiers échantillons sur {len(val_manifests)}.")
    val_manifests = val_manifests[:TARGET_SIZE] 

# Traitement avec barre de progression
for item in tqdm(val_manifests, desc="Transcription"):
    audio_filename = item['audio_filepath']
    audio_path = os.path.join(EXTRACT_DIR, audio_filename)
    
    # Vérifier que le fichier existe
    if not os.path.exists(audio_path):
        errors["file_not_found"] += 1
        continue
    
    # Charger l'audio
    speech = load_audio_pydub(audio_path, sr=16000)
    
    if speech is None:
        errors["load_failed"] += 1
        continue
    
    batch_speeches.append(speech)
    batch_refs.append(item['text'])
    batch_files.append(audio_filename)

    # Traiter le batch quand il est plein
    if len(batch_speeches) == BATCH_SIZE:
        # Préparation des inputs en batch
        input_values = processor(
            batch_speeches,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values
        input_values = input_values.to(device)


        # Prédiction batch
        with torch.no_grad():
            logits = model(input_values).logits
        
        # Décodage
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)
        
        predictions.extend(transcription)
        references.extend(batch_refs)
        audio_filepaths.extend(batch_files)

        # Réinitialiser les batchs
        batch_speeches = []
        batch_refs = []
        batch_files = []

# Traiter le dernier batch incomplet
if len(batch_speeches) > 0:
    input_values = processor(
        batch_speeches,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    ).input_values
    input_values = input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(pred_ids)
    
    predictions.extend(transcriptions)
    references.extend(batch_refs)
    audio_filepaths.extend(batch_files)




# ==========================================
# 8. RÉSULTATS ET CALCUL DU WER
# ==========================================

def generate_pdf_report(
    output_path,
    final_wer,
    errors,
    cleaned_filepaths,
    cleaned_references,
    cleaned_predictions,
    save_dir
):
    pdf_path = os.path.join(save_dir, output_path)

    styles = getSampleStyleSheet()
    elements = []

    # Titre
    elements.append(Paragraph("<b>RAPPORT D'ÉVALUATION BASELINE Wav2Vec2</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * cm))

    # WER
    elements.append(Paragraph(f"<b>WER Final :</b> {final_wer:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * cm))

    # Statistiques d'erreurs
    elements.append(Paragraph("<b>Statistiques d'erreurs :</b>", styles["Heading2"]))
    error_table_data = [["Type d'erreur", "Nombre"]]
    for k, v in errors.items():
        error_table_data.append([k, str(v)])

    error_table = Table(error_table_data)
    elements.append(error_table)
    elements.append(Spacer(1, 0.5 * cm))

    # Exemples de transcriptions
    elements.append(Paragraph("<b>Exemples de transcriptions :</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * cm))

    max_examples = min(10, len(cleaned_predictions))
    for i in range(max_examples):
        elements.append(Paragraph(f"<b>Fichier :</b> {cleaned_filepaths[i]}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Référence :</b> {cleaned_references[i]}", styles["Normal"]))
        elements.append(Paragraph(f"<b>Prédiction :</b> {cleaned_predictions[i]}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * cm))

    # === GRAPHIQUE LONGUEUR DES TRANSCRIPTIONS ===
    ref_lengths = [len(r.split()) for r in cleaned_references if r.strip()]
    pred_lengths = [len(p.split()) for p in cleaned_predictions if p.strip()]
    
    plt.figure()
    plt.hist(ref_lengths, alpha=0.7)
    plt.hist(pred_lengths, alpha=0.7)
    plt.title("Distribution longueurs REF vs PRED")
    plt.savefig(os.path.join(save_dir, "length_distribution.png"))
    plt.close()

    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph("<b>Distribution des longueurs de transcription :</b>", styles["Heading2"]))
    img = Image(os.path.join(save_dir, "length_distribution.png"), width=12*cm, height=7*cm)
    elements.append(img)

    # Génération du PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    doc.build(elements)

    print(f"✅ Rapport PDF généré : {pdf_path}")



print(f"\n📊 Statistiques d'erreurs:")
for error_type, count in errors.items():
    print(f"   {error_type}: {count}")

if len(predictions) == 0:
    print("\n❌ Aucune prédiction générée.")
    exit()

# VÉRIFICATION ET NETTOYAGE DES DONNÉES
print(f"\n🔍 Vérification des données...")
print(f"   Prédictions: {len(predictions)}")
print(f"   Références: {len(references)}")
print(f"   Fichiers: {len(audio_filepaths)}")

# Filtrer les None et vérifier la cohérence
cleaned_predictions = []
cleaned_references = []
cleaned_filepaths = []
has_valid_refs = False

for pred, ref, filepath in zip(predictions, references, audio_filepaths):
    # On garde la prédiction même si elle est vide (ça peut arriver)
    pred_str = str(pred) if pred is not None else ""
    
    # Gestion de la référence manquante (cas du Test Set)
    if ref is None:
        ref_str = "" 
    else:
        ref_str = str(ref)
        if ref_str.strip(): # Si on a du texte dans la ref
            has_valid_refs = True
    
    cleaned_predictions.append(pred_str)
    cleaned_references.append(ref_str)
    cleaned_filepaths.append(filepath)

print(f"✅ Données nettoyées: {len(cleaned_predictions)} exemples valides")

if has_valid_refs:
    try:
        # On ne garde que les paires où la ref n'est pas vide pour le WER
        wer_refs = []
        wer_preds = []
        for r, p in zip(cleaned_references, cleaned_predictions):
            if r.strip():
                wer_refs.append(r)
                wer_preds.append(p)
                
        if len(wer_refs) > 0:
            final_wer = wer(wer_refs, wer_preds)
            print(f"📉 WER estimé (sur références disponibles): {final_wer:.2%}")
        else:
            print("⚠️ Pas assez de références valides pour calculer un WER.")
            final_wer = None
    except Exception as e:
        print(f"⚠️ Erreur lors du calcul du WER: {e}")
        final_wer = None
else:
    print("ℹ️ Mode 'Test Set' détecté (pas de références): Calcul du WER ignoré.")
    final_wer = 0.0


# CALCUL DU WER
#final_wer = wer(cleaned_references, cleaned_predictions)

# Sauvegarde des transcriptions AVEC références
output_file = os.path.join(OUTPUT_DIR, "base_transcriptions.txt")
csv_output = os.path.join(OUTPUT_DIR, "submission.csv") 

with open(output_file, "w", encoding='utf-8') as f:
    for audio_file, ref, pred in zip(cleaned_filepaths, cleaned_references, cleaned_predictions):
        f.write(f"Audio: {audio_file}\n")
        f.write(f"Ref:  {ref if ref else '[PAS DE REF]'}\n")
        f.write(f"Pred: {pred}\n\n")

# Sauvegarde format CSV (Audio, Pred)
df = pd.DataFrame({
    'audio_filepath': cleaned_filepaths,
    'prediction': cleaned_predictions
})
df.to_csv(csv_output, index=False)

# Statistiques détaillées
print("\n" + "="*50)
print(f"🏁 ÉVALUATION BASELINE TERMINÉE")
print("="*50)
print(f"📊 Exemples traités: {len(cleaned_predictions)}")
print(f"📉 WER du modèle de base: {final_wer:.2%}")
print(f"📄 Transcriptions: {output_file}")
print(f"📄 Résultats (CSV): {csv_output}")
print(f"🔤 Vocabulaire: {len(vocab_dict)} tokens")
print(f"💾 Config modèle: {OUTPUT_DIR}/base_model_config")
print("="*50)

# Affichage de quelques exemples
print("\n📝 Exemples de transcriptions:\n")
for i in range(min(5, len(cleaned_predictions))):
    print(f"Exemple {i+1}: {cleaned_filepaths[i]}")
    print(f"  REF:  {cleaned_references[i]}")
    print(f"  PRED: {cleaned_predictions[i]}")
    print()

print(f"\n💡 Fichiers audio: {EXTRACT_DIR}")
print(f"💡 WER baseline: {final_wer:.2%}")
print("💡 Prochaine étape: Fine-tuner avec adapters!")


# ==========================================
# GÉNÉRATION DU RAPPORT PDF
# ==========================================

generate_pdf_report(
    output_path="rapport_baseline.pdf",
    final_wer=final_wer,
    errors=errors,
    cleaned_filepaths=cleaned_filepaths,
    cleaned_references=cleaned_references,
    cleaned_predictions=cleaned_predictions,
    save_dir=OUTPUT_DIR
)
