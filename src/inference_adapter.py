"""
Script d'inférence avec le modèle fine-tuné (adapters)
Génère finetuned_transcriptions.txt pour le challenge
"""

import os
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
from pydub import AudioSegment
from jiwer import wer

from adapters import (
    AdapterConfig,
    inject_adapters_into_model
)

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
DATA_DIR = "E:/projet_indabax/src/dataset"
BASELINE_DIR = "./baseline_results"
ADAPTER_DIR = "./adapter_results"
OUTPUT_DIR = "./final_results"

BATCH_SIZE = 16
EXTRACT_DIR_TEST = "./extracted_audio_test"
EXTRACT_DIR_VAL = "./extracted_audio_val"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
def load_manifests(manifest_dir, require_text=False):
    """Charge les manifests"""
    all_data = []
    for manifest_file in sorted(Path(manifest_dir).glob("manifest_*.json")):
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if require_text:
                        if data.get('text'):
                            all_data.append(data)
                    else:
                        all_data.append(data)
                except:
                    continue
    return all_data

def load_audio_pydub(audio_path, sr=16000):
    """Charge un fichier audio"""
    try:
        audio = AudioSegment.from_file(audio_path, format="webm")
        if audio.channels > 1:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sr)
        samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
    except:
        return None

# ==========================================
# MAIN
# ==========================================
def main():
    print("="*50)
    print("🎯 INFÉRENCE AVEC MODÈLE FINE-TUNÉ")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Chargement processor
    print("\n📥 Chargement du processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(BASELINE_DIR, "vocab.json"),
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
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Chargement config training
    print("\n📋 Chargement de la configuration...")
    with open(os.path.join(ADAPTER_DIR, "training_config.json"), "r") as f:
        training_config = json.load(f)
    
    print(f"   Type: {training_config['adapter_type']}")
    print(f"   Bottleneck: {training_config['bottleneck_dim']}")
    print(f"   WER entraînement: {training_config['final_wer']:.2%}")
    
    # Chargement modèle
    print("\n🤖 Chargement du modèle...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # Réinjection des adapters (même config qu'à l'entraînement)
    print("\n💉 Injection des adapters...")
    adapter_config = AdapterConfig(
        adapter_type=training_config['adapter_type'],
        bottleneck_dim=training_config['bottleneck_dim'],
        activation="relu",
        dropout=0.1
    )
    inject_adapters_into_model(model, adapter_config)
    
    # Chargement des poids fine-tunés
    print("\n📦 Chargement des poids fine-tunés...")
    adapter_weights = torch.load(
        os.path.join(ADAPTER_DIR, "best_adapter_weights.pt"),
        map_location=device
    )
    
    # Chargement sélectif (seulement les adapters)
    model_dict = model.state_dict()
    adapter_weights_filtered = {
        k: v for k, v in adapter_weights.items() if k in model_dict
    }
    model_dict.update(adapter_weights_filtered)
    model.load_state_dict(model_dict)
    
    print(f"   ✅ {len(adapter_weights_filtered)} paramètres chargés")
    
    model.to(device)
    model.eval()
    
    # ==========================================
    # ÉVALUATION SUR VALIDATION SET
    # ==========================================
    print("\n" + "="*50)
    print("📊 ÉVALUATION SUR VALIDATION SET")
    print("="*50)
    
    val_manifests = load_manifests(
        os.path.join(DATA_DIR, "val_tarred/sharded_manifests_with_image"),
        require_text=True
    )
    
    print(f"Échantillons validation: {len(val_manifests)}")
    
    val_predictions = []
    val_references = []
    val_filepaths = []
    
    # Batch processing
    batch_speeches = []
    batch_refs = []
    batch_files = []
    
    for item in tqdm(val_manifests, desc="Validation"):
        audio_path = os.path.join(EXTRACT_DIR_VAL, item['audio_filepath'])
        
        if not os.path.exists(audio_path):
            continue
        
        speech = load_audio_pydub(audio_path)
        if speech is None:
            continue
        
        batch_speeches.append(speech)
        batch_refs.append(item['text'])
        batch_files.append(item['audio_filepath'])
        
        if len(batch_speeches) == BATCH_SIZE:
            with torch.no_grad():
                input_values = processor(
                    batch_speeches,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(device)
                
                logits = model(input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)
                preds = processor.batch_decode(pred_ids)
            
            val_predictions.extend(preds)
            val_references.extend(batch_refs)
            val_filepaths.extend(batch_files)
            
            batch_speeches, batch_refs, batch_files = [], [], []
    
    # Dernier batch
    if batch_speeches:
        with torch.no_grad():
            input_values = processor(
                batch_speeches,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values.to(device)
            
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(pred_ids)
        
        val_predictions.extend(preds)
        val_references.extend(batch_refs)
        val_filepaths.extend(batch_files)
    
    # Calcul WER
    val_wer = wer(val_references, val_predictions)
    print(f"\n📉 WER Fine-tuned: {val_wer:.2%}")
    
    # Sauvegarde validation
    with open(os.path.join(OUTPUT_DIR, "finetuned_transcriptions_val.txt"), "w", encoding='utf-8') as f:
        for audio, ref, pred in zip(val_filepaths, val_references, val_predictions):
            f.write(f"Audio: {audio}\n")
            f.write(f"Ref:  {ref}\n")
            f.write(f"Pred: {pred}\n\n")
    
    # ==========================================
    # INFÉRENCE SUR TEST SET
    # ==========================================
    print("\n" + "="*50)
    print("🎯 INFÉRENCE SUR TEST SET")
    print("="*50)
    
    test_manifests = load_manifests(
        os.path.join(DATA_DIR, "test_tarred/sharded_manifests_with_image"),
        require_text=False
    )
    
    print(f"Échantillons test: {len(test_manifests)}")
    
    test_predictions = []
    test_filepaths = []
    
    batch_speeches = []
    batch_files = []
    
    for item in tqdm(test_manifests, desc="Test"):
        audio_path = os.path.join(EXTRACT_DIR_TEST, item['audio_filepath'])
        
        if not os.path.exists(audio_path):
            continue
        
        speech = load_audio_pydub(audio_path)
        if speech is None:
            continue
        
        batch_speeches.append(speech)
        batch_files.append(item['audio_filepath'])
        
        if len(batch_speeches) == BATCH_SIZE:
            with torch.no_grad():
                input_values = processor(
                    batch_speeches,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(device)
                
                logits = model(input_values).logits
                pred_ids = torch.argmax(logits, dim=-1)
                preds = processor.batch_decode(pred_ids)
            
            test_predictions.extend(preds)
            test_filepaths.extend(batch_files)
            
            batch_speeches, batch_files = [], []
    
    # Dernier batch
    if batch_speeches:
        with torch.no_grad():
            input_values = processor(
                batch_speeches,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values.to(device)
            
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            preds = processor.batch_decode(pred_ids)
        
        test_predictions.extend(preds)
        test_filepaths.extend(batch_files)
    
    # Sauvegarde test (format requis pour le challenge)
    with open(os.path.join(OUTPUT_DIR, "finetuned_transcriptions.txt"), "w", encoding='utf-8') as f:
        for audio, pred in zip(test_filepaths, test_predictions):
            f.write(f"{pred}\n")
    
    # Format CSV aussi
    df = pd.DataFrame({
        'audio_filepath': test_filepaths,
        'prediction': test_predictions
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "finetuned_predictions.csv"), index=False)
    
    # ==========================================
    # RAPPORT FINAL
    # ==========================================
    print("\n" + "="*50)
    print("📊 RAPPORT FINAL")
    print("="*50)
    
    print(f"\n✅ Validation:")
    print(f"   Exemples: {len(val_predictions)}")
    print(f"   WER: {val_wer:.2%}")
    
    print(f"\n✅ Test:")
    print(f"   Exemples: {len(test_predictions)}")
    print(f"   Prédictions: {OUTPUT_DIR}/finetuned_transcriptions.txt")
    
    # Comparaison avec baseline si disponible
    baseline_file = os.path.join(BASELINE_DIR, "base_transcriptions.txt")
    if os.path.exists(baseline_file):
        # Lire le WER baseline si sauvegardé, sinon utiliser valeur par défaut
        baseline_wer = 1.00  # ⚠️ METTEZ VOTRE WER BASELINE RÉEL ICI
        
        improvement = baseline_wer - val_wer
        improvement_pct = (improvement / baseline_wer) * 100 if baseline_wer > 0 else 0
        
        print(f"\n📈 Amélioration vs Baseline:")
        print(f"   Baseline WER: {baseline_wer:.2%}")
        print(f"   Fine-tuned WER: {val_wer:.2%}")
        print(f"   Amélioration absolue: {improvement:.2%}")
        print(f"   Amélioration relative: {improvement_pct:.1f}%")    
    
    print("\n🎉 Inférence terminée!")
    print(f"📂 Fichiers générés dans: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()