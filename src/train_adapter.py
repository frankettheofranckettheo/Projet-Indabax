"""
Script de fine-tuning avec Adapters pour ASR Kinyarwanda
Respecte les contraintes du challenge:
- Base model gelé
- Entraînement uniquement des adapters
- Évaluation sur validation set
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
from jiwer import wer
from pydub import AudioSegment

# Import du module adapters
from adapters import (
    AdapterConfig,
    inject_adapters_into_model,
    freeze_base_model,
    count_trainable_parameters
)
from functools import partial


# ==========================================
# CONFIGURATION
# ==========================================
class TrainingConfig:
    # Chemins
    DATA_DIR = "E:/projet_indabax/src/dataset"
    OUTPUT_DIR = "./adapter_results"
    BASE_MODEL_DIR = "./baseline_results/base_model_config"
    VOCAB_PATH = "./baseline_results/vocab.json"
    
    # Modèle
    MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
    
    # Adapter config
    ADAPTER_TYPE = "bottleneck"  # "bottleneck" ou "parallel"
    BOTTLENECK_DIM = 64  # Dimension du goulot (16, 32, 64, 128)
    ADAPTER_ACTIVATION = "relu"  # "relu", "gelu", "tanh"
    ADAPTER_DROPOUT = 0.1
    ADAPTER_LAYERS = [20, 21, 22, 23]  # None = tous les layers, ou [18,19,20,21,22,23] pour les derniers
    
    # Entraînement
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-4  # Plus élevé que le fine-tuning normal (adapters seulement)
    NUM_EPOCHS = 5
    WARMUP_STEPS = 500
    GRADIENT_CLIP = 1.0
    
    # Évaluation
    EVAL_STEPS = 500  # Évaluer tous les N steps
    SAVE_STEPS = 1000
    
    # Data
    MAX_DURATION = 20.0  # Secondes (filtrer les trop longs)
    SAMPLE_RATE = 16000
    
    # Optimisation
    NUM_WORKERS = 0
    USE_FP16 = False  # Mixed precision (si GPU avec Tensor Cores)

config = TrainingConfig()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# ==========================================
# DATASET CUSTOM
# ==========================================
class KinyarwandaDataset(Dataset):
    """Dataset pour audio Kinyarwanda"""
    
    def __init__(self, manifest_path, audio_dir, processor, max_duration=20.0):
        self.audio_dir = audio_dir
        self.processor = processor
        self.max_duration = max_duration
        
        # Chargement des manifests
        self.data = []
        for manifest_file in sorted(Path(manifest_path).glob("manifest_*.json")):
            with open(manifest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if item.get('text') and item.get('duration', 0) < max_duration:
                            self.data.append(item)
                    except:
                        continue
        
        print(f"📊 Dataset chargé: {len(self.data)} exemples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Chargement audio
        audio_path = os.path.join(self.audio_dir, item['audio_filepath'])
        try:
            audio = AudioSegment.from_file(audio_path, format="webm")
            if audio.channels > 1:
                audio = audio.set_channels(1)
            audio = audio.set_frame_rate(config.SAMPLE_RATE)
            
            samples = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
        except Exception as e:
            # Fallback sur silence si erreur
            samples = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        
        # Tokenization du texte
        #with self.processor.as_target_processor():
        #    labels = self.processor(item['text']).input_ids

        # Tokenization du texte
        labels = self.processor.tokenizer(text=item['text']).input_ids
        
        return {
            'input_values': samples,
            'labels': labels
        }

# ==========================================
# COLLATE FUNCTION
# ==========================================
def collate_fn_global(batch, processor):
    # batch est une liste de dictionnaires venant de __getitem__
    # Structure: [{'input_values': numpy_array, 'labels': list_ids}, ...]

    # 1. Séparer inputs et labels
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 2. Padding de l'audio (input_values)
    # Le processor gère intelligemment les listes de numpy arrays
    batch_features = processor.feature_extractor.pad(
        {"input_values": input_values},
        padding=True,
        return_tensors="pt"
    )

    # 3. Padding des labels (texte)
    # On utilise le tokenizer pour padder les séquences d'IDs
    labels_batch = processor.tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt"
    )

    # 4. Masquage du padding pour la Loss CTC
    # On remplace le padding_token_id par -100 pour que PyTorch l'ignore lors du calcul de l'erreur
    labels_final = labels_batch["input_ids"].masked_fill(
        labels_batch.attention_mask.ne(1), -100
    )

    # 5. Retourner le dictionnaire final attendu par le modèle
    return {
        'input_values': batch_features['input_values'],
        'labels': labels_final
    }

# ==========================================
# ÉVALUATION
# ==========================================
@torch.no_grad()
def evaluate(model, dataloader, processor, device):
    """Évaluation sur validation set"""
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Évaluation"):
        input_values = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        outputs = model(input_values, labels=labels)
        total_loss += outputs.loss.item()
        
        # Décodage
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Conversion texte
        predictions = processor.batch_decode(pred_ids)
        references = processor.batch_decode(
            labels.cpu().numpy(),
            group_tokens=False  # Important pour CTC
        )
        
        # Nettoyage (retirer -100 tokens)
        references = [ref.replace("[PAD]", "").replace("[UNK]", "").strip() for ref in references]
        
        all_predictions.extend(predictions)
        all_references.extend(references)
    
    # Calcul métriques
    avg_loss = total_loss / len(dataloader)
    word_error_rate = wer(all_references, all_predictions)
    
    model.train()
    return {
        'loss': avg_loss,
        'wer': word_error_rate
    }

# ==========================================
# MAIN TRAINING
# ==========================================
def main():
    print("="*50)
    print("🚀 FINE-TUNING AVEC ADAPTERS")
    print("="*50)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Chargement processor
    print("\n📥 Chargement du processor...")
    tokenizer = Wav2Vec2CTCTokenizer(
        config.VOCAB_PATH,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config.SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    global processor
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Chargement modèle
    print("\n🤖 Chargement du modèle de base...")
    model = Wav2Vec2ForCTC.from_pretrained(
        config.MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # Configuration adapters
    print("\n🔧 Configuration des adapters...")
    adapter_config = AdapterConfig(
        adapter_type=config.ADAPTER_TYPE,
        bottleneck_dim=config.BOTTLENECK_DIM,
        activation=config.ADAPTER_ACTIVATION,
        dropout=config.ADAPTER_DROPOUT,
        adapter_layers=config.ADAPTER_LAYERS
    )
    print(f"   {adapter_config}")
    
    # Injection des adapters
    print("\n💉 Injection des adapters dans le modèle...")
    inject_adapters_into_model(model, adapter_config, model_type="wav2vec2")
    
    # Gel du modèle de base
    print("\n❄️  Gel du modèle de base...")
    freeze_base_model(model)
    
    # Statistiques
    print("\n📊 Statistiques des paramètres:")
    stats = count_trainable_parameters(model)
    print(f"   Total: {stats['total']:,}")
    print(f"   Entraînables: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
    print(f"   Gelés: {stats['frozen']:,}")
    
    model.to(device)
    
    # Datasets
    print("\n📂 Chargement des datasets...")
    train_dataset = KinyarwandaDataset(
        manifest_path=os.path.join(config.DATA_DIR, "train_tarred/sharded_manifests_with_image"),
        audio_dir="./extracted_audio_train",  # Vous devrez extraire le train aussi
        processor=processor,
        max_duration=config.MAX_DURATION
    )
    
    val_dataset = KinyarwandaDataset(
        manifest_path=os.path.join(config.DATA_DIR, "val_tarred/sharded_manifests_with_image"),
        audio_dir="./extracted_audio_val",
        processor=processor,
        max_duration=config.MAX_DURATION
    )


    train_collate = partial(collate_fn_global, processor=processor)
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=train_collate,
        pin_memory=torch.cuda.is_available()
    )
    
    val_collate = partial(collate_fn_global, processor=processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=val_collate,
        pin_memory=torch.cuda.is_available()
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # Scheduler avec warmup
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=config.WARMUP_STEPS / total_steps,
        anneal_strategy='cos'
    )
    
    # Training loop
    print("\n" + "="*50)
    print("🏋️  DÉBUT DE L'ENTRAÎNEMENT")
    print("="*50)
    
    best_wer = float('inf')
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n📅 Epoch {epoch+1}/{config.NUM_EPOCHS}")
        model.train()
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.GRADIENT_CLIP
            )
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Évaluation périodique
            if global_step % config.EVAL_STEPS == 0:
                print(f"\n📊 Évaluation (step {global_step})...")
                metrics = evaluate(model, val_loader, processor, device)
                print(f"   Val Loss: {metrics['loss']:.4f}")
                print(f"   Val WER: {metrics['wer']:.2%}")
                
                # Sauvegarde du meilleur modèle
                if metrics['wer'] < best_wer:
                    best_wer = metrics['wer']
                    print(f"   ✅ Nouveau meilleur WER! Sauvegarde...")
                    
                    # Sauvegarde adapters uniquement
                    adapter_state = {
                        name: param for name, param in model.named_parameters()
                        if "adapter" in name or "lm_head" in name
                    }
                    torch.save(
                        adapter_state,
                        os.path.join(config.OUTPUT_DIR, "best_adapter_weights.pt")
                    )
        
        # Fin d'époque
        avg_loss = epoch_loss / len(train_loader)
        print(f"\n📈 Epoch {epoch+1} - Loss moyenne: {avg_loss:.4f}")
    
    # Évaluation finale
    print("\n" + "="*50)
    print("🏁 ÉVALUATION FINALE")
    print("="*50)
    
    final_metrics = evaluate(model, val_loader, processor, device)
    print(f"📉 WER final: {final_metrics['wer']:.2%}")
    print(f"📉 Loss final: {final_metrics['loss']:.4f}")
    
    # Sauvegarde finale
    print("\n💾 Sauvegarde des résultats...")
    
    # Sauvegarder config
    with open(os.path.join(config.OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump({
            'adapter_type': config.ADAPTER_TYPE,
            'bottleneck_dim': config.BOTTLENECK_DIM,
            'num_epochs': config.NUM_EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'final_wer': final_metrics['wer'],
            'trainable_params': stats['trainable'],
            'total_params': stats['total']
        }, f, indent=2)
    
    print("\n✅ Entraînement terminé!")
    print(f"📂 Résultats sauvegardés dans: {config.OUTPUT_DIR}")
    print(f"🎯 Meilleur WER: {best_wer:.2%}")

if __name__ == "__main__":
    main()