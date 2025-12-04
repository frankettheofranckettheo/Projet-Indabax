"""
Générateur de rapport PDF pour le ASR Fellowship Challenge
À exécuter après l'inférence finale
"""

import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ==========================================
# CONFIGURATION
# ==========================================
BASELINE_DIR = "./baseline_results"
ADAPTER_DIR = "./adapter_results"
FINAL_DIR = "./final_results"
OUTPUT_PDF = os.path.join(FINAL_DIR, "rapport.pdf")

# ==========================================
# INFORMATIONS PERSONNELLES
# ==========================================

CANDIDATE_INFO = {
    "name": "NOUNDJEU NOUBISSIE FRANCK",
    "email": "ingenieurnoundjeu@gmail.com",
    "phone": "+237 651 11 99 62",
    "institution": "ECOLE NATIONALE SUPERIEURE POLYTECHNIQUE DE YAOUNDE (ENSPY)"
}

# ==========================================
# CHARGEMENT DES RÉSULTATS
# ==========================================
def load_results():
    """Charge tous les résultats des expériences"""
    
    results = {
        "candidate": CANDIDATE_INFO,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "baseline": {},
        "finetuned": {},
        "training": {},
        "comparison": {}
    }
    
    # Baseline WER (depuis le rapport terminal ou à extraire du fichier)
    results["baseline"]["wer"] = 1.00
    results["baseline"]["file"] = "base_transcriptions.txt"
    
    # Training config
    training_config_path = os.path.join(ADAPTER_DIR, "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            results["training"] = json.load(f)
    
    # Fine-tuned WER
    results["finetuned"]["wer"] = results["training"].get("final_wer", 0.0)
    results["finetuned"]["file"] = "finetuned_transcriptions.txt"
    
    # Comparaison
    baseline_wer = results["baseline"]["wer"]
    finetuned_wer = results["finetuned"]["wer"]
    improvement = baseline_wer - finetuned_wer
    improvement_pct = (improvement / baseline_wer) * 100 if baseline_wer > 0 else 0
    
    results["comparison"] = {
        "baseline_wer": baseline_wer,
        "finetuned_wer": finetuned_wer,
        "absolute_improvement": improvement,
        "relative_improvement": improvement_pct
    }
    
    return results

# ==========================================
# GÉNÉRATION DU PDF
# ==========================================
def generate_report():
    """Génère le rapport PDF complet"""
    
    # Chargement des résultats
    results = load_results()
    
    # Création du document
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Styles personnalisés
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c5282'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        leftIndent=20,
        spaceAfter=10,
        fontName='Courier'
    )
    
    # Contenu du document
    story = []
    
    # ==========================================
    # PAGE DE TITRE
    # ==========================================
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("ASR Fellowship Challenge", title_style))
    story.append(Paragraph("Adapter-Based Fine-Tuning for Low-Resource Languages", styles['Heading3']))
    story.append(Spacer(1, 0.5*inch))
    
    # Informations candidat
    story.append(Paragraph("Candidate Information", heading_style))
    candidate_data = [
        ["Name:", results["candidate"]["name"]],
        ["Email:", results["candidate"]["email"]],
        ["Phone:", results["candidate"]["phone"]],
        ["Institution:", results["candidate"]["institution"]],
        ["Date:", results["date"]]
    ]
    
    candidate_table = Table(candidate_data, colWidths=[1.5*inch, 4*inch])
    candidate_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(candidate_table)
    story.append(PageBreak())
    
    # ==========================================
    # RÉSUMÉ EXÉCUTIF
    # ==========================================
    story.append(Paragraph("Executive Summary", heading_style))
    
    summary_text = f"""
    This report presents the results of fine-tuning a Wav2Vec2-XLSR-53 model for Kinyarwanda 
    Automatic Speech Recognition using parameter-efficient adapter modules. The approach follows 
    the methodology described in Thomas et al. (2022) and Houlsby et al. (2019), injecting 
    lightweight bottleneck adapters while keeping the base model frozen.
    <br/><br/>
    <b>Key Results:</b><br/>
    • Baseline WER: {results['comparison']['baseline_wer']:.2%}<br/>
    • Fine-tuned WER: {results['comparison']['finetuned_wer']:.2%}<br/>
    • Absolute Improvement: {results['comparison']['absolute_improvement']:.2%}<br/>
    • Relative Improvement: {results['comparison']['relative_improvement']:.1f}%<br/>
    • Trainable Parameters: {results['training'].get('trainable_params', 0):,} 
      ({(results['training'].get('trainable_params', 0) / results['training'].get('total_params', 1) * 100):.2f}% of total)
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ==========================================
    # RÉSULTATS DÉTAILLÉS
    # ==========================================
    story.append(Paragraph("Detailed Results", heading_style))
    
    # Tableau de comparaison
    story.append(Paragraph("Performance Comparison", subheading_style))
    
    comparison_data = [
        ["Metric", "Baseline Model", "Fine-tuned Model", "Improvement"],
        [
            "Word Error Rate (WER)",
            f"{results['comparison']['baseline_wer']:.2%}",
            f"{results['comparison']['finetuned_wer']:.2%}",
            f"{results['comparison']['absolute_improvement']:.2%}"
        ],
        [
            "Relative Improvement",
            "-",
            "-",
            f"{results['comparison']['relative_improvement']:.1f}%"
        ]
    ]
    
    comparison_table = Table(comparison_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Paramètres du modèle
    story.append(Paragraph("Model Parameters", subheading_style))
    
    params_data = [
        ["Parameter Type", "Count", "Percentage"],
        [
            "Total Parameters",
            f"{results['training'].get('total_params', 0):,}",
            "100%"
        ],
        [
            "Trainable (Adapters + LM Head)",
            f"{results['training'].get('trainable_params', 0):,}",
            f"{(results['training'].get('trainable_params', 0) / results['training'].get('total_params', 1) * 100):.2f}%"
        ],
        [
            "Frozen (Base Model)",
            f"{results['training'].get('total_params', 0) - results['training'].get('trainable_params', 0):,}",
            f"{100 - (results['training'].get('trainable_params', 0) / results['training'].get('total_params', 1) * 100):.2f}%"
        ]
    ]
    
    params_table = Table(params_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    params_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(params_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ==========================================
    # ARCHITECTURE DES ADAPTERS
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("Adapter Architecture", heading_style))
    
    arch_text = f"""
    <b>Adapter Type:</b> {results['training'].get('adapter_type', 'bottleneck').capitalize()} Adapter 
    (Houlsby et al., 2019)<br/><br/>
    
    <b>Configuration:</b><br/>
    • Bottleneck Dimension: {results['training'].get('bottleneck_dim', 64)}<br/>
    • Activation Function: ReLU<br/>
    • Dropout Rate: 0.1<br/>
    • Adapter Layers: Last 4 layers (20, 21, 22, 23)<br/>
    • Injection Point: After Feed-Forward Network (FFN) in each Transformer layer<br/><br/>
    
    <b>Architecture Details:</b><br/>
    The adapter module follows a bottleneck architecture with the following structure:<br/>
    1. Layer Normalization (stabilization)<br/>
    2. Down-projection: 1024 → {results['training'].get('bottleneck_dim', 64)} (dimensionality reduction)<br/>
    3. ReLU activation (non-linearity)<br/>
    4. Dropout (p=0.1) (regularization)<br/>
    5. Up-projection: {results['training'].get('bottleneck_dim', 64)} → 1024 (restore dimensionality)<br/>
    6. Dropout (p=0.1)<br/>
    7. Residual connection (preserve base model information)<br/><br/>
    
    This architecture introduces only ~130,000 parameters per adapter module, while the base 
    Wav2Vec2-XLSR-53 model contains ~317 million parameters. By injecting adapters in only the 
    last 4 layers, we achieve parameter efficiency with ~520k trainable parameters total.
    """
    story.append(Paragraph(arch_text, body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ==========================================
    # STRATÉGIE D'ENTRAÎNEMENT
    # ==========================================
    story.append(Paragraph("Training Strategy", heading_style))
    
    training_text = f"""
    <b>Training Configuration:</b><br/>
    • Optimizer: AdamW (β1=0.9, β2=0.999, ε=1e-8, weight_decay=0.01)<br/>
    • Learning Rate: {results['training'].get('learning_rate', 5e-4)}<br/>
    • Scheduler: OneCycleLR with cosine annealing<br/>
    • Warmup Steps: 500<br/>
    • Number of Epochs: {results['training'].get('num_epochs', 10)}<br/>
    • Batch Size: 8<br/>
    • Gradient Clipping: Max norm = 1.0<br/>
    • Mixed Precision: {('Enabled' if results['training'].get('use_fp16', False) else 'Disabled')}<br/><br/>
    
    <b>Data Processing:</b><br/>
    • Training Set: ~176,000 audio samples (Kinyarwanda health domain)<br/>
    • Validation Set: Used for model selection and WER calculation<br/>
    • Audio Preprocessing: Resampled to 16kHz, mono channel<br/>
    • Max Audio Duration: 20 seconds (longer samples filtered)<br/>
    • Data Augmentation: SpecAugment (built into Wav2Vec2)<br/><br/>
    
    <b>Key Design Choices:</b><br/>
    1. <b>Frozen Base Model:</b> All Wav2Vec2 parameters frozen to preserve pre-trained knowledge<br/>
    2. <b>Selective Layer Adaptation:</b> Adapters injected only in top 4 layers (linguistic features)<br/>
    3. <b>Higher Learning Rate:</b> 5e-4 (vs 1e-5 for full fine-tuning) due to fewer parameters<br/>
    4. <b>Initialization:</b> Near-zero initialization for adapters to start close to identity function<br/>
    5. <b>Loss Function:</b> CTC (Connectionist Temporal Classification) for sequence-to-sequence<br/>
    """
    story.append(Paragraph(training_text, body_style))
    
    # ==========================================
    # INSTRUCTIONS DE REPRODUCTION
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("Reproduction Instructions", heading_style))
    
    story.append(Paragraph("Step 1: Environment Setup", subheading_style))
    setup_code = """
# Create virtual environment
python -m venv asr_env
source asr_env/bin/activate  # On Windows: asr_env\\Scripts\\activate

# Install dependencies
pip install torch transformers datasets jiwer librosa pydub tqdm pandas
pip install reportlab  # For PDF generation

# Install ffmpeg (required for audio processing)
# Windows: Download from https://www.gyan.dev/ffmpeg/builds/
# Linux: sudo apt-get install ffmpeg
# Mac: brew install ffmpeg
    """
    story.append(Paragraph(setup_code, code_style))
    
    story.append(Paragraph("Step 2: Data Preparation", subheading_style))
    data_code = """
# Download dataset
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="DigitalUmuganda/ASR_Fellowship_Challenge_Dataset",
    repo_type='dataset',
    local_dir='./dataset'
)

# Extract audio files
python extract_train.py  # Extract train set (~30 min)
python baseline_evaluation.py  # Extract and evaluate validation set
    """
    story.append(Paragraph(data_code, code_style))
    
    story.append(Paragraph("Step 3: Baseline Evaluation", subheading_style))
    baseline_code = """
# Run baseline (zero-shot) evaluation
python baseline_evaluation.py

# Expected output:
# - baseline_results/base_transcriptions.txt
# - baseline_results/vocab.json
# - Baseline WER: ~95-100%
    """
    story.append(Paragraph(baseline_code, code_style))
    
    story.append(Paragraph("Step 4: Adapter Fine-Tuning", subheading_style))
    training_code = """
# Train with adapters
python train_adapter.py

# Expected duration:
# - CPU: 15-20 hours (not recommended)
# - GPU (RTX 3080): 2-4 hours

# Outputs:
# - adapter_results/best_adapter_weights.pt
# - adapter_results/training_config.json
    """
    story.append(Paragraph(training_code, code_style))
    
    story.append(Paragraph("Step 5: Final Inference", subheading_style))
    inference_code = """
# Generate predictions with fine-tuned model
python inference_adapter.py

# Outputs:
# - final_results/finetuned_transcriptions.txt (test set)
# - final_results/finetuned_transcriptions_val.txt (validation set)
# - WER improvement displayed in console
    """
    story.append(Paragraph(inference_code, code_style))
    
    story.append(Paragraph("Step 6: Generate Report", subheading_style))
    report_code = """
# Generate PDF report
python generate_report.py

# Output: final_results/rapport.pdf
    """
    story.append(Paragraph(report_code, code_style))
    
    # ==========================================
    # FICHIERS DE SOUMISSION
    # ==========================================
    story.append(Paragraph("Submission Files", subheading_style))
    
    files_text = """
    The following files are included in the submission:<br/><br/>
    
    <b>1. Transcription Files:</b><br/>
    • base_transcriptions.txt - Baseline model predictions on test set<br/>
    • finetuned_transcriptions.txt - Fine-tuned model predictions on test set<br/><br/>
    
    <b>2. Code Files:</b><br/>
    • adapters.py - Adapter module implementation<br/>
    • train_adapter.py - Training script<br/>
    • inference_adapter.py - Inference script<br/>
    • baseline_evaluation.py - Baseline evaluation<br/>
    • extract_train.py - Data extraction helper<br/>
    • generate_report.py - Report generation<br/><br/>
    
    <b>3. Model Files:</b><br/>
    • base_model_config/ - Base model configuration<br/>
    • best_adapter_weights.pt - Trained adapter weights (~2MB)<br/>
    • vocab.json - Vocabulary mapping<br/><br/>
    
    <b>4. Documentation:</b><br/>
    • rapport.pdf - This report<br/>
    • README.md - Setup and usage instructions<br/>
    • training_config.json - Training hyperparameters<br/>
    """
    story.append(Paragraph(files_text, body_style))
    
    # ==========================================
    # RÉFÉRENCES
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("References", heading_style))
    
    references_text = """
    [1] Thomas, B., Kessler, S., & Karout, S. (2022). <i>Efficient Adapter Transfer of 
    Self-Supervised Speech Models for Automatic Speech Recognition.</i> arXiv:2202.03218.<br/><br/>
    
    [2] Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., de Laroussilhe, Q., Gesmundo, A., 
    Attariyan, M., & Gelly, S. (2019). <i>Parameter-Efficient Transfer Learning for NLP.</i> 
    arXiv:1902.00751.<br/><br/>
    
    [3] Hou, W., Dong, Y., Zhuang, B., Yang, L., Shi, Y., & Shinozaki, T. (2021). 
    <i>Exploiting Adapters for Cross-lingual Low-resource Speech Recognition.</i> 
    arXiv:2105.11905.<br/><br/>
    
    [4] Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). <i>wav2vec 2.0: A Framework for 
    Self-Supervised Learning of Speech Representations.</i> arXiv:2006.11477.<br/><br/>
    
    [5] Conneau, A., Baevski, A., Collobert, R., Mohamed, A., & Auli, M. (2021). 
    <i>Unsupervised Cross-Lingual Representation Learning for Speech Recognition.</i> 
    arXiv:2006.13979.
    """
    story.append(Paragraph(references_text, body_style))
    
    # ==========================================
    # CONCLUSION
    # ==========================================
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Conclusion", heading_style))
    
    conclusion_text = f"""
    This work demonstrates the effectiveness of parameter-efficient adapter modules for 
    low-resource ASR. By training only {(results['training'].get('trainable_params', 0) / results['training'].get('total_params', 1) * 100):.2f}% of the model parameters, 
    we achieved a {results['comparison']['relative_improvement']:.1f}% relative improvement in WER over the baseline model.
    <br/><br/>
    The adapter-based approach offers several advantages:<br/>
    • <b>Efficiency:</b> Fast training with minimal computational resources<br/>
    • <b>Preservation:</b> Retains multilingual knowledge from pre-training<br/>
    • <b>Modularity:</b> Adapters can be easily swapped for different domains<br/>
    • <b>Scalability:</b> Applicable to other low-resource languages<br/>
    <br/>
    Future work could explore: (1) training adapters on larger datasets, (2) combining multiple 
    adapter modules for multi-domain adaptation, and (3) applying this approach to other 
    Kinyarwanda speech tasks.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Construction du PDF
    doc.build(story)
    print(f"✅ Rapport PDF généré: {OUTPUT_PDF}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("📄 Génération du rapport PDF...")
    
    # Vérifier que reportlab est installé
    try:
        from reportlab.lib.pagesizes import A4
    except ImportError:
        print("❌ Erreur: reportlab n'est pas installé")
        print("   Installez-le avec: pip install reportlab")
        exit(1)
    
    # Vérifier que les fichiers existent
    if not os.path.exists(ADAPTER_DIR):
        print(f"⚠️  Dossier {ADAPTER_DIR} introuvable")
        print("   Exécutez d'abord train_adapter.py")
        exit(1)
    
    # Générer le rapport
    generate_report()
    print(f"🎉 Rapport généré avec succès!")
    print(f"📂 Emplacement: {OUTPUT_PDF}")