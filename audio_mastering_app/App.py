import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import tempfile
import os

st.set_page_config(page_title="Audio Mastering AI", page_icon="🎵", layout="wide")

st.title("🎵 Audio Mastering avec IA")
st.markdown("Analysez et masterisez vos pistes audio automatiquement")

# Fonction pour détecter le BPM - CORRIGÉE (cache basé sur le contenu du fichier)
@st.cache_data
def detect_bpm(audio_bytes):
    try:
        # Créer un fichier temporaire à partir des bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Nettoyage
        os.unlink(tmp_path)
        
        return float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    except Exception as e:
        st.error(f"Erreur détection BPM: {e}")
        return 0.0

# Fonction pour détecter la tonalité (cache basé sur le contenu du fichier)
@st.cache_data
def detect_key(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_index = np.argmax(np.sum(chroma, axis=1))
        
        os.unlink(tmp_path)
        
        return key_names[key_index]
    except Exception as e:
        st.error(f"Erreur détection tonalité: {e}")
        return "N/A"

# Fonction pour détecter le genre  (cache basé sur le contenu du fichier)
@st.cache_data
def detect_genre(audio_bytes):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=30)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Correction: extraire la valeur scalaire du tempo
        tempo_value = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
        avg_centroid = np.mean(spectral_centroids)
        
        os.unlink(tmp_path)
        
        if tempo_value > 140 and avg_centroid > 3000:
            return "Electronic/Dance"
        elif tempo_value > 120 and avg_centroid > 2500:
            return "Pop/Rock"
        elif tempo_value < 90 and avg_centroid < 2000:
            return "Jazz/Blues"
        elif tempo_value < 100:
            return "Hip-Hop/R&B"
        else:
            return "Pop/Alternative"
    except Exception as e:
        st.error(f"Erreur détection genre: {e}")
        return "Unknown"

# Fonction de mastering 
def ai_mastering(y, sr, intensity=0.5):
    """
    Mastering automatique avec traitements audio professionnels
    """
    try:
        y_master = y.copy().astype(np.float64)
        
        # 1. Égalisation douce
        # Boost léger des basses (60-150 Hz)
        sos_low = signal.butter(2, [60, 150], btype='band', fs=sr, output='sos')
        low_boost = signal.sosfilt(sos_low, y_master) * (0.15 * intensity)
        
        # Boost léger des aigus (8000-12000 Hz)
        max_freq = min(12000, sr // 2 - 100)
        sos_high = signal.butter(2, [8000, max_freq], btype='band', fs=sr, output='sos')
        high_boost = signal.sosfilt(sos_high, y_master) * (0.1 * intensity)
        
        y_master = y_master + low_boost + high_boost
        
        # 2. Compression douce (soft knee)
        threshold = 0.5
        ratio = 3.0
        knee_width = 0.1
        
        abs_signal = np.abs(y_master)
        mask = abs_signal > (threshold - knee_width)
        
        for i in np.where(mask)[0]:
            if abs_signal[i] > threshold:
                excess = abs_signal[i] - threshold
                compressed = threshold + excess / ratio
                y_master[i] = np.sign(y_master[i]) * compressed
        
        # 3. Limitation douce
        y_master = np.tanh(y_master * 0.9) * 0.95
        
        # 4. Normalisation à -0.5 dB (évite la saturation)
        peak = np.max(np.abs(y_master))
        if peak > 0:
            target_level = 0.85  # -0.5 dB environ
            y_master = y_master * (target_level / peak)
        
        return y_master.astype(np.float32)
    
    except Exception as e:
        st.error(f"Erreur pendant le mastering: {e}")
        return y

# Interface principale
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier audio", 
        type=['wav', 'mp3', 'flac', 'ogg'],
        help="Formats supportés: WAV, MP3, FLAC, OGG (max 200MB)"
    )
    
    if uploaded_file is not None:
        # Vérification de la taille
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > 200:
            st.error("❌ Fichier trop volumineux (max 200MB)")
        else:
            # Sauvegarde temporaire
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.success(f"✅ Fichier chargé ({file_size_mb:.1f} MB)")
                st.audio(uploaded_file, format='audio/wav')
                
                # Analyse (limitée à 30 secondes pour performance)
                st.header("🔍 Analyse")
                with st.spinner("Analyse en cours..."):
                    # Passer les bytes au lieu du chemin pour éviter les problèmes de cache
                    audio_bytes = uploaded_file.getvalue()
                    bpm = detect_bpm(audio_bytes)
                    key = detect_key(audio_bytes)
                    genre = detect_genre(audio_bytes)
                    
                    # Durée du fichier complet
                    y_full, sr_full = librosa.load(tmp_path, sr=None, mono=True)
                    duration = len(y_full) / sr_full
                
                # Affichage des résultats
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("BPM", f"{bpm:.1f}")
                with metric_col2:
                    st.metric("Tonalité", key)
                with metric_col3:
                    st.metric("Genre", genre)
                with metric_col4:
                    st.metric("Durée", f"{duration:.1f}s")
            
            except Exception as e:
                st.error(f"❌ Erreur de chargement: {e}")
                uploaded_file = None

with col2:
    if uploaded_file is not None and 'tmp_path' in locals():
        st.header("🎚️ Mastering AI")
        
        st.markdown("### Paramètres")
        intensity = st.slider(
            "Intensité du mastering", 
            0.0, 1.0, 0.3, 0.1,
            help="0 = léger, 1 = intense"
        )
        
        st.info(f"💡 Intensité recommandée: 0.3-0.5 pour préserver la qualité audio")
        
        if st.button("🚀 Appliquer le Mastering", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Chargement
                status_text.text("Chargement de l'audio...")
                progress_bar.progress(20)
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
                
                # Mastering
                status_text.text("Application du mastering...")
                progress_bar.progress(50)
                y_mastered = ai_mastering(y, sr, intensity)
                
                # Sauvegarde
                status_text.text("Sauvegarde du fichier...")
                progress_bar.progress(80)
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                sf.write(output_path, y_mastered, sr, subtype='PCM_16')
                
                progress_bar.progress(100)
                status_text.text("✅ Terminé!")
                
                st.success("✅ Mastering terminé!")
                
                # Lecture du fichier masterisé
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()
                
                st.audio(audio_bytes, format='audio/wav')
                
                # Téléchargement
                st.download_button(
                    label="📥 Télécharger le fichier masterisé",
                    data=audio_bytes,
                    file_name=f"mastered_{uploaded_file.name.split('.')[0]}.wav",
                    mime="audio/wav"
                )
                
                # Comparaison des formes d'onde
                st.markdown("### Comparaison des formes d'onde")
                
                # Affichage des 5 premières secondes
                sample_length = min(sr * 5, len(y))
                
                fig_col1, fig_col2 = st.columns(2)
                
                with fig_col1:
                    st.markdown("**Original**")
                    st.line_chart(y[:sample_length])
                
                with fig_col2:
                    st.markdown("**Masterisé**")
                    st.line_chart(y_mastered[:sample_length])
                
                # Statistiques
                st.markdown("### Statistiques")
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    peak_original = np.max(np.abs(y))
                    rms_original = np.sqrt(np.mean(y**2))
                    st.metric("Peak Original", f"{20*np.log10(peak_original):.1f} dB")
                    st.metric("RMS Original", f"{20*np.log10(rms_original):.1f} dB")
                
                with stat_col2:
                    peak_mastered = np.max(np.abs(y_mastered))
                    rms_mastered = np.sqrt(np.mean(y_mastered**2))
                    st.metric("Peak Masterisé", f"{20*np.log10(peak_mastered):.1f} dB")
                    st.metric("RMS Masterisé", f"{20*np.log10(rms_mastered):.1f} dB")
                
                # Nettoyage
                os.unlink(output_path)
                
            except Exception as e:
                st.error(f"❌ Erreur pendant le mastering: {e}")
                progress_bar.empty()
                status_text.empty()

# Sidebar avec infos
with st.sidebar:
    st.header("ℹ️ À propos")
    st.markdown("""
    Cette application permet de:
    - **Analyser** vos pistes audio (BPM, tonalité, genre)
    - **Masteriser** automatiquement avec l'IA
    - **Télécharger** le résultat en haute qualité
    
    ### Traitements appliqués:
    - ✓ Égalisation intelligente
    - ✓ Compression dynamique douce
    - ✓ Limitation et normalisation
    - ✓ Optimisation du loudness
    
    ### Formats supportés:
    WAV, MP3, FLAC, OGG (max 200MB)
    
    ### Conseils:
    - Utilisez une intensité de 0.3-0.5
    - Testez plusieurs réglages
    - Écoutez sur différents systèmes
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")
    
    # Nettoyage en fin de session
    if st.button("🗑️ Nettoyer les fichiers temporaires"):
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
                st.success("Fichiers nettoyés!")
            except:
                pass