"""
Detector de Tom Musical
========================
Identifica a tonalidade (key) de uma música a partir de um arquivo de áudio
ou URL do YouTube usando análise de chroma features e o algoritmo
Krumhansl-Schmuckler.

Uso:
    python key_detector.py caminho/para/musica.mp3
    python key_detector.py https://youtu.be/l4bX1NPZeB0

Dependências:
    pip install librosa numpy soundfile yt-dlp
"""

import sys
import os
import tempfile
import subprocess
import numpy as np

# Perfis de Krumhansl-Schmuckler para tonalidades maiores e menores.
# Esses valores representam a "importância" de cada nota da escala cromática
# em relação à tônica, baseados em estudos perceptuais.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

NOTE_NAMES_PT = ['Dó', 'Dó#', 'Ré', 'Ré#', 'Mi', 'Fá',
                 'Fá#', 'Sol', 'Sol#', 'Lá', 'Lá#', 'Si']


def download_from_youtube(url: str) -> str:
    """Baixa o áudio de uma URL do YouTube usando yt-dlp."""
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "audio.wav")

    cmd = [
        "yt-dlp",
        "-x",                        # extrair apenas áudio
        "--audio-format", "wav",     # converter para wav
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    print(f"Baixando áudio de: {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao baixar áudio:\n{result.stderr}\n"
            "Certifique-se de que o yt-dlp está instalado: pip install yt-dlp"
        )

    # yt-dlp pode adicionar extensão extra ao nome do arquivo
    for f in os.listdir(tmp_dir):
        if f.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
            return os.path.join(tmp_dir, f)

    raise RuntimeError("Arquivo de áudio não encontrado após download.")


def extract_chroma(audio_path: str) -> np.ndarray:
    """Extrai o perfil cromático médio do áudio usando librosa."""
    import librosa

    print(f"Analisando: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Chroma features: energia de cada uma das 12 notas cromáticas ao longo do tempo
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Média temporal: perfil cromático geral da música
    chroma_mean = np.mean(chroma, axis=1)

    # Normaliza para comparação
    chroma_mean = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-8)

    return chroma_mean


def detect_key(chroma_mean: np.ndarray) -> dict:
    """
    Aplica o algoritmo Krumhansl-Schmuckler para determinar a tonalidade.

    Para cada uma das 24 tonalidades possíveis (12 maiores + 12 menores),
    calcula a correlação entre o perfil cromático da música e o perfil
    teórico da tonalidade. A tonalidade com maior correlação é escolhida.
    """
    results = []

    for shift in range(12):
        # Rotaciona o perfil de referência para cada tônica possível
        major_rotated = np.roll(MAJOR_PROFILE, shift)
        minor_rotated = np.roll(MINOR_PROFILE, shift)

        # Normaliza os perfis
        major_norm = major_rotated / (np.linalg.norm(major_rotated) + 1e-8)
        minor_norm = minor_rotated / (np.linalg.norm(minor_rotated) + 1e-8)

        # Correlação (produto escalar entre vetores normalizados)
        corr_major = np.dot(chroma_mean, major_norm)
        corr_minor = np.dot(chroma_mean, minor_norm)

        results.append({
            "note": NOTE_NAMES[shift],
            "note_pt": NOTE_NAMES_PT[shift],
            "mode": "Maior",
            "mode_en": "Major",
            "correlation": float(corr_major),
        })
        results.append({
            "note": NOTE_NAMES[shift],
            "note_pt": NOTE_NAMES_PT[shift],
            "mode": "Menor",
            "mode_en": "Minor",
            "correlation": float(corr_minor),
        })

    # Ordena por correlação (maior = melhor match)
    results.sort(key=lambda x: x["correlation"], reverse=True)

    return {
        "best": results[0],
        "ranking": results,
    }


def format_result(result: dict) -> str:
    """Formata o resultado da detecção de forma legível."""
    best = result["best"]
    lines = []
    lines.append("=" * 50)
    lines.append("  RESULTADO DA ANÁLISE DE TONALIDADE")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"  Tom detectado: {best['note_pt']} {best['mode']}"
                 f"  ({best['note']} {best['mode_en']})")
    lines.append(f"  Confiança:     {best['correlation']:.1%}")
    lines.append("")

    # Acordes típicos da tonalidade
    lines.append("  Acordes comuns nessa tonalidade:")
    chords = get_common_chords(best["note"], best["mode_en"])
    lines.append(f"  {chords}")
    lines.append("")

    # Notas para o baixo
    lines.append("  Notas raiz para o baixo:")
    bass_notes = get_bass_notes(best["note"], best["mode_en"])
    lines.append(f"  {bass_notes}")
    lines.append("")

    # Top 5 candidatos
    lines.append("-" * 50)
    lines.append("  Top 5 tonalidades mais prováveis:")
    lines.append("-" * 50)
    for i, r in enumerate(result["ranking"][:5], 1):
        lines.append(
            f"  {i}. {r['note_pt']} {r['mode']:5s} "
            f"({r['note']} {r['mode_en']:5s}) "
            f"- correlação: {r['correlation']:.1%}"
        )

    lines.append("=" * 50)
    return "\n".join(lines)


def get_common_chords(root: str, mode: str) -> str:
    """Retorna os acordes diatônicos comuns da tonalidade."""
    root_idx = NOTE_NAMES.index(root)

    if mode == "Major":
        # I  ii  iii  IV  V  vi  (graus da escala maior)
        intervals = [0, 2, 4, 5, 7, 9]
        suffixes = ["", "m", "m", "", "", "m"]
    else:
        # i  ii°  III  iv  v  VI  VII  (graus da escala menor)
        intervals = [0, 2, 3, 5, 7, 8, 10]
        suffixes = ["m", "dim", "", "m", "m", "", ""]

    chords = []
    for interval, suffix in zip(intervals, suffixes):
        note = NOTE_NAMES[(root_idx + interval) % 12]
        chords.append(f"{note}{suffix}")

    return " - ".join(chords)


def get_bass_notes(root: str, mode: str) -> str:
    """Retorna as notas raiz para o baixista."""
    root_idx = NOTE_NAMES.index(root)

    if mode == "Major":
        intervals = [0, 2, 4, 5, 7, 9]
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]

    notes = []
    for interval in intervals:
        idx = (root_idx + interval) % 12
        notes.append(f"{NOTE_NAMES_PT[idx]} ({NOTE_NAMES[idx]})")

    return " - ".join(notes)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Erro: forneça o caminho do arquivo de áudio ou URL do YouTube.")
        print("Exemplo: python key_detector.py musica.mp3")
        print("         python key_detector.py https://youtu.be/VIDEO_ID")
        sys.exit(1)

    source = sys.argv[1]
    cleanup_path = None

    try:
        # Verifica se é URL do YouTube
        if source.startswith(("http://", "https://", "www.")):
            audio_path = download_from_youtube(source)
            cleanup_path = os.path.dirname(audio_path)
        else:
            if not os.path.exists(source):
                print(f"Erro: arquivo não encontrado: {source}")
                sys.exit(1)
            audio_path = source

        # Extrai perfil cromático
        chroma = extract_chroma(audio_path)

        # Detecta a tonalidade
        result = detect_key(chroma)

        # Exibe resultado
        print(format_result(result))

    finally:
        # Limpa arquivos temporários do download
        if cleanup_path and os.path.isdir(cleanup_path):
            import shutil
            shutil.rmtree(cleanup_path, ignore_errors=True)


if __name__ == "__main__":
    main()
