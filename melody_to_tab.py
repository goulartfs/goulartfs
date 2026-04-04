"""
Transcritor de Melodia para Tablatura de Baixo
================================================
Analisa um arquivo de áudio, detecta as notas da melodia
e gera tablatura para baixo de 4 cordas automaticamente.

Uso:
    python melody_to_tab.py musica.mp3
    python melody_to_tab.py musica.mp3 --output tab.txt
    python melody_to_tab.py https://youtu.be/VIDEO_ID

Opções:
    --output, -o    Arquivo de saída (padrão: stdout)
    --bass-range    Filtrar apenas frequências de baixo (padrão: True)
    --min-freq      Frequência mínima em Hz (padrão: 30)
    --max-freq      Frequência máxima em Hz (padrão: 400)
    --hop           Resolução temporal em ms (padrão: 50)
    --threshold     Limiar de confiança 0-1 (padrão: 0.3)
    --measures-per-line  Compassos por linha da tab (padrão: 4)

Dependências:
    pip install librosa numpy soundfile yt-dlp
"""

import sys
import os
import argparse
import tempfile
import subprocess
import numpy as np


# Afinação padrão do baixo 4 cordas (do grave pro agudo)
BASS_STRINGS = {
    "E": 40,  # E1 - MIDI 40
    "A": 45,  # A1 - MIDI 45
    "D": 50,  # D2 - MIDI 50
    "G": 55,  # G2 - MIDI 55
}

# Número máximo de trastes considerados
MAX_FRET = 12

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


def download_from_youtube(url: str) -> str:
    """Baixa o áudio de uma URL do YouTube usando yt-dlp."""
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "audio.wav")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    print(f"Baixando áudio de: {url}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Erro ao baixar áudio:\n{result.stderr}\n"
            "Instale o yt-dlp: pip install yt-dlp"
        )

    for f in os.listdir(tmp_dir):
        if f.endswith((".wav", ".webm", ".m4a", ".mp3", ".ogg")):
            return os.path.join(tmp_dir, f)

    raise RuntimeError("Arquivo de áudio não encontrado após download.")


def freq_to_midi(freq):
    """Converte frequência (Hz) para número MIDI."""
    if freq <= 0:
        return 0
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_note_name(midi_num):
    """Converte número MIDI para nome da nota."""
    note = NOTE_NAMES[int(round(midi_num)) % 12]
    octave = int(round(midi_num)) // 12 - 1
    return f"{note}{octave}"


def midi_to_fret_position(midi_num):
    """
    Converte número MIDI para posição no braço do baixo.
    Retorna (corda, traste) escolhendo a posição mais confortável.
    """
    midi_rounded = int(round(midi_num))
    best = None
    best_cost = float("inf")

    for string_name, open_midi in BASS_STRINGS.items():
        fret = midi_rounded - open_midi
        if 0 <= fret <= MAX_FRET:
            # Prefere cordas mais graves e trastes mais baixos
            cost = fret + (0 if string_name in ("E", "A") else 1)
            if cost < best_cost:
                best_cost = cost
                best = (string_name, fret)

    return best


def detect_melody(audio_path, min_freq=30, max_freq=400, hop_ms=50,
                  threshold=0.3):
    """
    Detecta a melodia/linha de baixo do áudio usando pYIN.

    Retorna lista de dicts com:
        - time: tempo em segundos
        - freq: frequência em Hz (0 se silêncio)
        - midi: número MIDI
        - note: nome da nota
        - confidence: confiança da detecção
    """
    import librosa

    print("Carregando áudio...", file=sys.stderr)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    hop_length = int(sr * hop_ms / 1000)

    print("Detectando notas com pYIN...", file=sys.stderr)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=min_freq,
        fmax=max_freq,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    notes = []
    for t, freq, voiced, prob in zip(times, f0, voiced_flag, voiced_prob):
        if voiced and freq > 0 and prob >= threshold:
            midi = freq_to_midi(freq)
            note_name = midi_to_note_name(midi)
            notes.append({
                "time": float(t),
                "freq": float(freq),
                "midi": float(midi),
                "note": note_name,
                "confidence": float(prob),
            })
        else:
            notes.append({
                "time": float(t),
                "freq": 0,
                "midi": 0,
                "note": "-",
                "confidence": 0,
            })

    print(f"Análise concluída: {len(notes)} frames, "
          f"{sum(1 for n in notes if n['freq'] > 0)} notas detectadas.",
          file=sys.stderr)

    return notes, hop_ms / 1000.0


def quantize_to_beats(notes, frame_duration, bpm=None):
    """
    Quantiza as notas detectadas em subdivisões de tempo.
    Se bpm não for fornecido, tenta detectar automaticamente.

    Retorna lista de notas quantizadas por beat (semicolcheia = 16th note).
    """
    import librosa

    if bpm is None:
        # Estima BPM a partir das mudanças de nota
        note_onsets = []
        prev_midi = 0
        for n in notes:
            if n["midi"] > 0 and abs(n["midi"] - prev_midi) > 0.5:
                note_onsets.append(n["time"])
            prev_midi = n["midi"] if n["midi"] > 0 else prev_midi

        if len(note_onsets) > 2:
            intervals = np.diff(note_onsets)
            intervals = intervals[(intervals > 0.15) & (intervals < 2.0)]
            if len(intervals) > 0:
                median_interval = np.median(intervals)
                bpm = 60.0 / median_interval
                # Ajusta para faixa razoável
                while bpm < 70:
                    bpm *= 2
                while bpm > 180:
                    bpm /= 2

        if bpm is None:
            bpm = 120.0

    print(f"BPM estimado: {bpm:.0f}", file=sys.stderr)

    beat_duration = 60.0 / bpm
    sixteenth = beat_duration / 4  # subdivisão em semicolcheia

    # Agrupa notas por posição de semicolcheia
    total_time = notes[-1]["time"] if notes else 0
    total_sixteenths = int(total_time / sixteenth) + 1

    quantized = []
    for i in range(total_sixteenths):
        t_start = i * sixteenth
        t_end = (i + 1) * sixteenth

        # Pega notas nesse intervalo
        frame_notes = [n for n in notes
                       if t_start <= n["time"] < t_end and n["freq"] > 0]

        if frame_notes:
            # Usa a nota com maior confiança
            best = max(frame_notes, key=lambda x: x["confidence"])
            quantized.append({
                "position": i,
                "beat": i // 4,
                "sub": i % 4,
                "midi": best["midi"],
                "note": best["note"],
                "freq": best["freq"],
            })
        else:
            quantized.append({
                "position": i,
                "beat": i // 4,
                "sub": i % 4,
                "midi": 0,
                "note": "-",
                "freq": 0,
            })

    return quantized, bpm


def simplify_notes(quantized):
    """
    Remove notas repetidas consecutivas (sustain) e mantém apenas
    a primeira ocorrência + silêncios.
    """
    simplified = []
    prev_midi = None

    for q in quantized:
        if q["midi"] > 0:
            rounded = int(round(q["midi"]))
            if rounded != prev_midi:
                simplified.append(q)
                prev_midi = rounded
            else:
                # Nota sustentada - marca como sustain
                simplified.append({
                    **q,
                    "sustain": True,
                })
        else:
            simplified.append(q)
            prev_midi = None

    return simplified


def generate_tablature(quantized, bpm, measures_per_line=4):
    """
    Gera tablatura de baixo em formato texto a partir das notas quantizadas.
    """
    beats_per_measure = 4
    sixteenths_per_measure = 16
    sixteenths_per_line = sixteenths_per_measure * measures_per_line

    total_measures = len(quantized) // sixteenths_per_measure + 1

    lines_output = []
    lines_output.append("=" * 70)
    lines_output.append("  TABLATURA DE BAIXO - Gerada automaticamente")
    lines_output.append(f"  BPM: {bpm:.0f}")
    lines_output.append(f"  Afinação: E A D G (padrão)")
    lines_output.append("  h = hammer-on | p = pull-off | x = ghost note")
    lines_output.append("=" * 70)
    lines_output.append("")

    # Processa linha por linha
    for line_start in range(0, len(quantized), sixteenths_per_line):
        line_end = min(line_start + sixteenths_per_line, len(quantized))
        chunk = quantized[line_start:line_end]

        if not chunk:
            break

        # Verifica se a linha tem alguma nota
        has_notes = any(q["midi"] > 0 for q in chunk)
        if not has_notes:
            continue

        # Calcula número do compasso
        measure_start = line_start // sixteenths_per_measure + 1
        measure_end = (line_end - 1) // sixteenths_per_measure + 1

        # Monta as 4 cordas
        string_lines = {"G": "", "D": "", "A": "", "E": ""}
        beat_line = " "

        for i, q in enumerate(chunk):
            pos_in_measure = (line_start + i) % sixteenths_per_measure

            # Separador de compasso
            if pos_in_measure == 0 and i > 0:
                for s in string_lines:
                    string_lines[s] += "|"
                beat_line += "|"

            if q["midi"] > 0:
                is_sustain = q.get("sustain", False)
                if is_sustain:
                    for s in string_lines:
                        string_lines[s] += "--"
                    beat_line += "  "
                else:
                    fret_pos = midi_to_fret_position(q["midi"])
                    if fret_pos:
                        string_name, fret = fret_pos
                        for s in string_lines:
                            if s == string_name:
                                fret_str = str(fret)
                                string_lines[s] += fret_str.ljust(2)
                            else:
                                string_lines[s] += "--"
                    else:
                        # Nota fora do range do baixo
                        for s in string_lines:
                            string_lines[s] += "--"

                    # Marca de beat
                    beat_num = pos_in_measure // 4 + 1
                    if pos_in_measure % 4 == 0:
                        beat_line += f"{beat_num} "
                    elif pos_in_measure % 4 == 2:
                        beat_line += "& "
                    else:
                        beat_line += "  "
            else:
                for s in string_lines:
                    string_lines[s] += "--"

                beat_num = pos_in_measure // 4 + 1
                if pos_in_measure % 4 == 0:
                    beat_line += f"{beat_num} "
                elif pos_in_measure % 4 == 2:
                    beat_line += "& "
                else:
                    beat_line += "  "

        # Adiciona barra final
        for s in string_lines:
            string_lines[s] += "|"

        # Escreve a linha
        lines_output.append(f"  Compassos {measure_start}-{measure_end}")
        lines_output.append(f"G|{string_lines['G']}")
        lines_output.append(f"D|{string_lines['D']}")
        lines_output.append(f"A|{string_lines['A']}")
        lines_output.append(f"E|{string_lines['E']}")
        lines_output.append(f"  {beat_line}")
        lines_output.append("")

    # Resumo de notas encontradas
    lines_output.append("=" * 70)
    lines_output.append("  NOTAS DETECTADAS (resumo)")
    lines_output.append("=" * 70)

    note_counts = {}
    for q in quantized:
        if q["midi"] > 0 and not q.get("sustain", False):
            note = midi_to_note_name(q["midi"])
            # Pega só o nome sem oitava
            base = note[:-1]
            note_counts[base] = note_counts.get(base, 0) + 1

    if note_counts:
        sorted_notes = sorted(note_counts.items(), key=lambda x: -x[1])
        for note, count in sorted_notes:
            bar = "█" * min(count, 40)
            lines_output.append(f"  {note:3s} | {bar} ({count})")
    else:
        lines_output.append("  Nenhuma nota detectada.")

    lines_output.append("")

    # Tonalidade provável
    lines_output.append("=" * 70)
    lines_output.append("  MAPA DE POSIÇÕES USADAS NO BRAÇO")
    lines_output.append("=" * 70)
    lines_output.append("")

    fret_map = {s: set() for s in BASS_STRINGS}
    for q in quantized:
        if q["midi"] > 0 and not q.get("sustain", False):
            pos = midi_to_fret_position(q["midi"])
            if pos:
                fret_map[pos[0]].add(pos[1])

    for string in ["G", "D", "A", "E"]:
        frets = fret_map[string]
        line = f"  {string} |"
        for f in range(MAX_FRET + 1):
            if f in frets:
                line += f"-[{f}]"
            else:
                line += "----"
        line += "|"
        lines_output.append(line)

    lines_output.append(f"       {''.join(str(i).ljust(4) for i in range(MAX_FRET + 1))}")
    lines_output.append("")
    lines_output.append("=" * 70)

    return "\n".join(lines_output)


def main():
    parser = argparse.ArgumentParser(
        description="Transcreve melodia de áudio para tablatura de baixo"
    )
    parser.add_argument("source", help="Arquivo de áudio ou URL do YouTube")
    parser.add_argument("-o", "--output", help="Arquivo de saída", default=None)
    parser.add_argument("--min-freq", type=float, default=30,
                        help="Frequência mínima Hz (padrão: 30)")
    parser.add_argument("--max-freq", type=float, default=400,
                        help="Frequência máxima Hz (padrão: 400)")
    parser.add_argument("--hop", type=float, default=50,
                        help="Resolução temporal em ms (padrão: 50)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Limiar de confiança 0-1 (padrão: 0.3)")
    parser.add_argument("--measures-per-line", type=int, default=4,
                        help="Compassos por linha (padrão: 4)")
    parser.add_argument("--bpm", type=float, default=None,
                        help="BPM manual (padrão: auto-detectar)")

    args = parser.parse_args()
    cleanup_path = None

    try:
        if args.source.startswith(("http://", "https://", "www.")):
            audio_path = download_from_youtube(args.source)
            cleanup_path = os.path.dirname(audio_path)
        else:
            if not os.path.exists(args.source):
                print(f"Erro: arquivo não encontrado: {args.source}",
                      file=sys.stderr)
                sys.exit(1)
            audio_path = args.source

        # 1. Detecta notas
        notes, frame_dur = detect_melody(
            audio_path,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            hop_ms=args.hop,
            threshold=args.threshold,
        )

        # 2. Quantiza em beats
        quantized, bpm = quantize_to_beats(notes, frame_dur, bpm=args.bpm)

        # 3. Simplifica (remove sustain repetido)
        simplified = simplify_notes(quantized)

        # 4. Gera tablatura
        tab = generate_tablature(simplified, bpm,
                                 measures_per_line=args.measures_per_line)

        # 5. Output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(tab)
            print(f"Tablatura salva em: {args.output}", file=sys.stderr)
        else:
            print(tab)

    finally:
        if cleanup_path and os.path.isdir(cleanup_path):
            import shutil
            shutil.rmtree(cleanup_path, ignore_errors=True)


if __name__ == "__main__":
    main()
