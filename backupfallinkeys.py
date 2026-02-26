import pygame
import random
import sys
import tkinter as tk
from tkinter import filedialog
import os
from mido import MidiFile
import numpy as np
import math

try:
    import pygame.midi as pgmidi
except Exception:
    pgmidi = None

# Initialiseer Pygame
pygame.init()

# Audio
SOUND_SAMPLE_RATE = 44100
# Grotere buffer om clicks te voorkomen, 2048 geeft betere audio kwaliteit
pygame.mixer.init(frequency=SOUND_SAMPLE_RATE, size=-16, channels=2, buffer=2048)
SUSTAIN_SOUND_SECONDS = 8.0

# Schermafmetingen
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Judgement UI animatie (snel & clean)
JUDGEMENT_IN_FRAMES = int(FPS * 0.12)  # ~0.12s fade/slide-in
JUDGEMENT_SLIDE_PX = 14
JUDGEMENT_OUT_FRAMES = int(FPS * 0.22)  # fade/slide-out wanneer er een nieuwe judgement komt
JUDGEMENT_OUT_SLIDE_PX = 18
JUDGEMENT_HISTORY_MAX = 3

# Pixel smoke (boven piano toetsen)
SMOKE_SPAWN_PER_FRAME = 2          # per ingedrukte toets per frame
SMOKE_SPAWN_BURST = 6              # extra burst bij keydown
SMOKE_LIFE_FRAMES_MIN = int(FPS * 0.25)
SMOKE_LIFE_FRAMES_MAX = int(FPS * 0.70)
SMOKE_RISE_MIN = 0.55
SMOKE_RISE_MAX = 1.35
SMOKE_DRIFT = 0.35
SMOKE_SIZE_MIN = 2
SMOKE_SIZE_MAX = 5
SMOKE_ALPHA_MIN = 120
SMOKE_ALPHA_MAX = 215
# Smoke particle kleuren: meer wit, minder blauw.
SMOKE_COLORS = [
    (255, 255, 255),
    (242, 248, 255),
    (226, 240, 255),
    (235, 244, 255),
]

# Kleuren
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GREEN = (0, 200, 0)
DARK_GRAY = (50, 50, 50)
TILE_COLOR = (255, 255, 0)  # Geel
TILE_HIT_COLOR = DARK_GREEN  # Groen wanneer geraakt
TILE_MISS_COLOR = (255, 0, 0)  # Rood wanneer gemist
BAR_COLOR = (255, 255, 255)
BAR_DARK_COLOR = (10, 10, 10)  # bijna zwart, verschillend van achtergrond en lijnkleur
MENU_BUTTON_COLOR = (100, 100, 100)
MENU_BUTTON_HOVER = (150, 150, 150)

# Kolom instellingen
NUM_COLUMNS = 12

# Kolom breedtes
BLACK_FACTOR = 0.6
BLACK_INDICES = (1, 3, 6, 8, 10)
RAW_WIDTHS = [BLACK_FACTOR if i in BLACK_INDICES else 1.0 for i in range(NUM_COLUMNS)]
total_raw = sum(RAW_WIDTHS)
column_widths = [int(w / total_raw * SCREEN_WIDTH) for w in RAW_WIDTHS]
column_widths[-1] += SCREEN_WIDTH - sum(column_widths)

# X-positions voor elke kolom
column_x = []
cx = 0
for w in column_widths:
    column_x.append(cx)
    cx += w

BAR_HEIGHT = 80
BAR_Y = SCREEN_HEIGHT - BAR_HEIGHT
TILE_SPEED = 3
SPAWN_RATE = 50
# Hoe lang (in frames) je de fout/miss feedback ziet voordat Game Over start.
# Korter = sneller door naar de Game Over overlay.
MISS_DELAY = int(FPS * 0.55)
# Hoeveel pixels vóór/na de bar een tile al "raakbaar" is (keydown hit detectie).
# Groter = je kunt iets eerder raken terwijl de noot nog net boven de bar is.
HIT_MARGIN_PX = 24
# Level-complete: eerst tekst/paneel, daarna pas de vallende noten.
LEVEL_COMPLETE_FALL_DELAY_FRAMES = int(FPS * 0.65)
# Grotere marge = iets eerder loslaten is nog OK (soepeler hold timing)
HOLD_RELEASE_MARGIN_PX = 45
# Extra strakke marge voor "Perfect" release (kleiner venster)
PERFECT_RELEASE_MARGIN_PX = 20


class SmokeParticle:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-SMOKE_DRIFT, SMOKE_DRIFT)
        self.vy = -random.uniform(SMOKE_RISE_MIN, SMOKE_RISE_MAX)
        self.age = 0
        self.life = random.randint(int(SMOKE_LIFE_FRAMES_MIN), int(SMOKE_LIFE_FRAMES_MAX))
        self.alpha0 = random.randint(int(SMOKE_ALPHA_MIN), int(SMOKE_ALPHA_MAX))
        self.size0 = random.randint(int(SMOKE_SIZE_MIN), int(SMOKE_SIZE_MAX))
        self.grow = random.uniform(0.02, 0.08)
        self.dead = False

        # Kies een iets wittere kleur per particle (met kleine jitter)
        base = random.choice(SMOKE_COLORS)
        j = random.randint(-10, 10)
        self.base_color = (_clamp_u8(base[0] + j), _clamp_u8(base[1] + j), _clamp_u8(base[2] + j))

    def update(self):
        if self.dead:
            return
        self.age += 1
        self.x += self.vx
        self.y += self.vy
        # Heel subtiele opwaartse versnelling + drift dempen
        self.vy -= 0.02
        self.vx *= 0.985
        if self.age >= self.life:
            self.dead = True

    def draw(self, screen: pygame.Surface):
        if self.dead:
            return
        t = min(1.0, float(self.age) / float(max(1, self.life)))
        # Smooth fade-out
        a = int(float(self.alpha0) * (1.0 - t) * (1.0 - 0.15 * t))
        if a <= 0:
            return

        # Pixelated: teken blokjes (geen circles)
        size = int(max(1.0, float(self.size0) + float(self.age) * float(self.grow)))
        x = int(self.x)
        y = int(self.y)

        # Kleur: wit/pastel met klein randje variatie
        r, g, b = getattr(self, 'base_color', random.choice(SMOKE_COLORS))
        c1 = (r, g, b, a)
        c2 = (min(255, r + 30), min(255, g + 18), min(255, b + 10), int(a * 0.65))
        c3 = (max(0, r - 18), max(0, g - 18), max(0, b - 10), int(a * 0.55))

        # Kern
        pygame.draw.rect(screen, c1, pygame.Rect(x, y, size, size))
        # 2-3 losse pixels eromheen voor "rook" look
        if size >= 2:
            pygame.draw.rect(screen, c2, pygame.Rect(x + size, y, max(1, size - 2), max(1, size - 2)))
            pygame.draw.rect(screen, c3, pygame.Rect(x - max(1, size // 2), y + max(1, size // 2), max(1, size - 3), max(1, size - 3)))
            if random.random() < 0.35:
                pygame.draw.rect(screen, c2, pygame.Rect(x + max(1, size // 2), y - max(1, size // 2), 1, 1))

def _clamp_u8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else int(x))

def _offset_color(rgb, dr=0, dg=0, db=0):
    r, g, b = rgb
    return (_clamp_u8(r + dr), _clamp_u8(g + dg), _clamp_u8(b + db))

def apply_retro_key_texture(surface: pygame.Surface, *, base_color, is_black: bool, seed: int = 0) -> None:
    """Geef een clean, blokachtig retro-textuurtje aan een toets (surface).

    We houden het bewust simpel (rects/lines) voor performance en een Minecraft/Terraria vibe.
    """
    w, h = surface.get_size()
    rng = random.Random(int(seed) & 0xFFFFFFFF)

    # Basis: iets minder "plat" dan volledig wit/zwart.
    if is_black:
        base = base_color
        hi = _offset_color(base, 35, 35, 35)
        lo = _offset_color(base, -25, -25, -25)
        edge = _offset_color(base, -40, -40, -40)
    else:
        base = base_color
        hi = _offset_color(base, 10, 10, 10)
        lo = _offset_color(base, -18, -18, -18)
        edge = _offset_color(base, -35, -35, -35)

    surface.fill(base)

    # Blocky bevel (2-3px) voor "3D" gevoel.
    bevel = 3 if h >= 60 else 2
    pygame.draw.rect(surface, hi, pygame.Rect(0, 0, w, bevel))           # top highlight
    pygame.draw.rect(surface, hi, pygame.Rect(0, 0, bevel, h))           # left highlight
    pygame.draw.rect(surface, lo, pygame.Rect(0, h - bevel, w, bevel))   # bottom shadow
    pygame.draw.rect(surface, lo, pygame.Rect(w - bevel, 0, bevel, h))   # right shadow

    # Subtiele glossy highlight (blocky bands), ook op witte toetsen.
    # Dit geeft een "glans" zoals retro/Minecraft-ish UI, zonder drukke details.
    if w > 12 and h > 18:
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        # Zwarte toetsen: iets bredere glans
        band_h = 4 if (not is_black) else 5
        band_end = int(h * (0.45 if (not is_black) else 0.58))
        band_end = max(band_h, min(h, band_end))
        max_alpha = 38 if (not is_black) else 26
        for y in range(bevel + 1, band_end, band_h):
            t = y / float(max(1, band_end))
            a = int(max_alpha * (1.0 - t))
            # Wit: iets grijzer zodat glans zichtbaar is.
            # Zwart: neutrale lichte glans.
            if is_black:
                c = (255, 255, 255, a)
            else:
                c = (212, 212, 212, a)
            overlay.fill(c, rect=pygame.Rect(bevel + 1, y, w - (bevel + 1) * 2, band_h))
        surface.blit(overlay, (0, 0))

    # Subtiele pixel-textuur: alleen op witte toetsen.
    # Zwarte toetsen: geen stipjes (clean).
    if (not is_black) and w > 20 and h > 20:
        step = 7
        dot = 2
        for y in range(bevel + 2, h - bevel - 2, step):
            for x in range(bevel + 2, w - bevel - 2, step):
                if rng.random() < 0.16:
                    if ((x // step) + (y // step)) % 2 == 0:
                        c = _offset_color(base, -8, -8, -8)
                    else:
                        c = _offset_color(base, 10, 10, 10)
                    pygame.draw.rect(surface, c, pygame.Rect(x, y, dot, dot))

    # Geen extra lijnen op witte toetsen: clean vlak onder de letters.
    # Zwarte toetsen houden een héél subtiele groef voor textuur.
    if is_black:
        line_c = _offset_color(base, -16, -16, -16)
        y = int(h * 0.82)
        if bevel + 2 < y < h - bevel - 2:
            pygame.draw.line(surface, line_c, (bevel + 2, y), (w - bevel - 3, y), 1)

    # Buitenrand strak (retro UI)
    pygame.draw.rect(surface, edge, pygame.Rect(0, 0, w, h), 1)

def apply_tile_pixel_gradient(surface: pygame.Surface, *, base_color, seed: int = 0) -> None:
    """Pixel-achtige gradient voor gele tiles (Terraria/Minecraft vibe)."""
    w, h = surface.get_size()
    rng = random.Random(int(seed) & 0xFFFFFFFF)

    # Verticale gradient: boven iets lichter, onder iets donkerder
    for y in range(h):
        t = 0.0 if h <= 1 else (float(y) / float(h - 1))
        # Lichte top, donkere bottom
        dr = int(18 * (1.0 - t) - 14 * t)
        dg = int(16 * (1.0 - t) - 12 * t)
        db = int(6 * (1.0 - t) - 10 * t)
        c = _offset_color(base_color, dr, dg, db)
        pygame.draw.line(surface, c, (0, y), (w, y))

    # Pixel-noise voor textuur
    if w > 6 and h > 6:
        step = 4
        for y in range(1, h - 1, step):
            for x in range(1, w - 1, step):
                if rng.random() < 0.25:
                    jitter = rng.randint(-18, 18)
                    c = _offset_color(base_color, jitter, jitter, rng.randint(-8, 8))
                    pygame.draw.rect(surface, c, pygame.Rect(x, y, 2, 2))

    # Subtiele rand om de tile iets meer "blocky" te maken
    edge = _offset_color(base_color, -28, -28, -28)
    pygame.draw.rect(surface, edge, pygame.Rect(0, 0, w, h), 1)

# Game states
STATE_MENU = "menu"
STATE_PLAY_SELECT = "play_select"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"
STATE_LEVELS = "levels"
STATE_MIDI_INFO = "midi_info"
STATE_SETTINGS = "settings"
STATE_SHOP = "shop"
STATE_PAUSED = "paused"
STATE_COUNTDOWN = "countdown"
STATE_TRANSITION = "transition"
STATE_LEVEL_COMPLETED = "level_completed"
STATE_AUTO_REPLAY = "auto_replay"
STATE_TUTORIAL_COMPLETED = "tutorial_completed"
STATE_TUTORIAL_WELCOME = "tutorial_welcome"

# Kolom labels
KEY_LABELS = {
    0: 'A', 1: 'W', 2: 'S', 3: 'E', 4: 'D',
    5: 'C', 6: 'U', 7: 'J', 8: 'I', 9: 'K', 10: 'O', 11: 'L'
}

STRAIGHT_KEY_LABELS = {
    0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
    5: '6', 6: '7', 7: '8', 8: '9', 9: '0', 10: '-', 11: '='
}

NOTE_LABELS = {
    0: 'C3', 1: 'C#3', 2: 'D3', 3: 'D#3', 4: 'E3', 5: 'F3',
    6: 'F#3', 7: 'G3', 8: 'G#3', 9: 'A3', 10: 'A#3', 11: 'B3'
}

MIDI_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Piano frequenties voor MIDI noten 48-59 (C3 tot B3) - één octaaf hoger
# Formule: freq = 440 * 2^((n-69)/12) waar n = MIDI note nummer
NOTE_FREQUENCIES = {
    48: 130.81,   # C3  (A)
    49: 138.59,   # C#3 (W)
    50: 146.83,   # D3  (S)
    51: 155.56,   # D#3 (E)
    52: 164.81,   # E3  (D)
    53: 174.61,   # F3  (C)
    54: 185.00,   # F#3 (U)
    55: 196.00,   # G3  (J)
    56: 207.65,   # G#3 (I)
    57: 220.00,   # A3  (K)
    58: 233.08,   # A#3 (O)
    59: 246.94    # B3  (L)
}

def get_frequency_for_midi_note(note_num):
    """Bereken de frequentie voor elke MIDI-noot.
    
    Gebruikt de formule: freq = 440 * 2^((n-69)/12)
    waarbij n = MIDI note nummer (69 is A4 = 440Hz)
    """
    try:
        note_num = int(note_num)
        # Controleer of het in onze pre-berekende tabel staat
        if note_num in NOTE_FREQUENCIES:
            return NOTE_FREQUENCIES[note_num]
        # Bereken voor alle andere noten
        freq = 440.0 * (2.0 ** ((float(note_num) - 69.0) / 12.0))
        return float(freq)
    except Exception:
        # Fallback naar A4
        return 440.0

def generate_piano_sound(frequency, duration=0.3, sample_rate=SOUND_SAMPLE_RATE, volume=0.28):
    """Genereer een piano-achtig geluid met harmonics en ADSR envelope.
    
    Gebruikt meerdere harmonics om een realistischer piano timbre te creëren,
    en een ADSR envelope om clicks te voorkomen en natuurlijke attack/decay te geven.
    """
    try:
        frequency = float(frequency)
    except Exception:
        frequency = 440.0
    try:
        duration = float(duration)
    except Exception:
        duration = 0.3
    duration = max(0.001, duration)

    num_samples = int(duration * float(sample_rate))
    if num_samples < 2:
        num_samples = 2
    
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    
    # Maak piano-achtige harmonics (fundamental + overtones)
    # Piano heeft sterke fundamentele toon + zwakkere harmonics
    wave = np.zeros(num_samples, dtype=np.float32)
    
    # Fundamental frequency (sterkste component)
    wave += 1.0 * np.sin(2.0 * np.pi * frequency * t)
    
    # 2de harmonic (octaaf hoger, zwakker)
    wave += 0.5 * np.sin(2.0 * np.pi * frequency * 2.0 * t)
    
    # 3de harmonic (kwint hoger)
    wave += 0.25 * np.sin(2.0 * np.pi * frequency * 3.0 * t)
    
    # 4de harmonic
    wave += 0.15 * np.sin(2.0 * np.pi * frequency * 4.0 * t)
    
    # 5de harmonic (zachter)
    wave += 0.1 * np.sin(2.0 * np.pi * frequency * 5.0 * t)
    
    # Normaliseer de golf
    max_amplitude = np.abs(wave).max()
    if max_amplitude > 0:
        wave = wave / max_amplitude
    
    # ADSR Envelope voor natuurlijk piano-geluid
    # Attack: 5ms snelle attack (piano hammer)
    attack_samples = int(0.005 * sample_rate)
    # Decay: 50ms geleidelijke decay
    decay_samples = int(0.05 * sample_rate)
    # Sustain level: 70% van max volume
    sustain_level = 0.7
    # Release: rest van de noot met exponentiële decay
    
    envelope = np.ones(num_samples, dtype=np.float32)
    
    # Attack fase
    if attack_samples > 0 and attack_samples < num_samples:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)
    
    # Decay fase
    decay_end = min(attack_samples + decay_samples, num_samples)
    if decay_end > attack_samples:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_end - attack_samples, dtype=np.float32)
    
    # Sustain + Release fase (exponentiële decay)
    if decay_end < num_samples:
        release_samples = num_samples - decay_end
        # Exponentiële decay voor natuurlijk piano release
        decay_curve = np.exp(-3.0 * np.linspace(0, 1, release_samples, dtype=np.float32))
        envelope[decay_end:] = sustain_level * decay_curve
    
    # Fade-out aan het einde om clicks te voorkomen (laatste 5ms)
    fade_samples = min(int(0.005 * sample_rate), num_samples // 4)
    if fade_samples > 0:
        envelope[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    
    # Pas envelope toe op de golf
    wave = wave * envelope
    
    # Converteer naar 16-bit stereo
    wave_i16 = np.clip(wave * (32767.0 * float(volume)), -32767, 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave_i16, wave_i16))
    
    return pygame.sndarray.make_sound(stereo_wave)

def generate_piano_sustain_sound(frequency, duration=SUSTAIN_SOUND_SECONDS, sample_rate=SOUND_SAMPLE_RATE, volume=0.18):
    """Genereer een sustain geluid voor hold notes met harmonics en natuurlijke decay.
    
    Gebruikt dezelfde harmonics als de tap sound maar met een langere, geleidelijke decay
    voor hold notes.
    """
    try:
        frequency = float(frequency)
    except Exception:
        frequency = 440.0
    try:
        duration = float(duration)
    except Exception:
        duration = float(SUSTAIN_SOUND_SECONDS)
    duration = max(0.001, duration)

    num_samples = int(duration * float(sample_rate))
    if num_samples < 2:
        num_samples = 2
    
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    
    # Maak piano-achtige harmonics (zelfde als tap sound)
    wave = np.zeros(num_samples, dtype=np.float32)
    
    # Fundamental frequency
    wave += 1.0 * np.sin(2.0 * np.pi * frequency * t)
    
    # Harmonics
    wave += 0.5 * np.sin(2.0 * np.pi * frequency * 2.0 * t)
    wave += 0.25 * np.sin(2.0 * np.pi * frequency * 3.0 * t)
    wave += 0.15 * np.sin(2.0 * np.pi * frequency * 4.0 * t)
    wave += 0.1 * np.sin(2.0 * np.pi * frequency * 5.0 * t)
    
    # Normaliseer
    max_amplitude = np.abs(wave).max()
    if max_amplitude > 0:
        wave = wave / max_amplitude
    
    # Langzame exponentiële decay voor sustain
    envelope = np.exp(-0.8 * np.linspace(0, 1, num_samples, dtype=np.float32))
    
    # Zachte fade-in (10ms) om clicks te voorkomen
    fade_in_samples = int(0.01 * sample_rate)
    if fade_in_samples > 0 and fade_in_samples < num_samples:
        envelope[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples, dtype=np.float32)
    
    # Zachte fade-out aan het einde
    fade_out_samples = min(int(0.02 * sample_rate), num_samples // 4)
    if fade_out_samples > 0:
        envelope[-fade_out_samples:] *= np.linspace(1, 0, fade_out_samples, dtype=np.float32)
    
    # Pas envelope toe
    wave = wave * envelope
    
    # Converteer naar 16-bit stereo
    wave_i16 = np.clip(wave * (32767.0 * float(volume)), -32767, 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave_i16, wave_i16))
    
    return pygame.sndarray.make_sound(stereo_wave)

def generate_metronome_click(frequency, duration=0.05, sample_rate=SOUND_SAMPLE_RATE, volume=0.3):
    """Genereer een metronoom click geluid.
    
    Args:
        frequency: Frequentie van de click (Hz)
        duration: Duur in seconden
        sample_rate: Sample rate
        volume: Volume (0.0-1.0)
    """
    try:
        frequency = float(frequency)
    except Exception:
        frequency = 800.0
    try:
        duration = float(duration)
    except Exception:
        duration = 0.05
    duration = max(0.001, min(duration, 0.2))
    
    num_samples = int(duration * float(sample_rate))
    if num_samples < 2:
        num_samples = 2
    
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    
    # Sinusgolf met decay
    wave = np.sin(2.0 * np.pi * frequency * t).astype(np.float32)
    
    # Snelle exponentiële decay
    envelope = np.exp(-8.0 * np.linspace(0, 1, num_samples, dtype=np.float32))
    
    # Zachte fade-in
    fade_in_samples = max(1, int(0.002 * sample_rate))
    if fade_in_samples < num_samples:
        envelope[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples, dtype=np.float32)
    
    # Pas envelope toe
    wave = wave * envelope
    
    # Converteer naar 16-bit stereo
    wave_i16 = np.clip(wave * (32767.0 * float(volume)), -32767, 32767).astype(np.int16)
    stereo_wave = np.column_stack((wave_i16, wave_i16))
    
    return pygame.sndarray.make_sound(stereo_wave)

def get_midi_note_name(note_num):
    octave = (note_num // 12) - 1
    note_name = MIDI_NOTE_NAMES[note_num % 12]
    return f"{note_name}{octave}"

def transpose_to_range(note_num, min_note=36, max_note=47):
    while note_num < min_note:
        note_num += 12
    while note_num > max_note:
        note_num -= 12
    return note_num

class Tile:
    def __init__(self, column, length_pixels=None, note_num=None, font=None, label_text=None):
        self.column = column
        
        # Breedte afhankelijk van zwart/wit toets
        if column in BLACK_INDICES:
            self.width = column_widths[column] - 2
        else:
            self.width = column_widths[column] - 4
        
        # Lengte bepalen
        if length_pixels is None:
            self.height = self.width
        else:
            self.height = int(length_pixels)
        
        # Positie bepalen
        if column in BLACK_INDICES:
            prev_column_end = column_x[column] + column_widths[column]
            self.x = prev_column_end - self.width
        else:
            self.x = column_x[column] + 2
        
        self.y = -self.height
        self.speed = TILE_SPEED
        self.hit = False
        self.missed = False
        self.holding = False
        self.latched = False
        self.hold_frames_total = 0
        self.hold_frames_elapsed = 0
        self.hold_frames_required = 0
        self.fade_timer = 0
        self.fade_duration = 20
        self.note_num = note_num
        if label_text is None:
            self.label = KEY_LABELS.get(column, '?')
        else:
            self.label = label_text

        # OPTIMALISATIE: Maak surfaces en tekst één keer aan
        self.surface = pygame.Surface((self.width, self.height)).convert_alpha()
        # Pixel texture voor gele tiles (Terraria/Minecraft vibe)
        self.texture_surface = pygame.Surface((self.width, self.height)).convert_alpha()
        apply_tile_pixel_gradient(self.texture_surface, base_color=TILE_COLOR, seed=(column * 997 + self.height * 13))
        
        # Pre-render tekst voor elke state
        if font:
            self.text_surf = font.render(self.label, True, BLACK).convert_alpha()
            # Plaats tekst helemaal onderaan de tile
            self.text_rect = self.text_surf.get_rect(center=(self.width // 2, self.height - 10))
        else:
            self.text_surf = None

    def update(self):
        if not self.hit and not self.missed:
            if self.holding and self.latched:
                # Tijdens hold: blijf gewoon met dezelfde snelheid naar beneden gaan.
                # Clampen op de bar-bottom kan de indruk geven dat de balk "wacht"
                # en door afronding zelfs een klein sprongetje kan tonen.
                self.y += self.speed
            else:
                self.y += self.speed
        elif self.hit:
            # Tijdens fade-out: blijf ook vallen (zodat een losgelaten hold niet "stil" hangt).
            self.y += self.speed
            self.fade_timer += 1

    def draw(self, screen, show_green_hit=True):
        # Tijdens een correcte hold (latched): laat de tile zichtbaar, maar toon alleen
        # het "overgebleven" deel (shrinks) zodat de speler ziet wanneer hij kan loslaten.
        if self.holding and self.latched and (not self.hit) and (not self.missed):
            # Gebruik float-based berekeningen om 1px "jitter" door afronding te voorkomen.
            speed = float(self.speed) if self.speed else 0.0
            if speed <= 0.0:
                speed = 1.0

            # Hoeveel van de noot zit al "achter" de toets (vanaf de bovenrand van de bar)?
            # Belangrijk: NIET clampen op BAR_HEIGHT, anders kan de zichtbare balk
            # tijdelijk "stil" lijken te staan wanneer hij die diepte bereikt.
            entered_px = max(0.0, float(self.y + self.height) - float(BAR_Y))
            entered_px = min(float(self.height), entered_px)

            # Overblijvende lengte op basis van tijd/progress (monotonic dalend).
            remaining_time_px = max(0.0, float(self.height) - (float(self.hold_frames_elapsed) * speed))
            # Deel dat nog boven de toets zit (monotonic dalend totdat head volledig in bar zit).
            above_key_px = max(0.0, float(self.height) - entered_px)

            visible_px_f = min(remaining_time_px, above_key_px)
            visible_px = int(visible_px_f)  # trunc = floor voor positieve waarden
            if visible_px <= 0:
                return

            # Tijdens hold: laat direct de "hit" kleur zien zodra je indrukt.
            hold_color = TILE_HIT_COLOR if show_green_hit else TILE_COLOR
            self.surface.fill(hold_color)
            self.surface.set_alpha(255)

            # Neem het deel net boven het "verborgen" stuk (entered_px).
            src_top = int(float(self.height) - entered_px - float(visible_px))
            if src_top < 0:
                src_top = 0
            src_rect = pygame.Rect(0, src_top, self.width, visible_px)

            # Anker op de bovenkant van de toets (BAR_Y)
            dst_y = int(float(BAR_Y) - float(visible_px))
            screen.blit(self.surface, (self.x, dst_y), area=src_rect)
            return

        # Bepaal kleur en alpha
        if self.missed:
            color = TILE_MISS_COLOR
            alpha = 255
        elif self.hit:
            progress = self.fade_timer / self.fade_duration
            alpha = int(255 * (1 - progress))
            # Altijd blauw bij hit (zoals gevraagd)
            color = TILE_HIT_COLOR
        elif self.holding:
            # Zodra je een (hold) noot indrukt, meteen groen tonen (ook vóór de uiteindelijke score).
            color = TILE_HIT_COLOR if show_green_hit else TILE_COLOR
            alpha = 255
        else:
            color = TILE_COLOR
            alpha = 255

        # Vul surface met kleur / texture
        if (not self.missed) and (not self.hit) and (not self.holding):
            # Gebruik pixel texture voor de gele tile
            self.surface.blit(self.texture_surface, (0, 0))
        else:
            self.surface.fill(color)
        
        # Teken tekst als die bestaat
        if self.text_surf:
            self.surface.blit(self.text_surf, self.text_rect)
        
        # Set alpha en teken op scherm
        self.surface.set_alpha(alpha)
        screen.blit(self.surface, (self.x, self.y))

    def is_on_bar(self):
        # Iets ruimere hitbox rond de balk zodat raak-momenten
        # bij MIDI-noten beter geregistreerd worden.
        margin = 10
        return (
            self.y + self.height >= BAR_Y - margin and
            self.y <= BAR_Y + BAR_HEIGHT + margin
        )

    def is_head_on_bar(self):
        # Voor (lange) noten: check alleen de "kop" (onderkant van de tile)
        # zodat je niet te vroeg kunt starten doordat de tile heel lang is.
        margin = int(HIT_MARGIN_PX)
        head_y = self.y + self.height
        return (BAR_Y - margin) <= head_y <= (BAR_Y + BAR_HEIGHT + margin)

    def is_missed(self):
        return self.y + self.height >= SCREEN_HEIGHT

    def mark_as_hit(self):
        self.hit = True

class Bar:
    def __init__(self, column, font=None, label_text=None):
        self.column = column
        self.y = BAR_Y
        
        # Zwarte toetsen tegen rechterrand plaatsen
        if column in BLACK_INDICES:
            self.width = column_widths[column] - 2
            prev_column_end = column_x[column] + column_widths[column]
            self.x = prev_column_end - self.width
            self.height = BAR_HEIGHT
            fill_color = BAR_DARK_COLOR
            label_color = WHITE
        else:
            self.x = column_x[column] + 2
            self.width = column_widths[column] - 4
            self.height = BAR_HEIGHT
            fill_color = BAR_COLOR
            label_color = BLACK
        
        if label_text is None:
            self.label = KEY_LABELS.get(column, '?')
        else:
            self.label = label_text
        
        # OPTIMALISATIE: Pre-render de bar surface
        self.surface = pygame.Surface((self.width, self.height)).convert()

        # Retro/pixel textuur (clean maar niet "plat")
        is_black = column in BLACK_INDICES
        # Base tint iets mooier dan volledig wit/zwart
        base = (24, 24, 26) if is_black else (220, 220, 220)
        apply_retro_key_texture(self.surface, base_color=base, is_black=is_black, seed=column)
        
        # Pre-render tekst
        if font:
            # Gebruik per-pixel alpha zodat er geen 'vierkant' om de letters verschijnt.
            # Anti-alias uit = strakker/retro en voorkomt halo-achtige randen op texture.
            text_surf = font.render(self.label, False, label_color).convert_alpha()
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            self.surface.blit(text_surf, text_rect)
    
    def draw(self, screen):
        screen.blit(self.surface, (self.x, self.y))

class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.hovered = False
        self.enabled = True
        # Optional: tweak text position / force consistent font size
        self.text_y_offset = 0
        self.fixed_font_size = None
    
    def draw(self, screen):
        if not getattr(self, 'enabled', True):
            color = (70, 70, 70)
            border = (160, 160, 160)
            text_color = (210, 210, 210)
        else:
            color = MENU_BUTTON_HOVER if self.hovered else MENU_BUTTON_COLOR
            border = WHITE
            text_color = WHITE

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border, self.rect, 2)

        # Auto-fit tekst zodat labels (zoals Settings tabs) altijd netjes passen.
        # Voor tabs kunnen we fixed_font_size zetten zodat alle tabs dezelfde grootte hebben.
        font_size = 30
        min_font_size = 14
        padding_x = 10
        padding_y = 8

        forced = getattr(self, 'fixed_font_size', None)
        if forced is not None:
            try:
                font_size = int(forced)
            except Exception:
                font_size = 30
            font = pygame.font.SysFont('opensans', font_size)
            text_surface = font.render(self.text, True, text_color)
        else:
            while True:
                font = pygame.font.SysFont('opensans', font_size)
                text_surface = font.render(self.text, True, text_color)
                if (
                    text_surface.get_width() <= (self.rect.width - padding_x) and
                    text_surface.get_height() <= (self.rect.height - padding_y)
                ):
                    break
                if font_size <= min_font_size:
                    break
                font_size -= 2

        text_rect = text_surface.get_rect(center=self.rect.center)
        try:
            text_rect.centery += int(getattr(self, 'text_y_offset', 0))
        except Exception:
         
            pass
        screen.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        if not getattr(self, 'enabled', True):
            return False
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        if not getattr(self, 'enabled', True):
            self.hovered = False
            return
        self.hovered = self.rect.collidepoint(pos)

class RefreshButton:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, size, size)
        self.size = size
        self.hovered = False
        self.enabled = True
    
    def draw(self, screen):
        if not self.enabled:
            color = (70, 70, 70)
            border = (160, 160, 160)
            icon_color = (210, 210, 210)
        else:
            color = MENU_BUTTON_HOVER if self.hovered else MENU_BUTTON_COLOR
            border = WHITE
            icon_color = WHITE
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border, self.rect, 2)
        
        # Teken refresh icon (twee gebogen pijlen, geen cirkel)
        cx = self.x + self.size // 2
        cy = self.y + self.size // 2
        margin = int(self.size * 0.18)
        arrow_len = int(self.size * 0.22)
        thickness = 3

        def draw_arrowhead(end_x, end_y, dir_x, dir_y, size=6):
            # dir_x, dir_y = richting van de pijl (genormaliseerd op axis)
            if dir_x == 1 and dir_y == 0:
                pts = [(end_x, end_y), (end_x - size, end_y - size), (end_x - size, end_y + size)]
            elif dir_x == -1 and dir_y == 0:
                pts = [(end_x, end_y), (end_x + size, end_y - size), (end_x + size, end_y + size)]
            elif dir_x == 0 and dir_y == 1:
                pts = [(end_x, end_y), (end_x - size, end_y - size), (end_x + size, end_y - size)]
            else:  # dir_x == 0 and dir_y == -1
                pts = [(end_x, end_y), (end_x - size, end_y + size), (end_x + size, end_y + size)]
            pygame.draw.polygon(screen, icon_color, pts)

        # Bovenste pijl: links->rechts, dan omlaag (↱)
        top_y = cy - margin
        left_x = cx - margin
        right_x = cx + margin
        pygame.draw.line(screen, icon_color, (left_x, top_y), (right_x, top_y), thickness)
        pygame.draw.line(screen, icon_color, (right_x, top_y), (right_x, top_y + arrow_len), thickness)
        draw_arrowhead(right_x, top_y + arrow_len, 0, 1, size=6)

        # Onderste pijl: rechts->links, dan omhoog (↰)
        bottom_y = cy + margin
        pygame.draw.line(screen, icon_color, (right_x, bottom_y), (left_x, bottom_y), thickness)
        pygame.draw.line(screen, icon_color, (left_x, bottom_y), (left_x, bottom_y - arrow_len), thickness)
        draw_arrowhead(left_x, bottom_y - arrow_len, 0, -1, size=6)
    
    def is_clicked(self, pos):
        if not self.enabled:
            return False
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        if not self.enabled:
            self.hovered = False
            return
        self.hovered = self.rect.collidepoint(pos)

class DisconnectButton:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, size, size)
        self.size = size
        self.hovered = False
        self.enabled = True
    
    def draw(self, screen):
        if not self.enabled:
            color = (70, 70, 70)
            border = (160, 160, 160)
            icon_color = (210, 210, 210)
        else:
            color = MENU_BUTTON_HOVER if self.hovered else MENU_BUTTON_COLOR
            border = WHITE
            icon_color = WHITE
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border, self.rect, 2)
        
        # Teken X icon (kruis)
        cx = self.x + self.size // 2
        cy = self.y + self.size // 2
        margin = int(self.size * 0.25)
        thickness = 3
        
        # Diagonaal linksomhoog naar rechtsomlaag
        pygame.draw.line(screen, icon_color, (cx - margin, cy - margin), (cx + margin, cy + margin), thickness)
        # Diagonaal rechtsboven naar linksomlaag
        pygame.draw.line(screen, icon_color, (cx + margin, cy - margin), (cx - margin, cy + margin), thickness)
    
    def is_clicked(self, pos):
        if not self.enabled:
            return False
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        if not self.enabled:
            self.hovered = False
            return
        self.hovered = self.rect.collidepoint(pos)

class ArrowButton:
    def __init__(self, x, y, width, height, direction):
        """
        direction: 'left' of 'right'
        """
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, width, height)
        self.width = width
        self.height = height
        self.direction = direction
        self.hovered = False
        self.enabled = True
    
    def draw(self, screen):
        if not self.enabled:
            color = (70, 70, 70)
            border = (160, 160, 160)
            arrow_color = (210, 210, 210)
        else:
            color = MENU_BUTTON_HOVER if self.hovered else MENU_BUTTON_COLOR
            border = WHITE
            arrow_color = WHITE
        
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, border, self.rect, 2)
        
        # Teken pijl symbool (driehoek)
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        arrow_size = min(self.width, self.height) // 3
        
        if self.direction == 'left':
            # Pijl naar links: driehoek met punt naar links
            points = [
                (center_x - arrow_size, center_y),  # Punt (links)
                (center_x + arrow_size, center_y - arrow_size),  # Boven rechts
                (center_x + arrow_size, center_y + arrow_size)   # Onder rechts
            ]
        else:  # 'right'
            # Pijl naar rechts: driehoek met punt naar rechts
            points = [
                (center_x + arrow_size, center_y),  # Punt (rechts)
                (center_x - arrow_size, center_y - arrow_size),  # Boven links
                (center_x - arrow_size, center_y + arrow_size)   # Onder links
            ]
        
        pygame.draw.polygon(screen, arrow_color, points)
    
    def is_clicked(self, pos):
        if not self.enabled:
            return False
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        if not self.enabled:
            self.hovered = False
            return
        self.hovered = self.rect.collidepoint(pos)

class UploadButton:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, size, size)
        self.size = size
        self.hovered = False
    
    def draw(self, screen):
        color = MENU_BUTTON_HOVER if self.hovered else MENU_BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        
        # Teken "+" symbooltje (perfect gecentreerd)
        font = pygame.font.SysFont('opensans', 48)
        text_surface = font.render("+", True, WHITE)
        text_rect = text_surface.get_rect(center=(self.x + self.size // 2, self.y + self.size // 2))
        screen.blit(text_surface, text_rect)
    
    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)
    
    def update_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Fallin' Keys")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = STATE_MENU
        self.score = 0
        self.spawn_counter = 0
        self.miss_pending = False
        self.miss_timer = 0

        # Wrong-key feedback (toon eerst rood, dan pas game over)
        self.wrong_key_column = None

        # Game over overlay (smooth blur + panel)
        self.game_over_frame = 0
        self.game_over_bg = None
        self.game_over_bg_blur = None

        # Level rating (stars)
        self.level_stars = None  # None wanneer niet van toepassing
        self.level_note_count = 0
        self.level_score_thresholds = None  # {'allright': int, 'great': int, 'perfect': int}

        # Per-level best rating (Levels menu)
        self.level_ratings = {}  # level_id -> best stars (0..3)
        self.current_level_id = None
        self.play_time = 0  # Tijd in frames
        self.current_fps = 60
        self.show_note_names = False
        self.cheats_enabled = False
        self.easy_hold_enabled = False

        # Tutorial
        self.tutorial_mode = False
        self.tutorial_complete_bg = None
        self.tutorial_complete_bg_blur = None
        self.tutorial_complete_frame = 0

        # Pixel smoke particles (boven toetsen)
        self.smoke_particles = []  # list[SmokeParticle]

        # Judgement UI (Perfect/Great/Allright)
        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.last_judgement_duration = int(FPS * 0.9)
        self.last_judgement_age = 0  # frames sinds laatste judgement-set (voor fade/slide-in)
        # Vorige judgements die naar beneden schuiven + vervagen zodra er een nieuwe komt.
        # items: {'text': str, 'age': int, 'alpha0': int, 'y0': int}
        self.judgement_history = []

        # Combo / multiplier UI (x2, x3, ...)
        # Loopt op bij opeenvolgende Perfect/Great/Allright en reset bij miss/game over.
        self.combo_streak = 0
        self.combo_kind = None  # 'Perfect' | 'Great' | 'Allright'

        # Settings state
        self.settings_tab = "gameplay"  # 'gameplay' | 'visuals' | 'audio' | 'midi'
        # Waar Settings naar terug moet (menu of pause)
        self.settings_return_state = STATE_MENU
        # Blauw bij hit
        self.green_note_enabled = True

        # Audio settings
        # Wanneer Cheats aan staan en een noot vanzelf verdwijnt, speel dan (voor lange noten)
        # een korte sustain zodat je toch iets "langers" hoort.
        self.cheat_auto_sustain_enabled = True
        self.cheat_auto_sustain_seconds = 0.3

        # Pause / countdown state
        self.pause_background = None
        self.pause_background_blurred = None
        self.countdown_value = 3
        self.countdown_timer = 0

        # UI transitions
        self.transition_from = None
        self.transition_to = None
        self.transition_dir = None  # 'down' | 'right'
        self.transition_frame = 0
        self.transition_duration = int(FPS * 0.25)
        self.transition_from_surf = None
        self.transition_to_surf = None

        # Input / hold-notes
        self.pressed_columns = set()
        self.active_holds = {}  # column -> Tile
        # Track welke keys daadwerkelijk ingedrukt zijn (om OS key-repeat te negeren)
        self.keys_currently_pressed = set()
        
        # OPTIMALISATIE: Fonts één keer laden - Open Sans
        self.font_small = pygame.font.SysFont('opensans', 20)
        self.font_medium = pygame.font.SysFont('opensans', 24)
        # Extra kleinere fonts voor smalle (zwarte) toetsen in note-name modus
        self.font_medium_small = pygame.font.SysFont('opensans', 16)
        self.font_bar = pygame.font.SysFont('opensans', 32, bold=True)
        self.font_bar_small = pygame.font.SysFont('opensans', 20, bold=True)
        self.font_large = pygame.font.SysFont('opensans', 36)
        self.font_xlarge = pygame.font.SysFont('opensans', 48)
        
        # Keys carousel variabelen (voor get_label_for_column)
        self.keys_carousel_options = ["Keys", "Note names", "Straight keys"]
        self.keys_carousel_current_index = 0  # Index van de huiding getoonde optie
        self.keys_carousel_previous_index = 0  # Voor animatie: de vorige index
        self.keys_carousel_slide_frames = 0  # Animatie counter
        self.keys_carousel_slide_direction = 0  # 1=van links, -1=van rechts, 0=geen animatie
        self.KEYS_CAROUSEL_SLIDE_DURATION = int(FPS * 0.3)  # ~0.3s animatie
        
        # OPTIMALISATIE: Maak een achtergrond surface met lijnen
        self.background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)).convert()
        self.background.fill(DARK_GRAY)
        for i in range(1, NUM_COLUMNS):
            x = column_x[i]
            pygame.draw.line(self.background, BLACK, (x, 0), (x, SCREEN_HEIGHT), 2)
        
        # Maak bars (nu met font parameter)
        self.bars = [Bar(i, self.get_bar_font_for_column(i), label_text=self.get_label_for_column(i)) for i in range(NUM_COLUMNS)]
        
        self.tiles = []
        
        # Menu buttons
        button_width = 150
        button_height = 50
        button_x = (SCREEN_WIDTH - button_width) // 2
        self.play_button = Button(button_x, 220, button_width, button_height, "Play")
        self.levels_button = Button(button_x, 290, button_width, button_height, "Levels")
        # Swap: Shop komt op de oude Settings plek, Settings eronder
        self.shop_button = Button(button_x, 360, button_width, button_height, "Shop")
        self.settings_button = Button(button_x, 430, button_width, button_height, "Settings")

        # Shop state
        self.shop_tab = "themes"  # 'themes' | 'background' | 'instruments'
        self.shop_swipe_frame = 0  # Voor swipe animatie wanneer instruments geselecteerd wordt
        self.shop_swipe_duration = int(FPS * 0.5)  # 0.5 seconden animatie
        self.shop_swipe_direction = 0  # 1 = forward (naar instruments), -1 = reverse (terug)
        self.shop_selected_instrument = (0, 0)  # (row, col) - default het eerste vakje (E-piano)
        shop_btn_w = 150
        shop_btn_h = 50
        shop_left_x = 20
        shop_top_y = 190
        shop_gap_y = 16
        self.shop_themes_button = Button(shop_left_x, shop_top_y, shop_btn_w, shop_btn_h, "Themes")
        self.shop_background_button = Button(shop_left_x, shop_top_y + shop_btn_h + shop_gap_y, shop_btn_w, shop_btn_h, "Background")
        self.shop_instruments_button = Button(shop_left_x, shop_top_y + (shop_btn_h + shop_gap_y) * 2, shop_btn_w, shop_btn_h, "Instruments")

        # Play mode select
        self.free_play_button = Button((SCREEN_WIDTH - 240) // 2, 240, 240, 60, "Free Play")
        self.tutorial_button = Button((SCREEN_WIDTH - 240) // 2, 320, 240, 60, "Tutorial")
        
        # Levels UI
        self.level1_midi_path = os.path.join(os.path.dirname(__file__), "Levels", "Sunrise.mid")
        self.level2_midi_path = os.path.join(os.path.dirname(__file__), "Levels", "Green Hills.mid")
        # Grotere balk zodat sterren onder de tekst passen
        self.level1_button = Button((SCREEN_WIDTH - 260) // 2, 140, 260, 80, "Level 1: sunrise")
        self.level1_button.text_y_offset = -12
        self.level2_button = Button((SCREEN_WIDTH - 260) // 2, 240, 260, 80, "Level 2: green hills")
        self.level2_button.text_y_offset = -12

        # Plus-button staat onder Level 2
        self.upload_button = UploadButton((SCREEN_WIDTH - 60) // 2, 340, 60)
        self.start_button = Button((SCREEN_WIDTH - 150) // 2, 420, 150, 50, "Check Code")
        self.play_now_button = Button((SCREEN_WIDTH - 150) // 2, SCREEN_HEIGHT - 100, 150, 40, "Start Game")
        self.label_mode_button = Button((SCREEN_WIDTH - 200) // 2, 240, 200, 50, "Keys")
        self.cheats_button = Button((SCREEN_WIDTH - 200) // 2, 350, 200, 50, "Cheats: OFF")
        self.easy_hold_button = Button((SCREEN_WIDTH - 240) // 2, 440, 240, 50, "Easy: OFF")

        # Keys carousel (Links pijl, Main button, Rechts pijl)
        arrow_size = 40
        main_button_x = (SCREEN_WIDTH - 200) // 2
        self.keys_carousel_left_arrow = ArrowButton(main_button_x - arrow_size - 10, 245, arrow_size, 40, 'left')
        self.keys_carousel_right_arrow = ArrowButton(main_button_x + 200 + 10, 245, arrow_size, 40, 'right')

        # Settings tabs + visuals
        tab_y = 140
        tab_w = 90
        tab_h = 42
        tab_gap = 10
        tab_total = tab_w * 4 + tab_gap * 3
        tab_left_x = (SCREEN_WIDTH - tab_total) // 2
        self.settings_gameplay_tab_button = Button(tab_left_x, tab_y, tab_w, tab_h, "Gameplay")
        self.settings_visuals_tab_button = Button(tab_left_x + (tab_w + tab_gap) * 1, tab_y, tab_w, tab_h, "Visuals")
        self.settings_audio_tab_button = Button(tab_left_x + (tab_w + tab_gap) * 2, tab_y, tab_w, tab_h, "Audio")
        self.settings_midi_tab_button = Button(tab_left_x + (tab_w + tab_gap) * 3, tab_y, tab_w, tab_h, "MIDI")

        # Tab-tekstgrootte: maak alle tabs exact zo groot als "Gameplay" (auto-fit size van de langste).
        tab_font_size = 30
        min_font_size = 14
        padding_x = 10
        padding_y = 8
        while True:
            font = pygame.font.SysFont('opensans', tab_font_size)
            s = font.render("Gameplay", True, WHITE)
            if (s.get_width() <= (tab_w - padding_x)) and (s.get_height() <= (tab_h - padding_y)):
                break
            if tab_font_size <= min_font_size:
                break
            tab_font_size -= 2

        self.settings_gameplay_tab_button.fixed_font_size = tab_font_size
        self.settings_visuals_tab_button.fixed_font_size = tab_font_size
        self.settings_audio_tab_button.fixed_font_size = tab_font_size
        self.settings_midi_tab_button.fixed_font_size = tab_font_size
        self.green_note_button = Button((SCREEN_WIDTH - 260) // 2, 260, 260, 50, "Blue hit: ON")

        self.cheat_sustain_button = Button((SCREEN_WIDTH - 260) // 2, 260, 260, 50, "Cheat sustain: ON")

        # MIDI input (device) settings
        self.midi_search_device_button = Button((SCREEN_WIDTH - 260) // 2, 220, 260, 50, "Search Device")
        self.midi_test_device_button = Button((SCREEN_WIDTH - 260) // 2, 280, 260, 50, "Test MIDI")
        # Refresh verschijnt naast 'Connected' wanneer er een device verbonden is (vierkant icon)
        self.midi_refresh_device_button = RefreshButton((SCREEN_WIDTH - 260) // 2 + 140, 220, 50)
        # Disconnect button naast refresh button
        self.midi_disconnect_device_button = DisconnectButton((SCREEN_WIDTH - 260) // 2 + 200, 220, 50)
        self.midi_input_supported = (pgmidi is not None)
        self.midi_input_initialized = False
        self.midi_input = None
        self.midi_input_device_id = None
        self.midi_input_device_name = None
        self.midi_input_connected = False
        self.midi_input_last_error = ""
        self.midi_input_last_search = "Nog niet gezocht"

        # Note-name auto toggle via MIDI connect/disconnect
        self.midi_forced_note_names = False

        # MIDI RX diagnostics
        self.midi_rx_count = 0
        self.midi_last_note_num = None
        self.midi_last_note_name = None
        self.midi_last_velocity = None
        # RX refresh check (voor unplug detectie)
        self.midi_rx_last_refresh_count = None

        # MIDI output (system synth) voor echte piano klank
        self.midi_output_supported = (pgmidi is not None)
        self.midi_output = None
        self.midi_output_device_id = None
        self.midi_output_device_name = None
        self.midi_output_connected = False
        self.midi_output_last_error = ""
        # Temporair uitgeschakeld - beter mixer sounds gebruiken
        self.use_midi_piano_output = False
        # Tap-notes: note_off scheduling zodat we niet hoeven te blocken.
        self._pending_midi_note_off = []  # list of (off_time_ms, note_num, channel)
        # Hold-notes: onthoud welke MIDI note actief is per column
        self._active_midi_hold_notes = {}  # column -> (note_num, channel)

        # Pause menu buttons
        # Volgorde: Continue (boven), Settings (midden), Return (onder)
        self.pause_continue_button = Button((SCREEN_WIDTH - 240) // 2, 250, 240, 55, "Continue")
        # Settings op de oude Continue-plek
        self.pause_settings_button = Button((SCREEN_WIDTH - 240) // 2, 320, 240, 55, "Settings")
        self.pause_return_button = Button((SCREEN_WIDTH - 240) // 2, 390, 240, 55, "Return to Main")
        self.midi_file_path = None
        self.midi_valid = False
        self.midi_notes = []
        self.use_midi_schedule = False
        # MIDI loading state (toon "Loading" na kiezen, analyse pas later)
        self.midi_loading = False
        self.midi_analysis_pending = False
        self.midi_loading_frames = 0
        self.scheduled_notes = []
        self.scheduled_notes_original = []
        self.game_frame = 0

        # MIDI info scroll/expand state
        self.midi_info_scroll = 0
        self.midi_info_max_scroll = 0
        self.midi_info_show_all = False
        self.midi_info_more_rect_kept = None
        self.midi_info_more_rect_removed = None
        self.midi_info_scrollbar_rect = None
        self.midi_info_scrollbar_handle_rect = None
        self.midi_info_scroll_dragging = False
        self.midi_info_scrollbar_drag_offset = 0
        self.midi_info_show_less_rect = None

        # Level-complete effect state
        self.played_notes_history = []  # list of {'column': int, 'height': int, 'y': float, 'note': int|None}
        self.celebration_tiles = []
        self.level_complete_frame = 0
        self.level_complete_bg = None
        self.level_complete_bg_blur = None

        # Auto-replay (na level completion): noten opnieuw vanaf begin, geen input nodig.
        self.auto_replay_done = False
        self.auto_replay_time = 0.0
        self.auto_replay_speed_mult = 2.0
        self.auto_replay_speed_growth = 1.0015  # per frame multiplicative
        self.auto_replay_speed_cap = 6.0
        self.auto_replay_frame = 0
        self.auto_replay_bg = None
        self.auto_replay_bg_blur = None

        self.level_complete_from_auto_replay = False

        # Tutorial welcome screen animatie
        self.tutorial_welcome_frame = 0
        self.tutorial_welcome_duration = int(FPS * 0.8)  # 0.8s om in te vallen

        # Tutorial slowmo + lichte blur na 2s vallen
        self.tutorial_slowmo_active = False
        self.tutorial_slowmo_after_frames = int(FPS * 1.0)
        self.tutorial_slowmo_multiplier = 0.12
        self.tutorial_slowmo_blur_alpha = 90
        self.tutorial_slowmo_start_frame = None
        self.tutorial_slowmo_done = False
        self.tutorial_overlay_active = False
        self.tutorial_overlay_step = 1
        self.tutorial_overlay_start_frame = None

        # Key mapping
        self.key_to_column = {
            pygame.K_a: 0, pygame.K_w: 1, pygame.K_s: 2, pygame.K_e: 3,
            pygame.K_d: 4, pygame.K_c: 5, pygame.K_u: 6, pygame.K_j: 7,
            pygame.K_i: 8, pygame.K_k: 9, pygame.K_o: 10, pygame.K_l: 11
        }
        
        # Straight keys mapping (cijfers bovenaan toetsenbord)
        self.straight_key_to_column = {
            pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
            pygame.K_5: 4, pygame.K_6: 5, pygame.K_7: 6, pygame.K_8: 7,
            pygame.K_9: 8, pygame.K_0: 9, pygame.K_MINUS: 10, pygame.K_EQUALS: 11
        }

        # Genereer piano geluiden voor elke kolom
        print("Generating piano sounds...")
        # Per kolom een tap-sound en een sustain-sound (voor hold-notes)
        self.piano_sounds = {}
        self.piano_sustain_sounds = {}
        self.piano_auto_sustain_sounds = {}
        # Cache voor alle MIDI-noten (om lag spikes te voorkomen)
        self.midi_note_sustain_cache = {}
        self.midi_note_tap_cache = {}
        # Dedicated channels voor sustain zodat we netjes kunnen loopen/stoppen.
        pygame.mixer.set_num_channels(max(64, NUM_COLUMNS * 4))
        self.note_channels = {col: pygame.mixer.Channel(col) for col in range(NUM_COLUMNS)}
        # Sustain-kanalen per column (laat polyfonie toe)
        self.sustain_channel_by_column = {}
        for col in range(NUM_COLUMNS):
            midi_note = 48 + col  # C3 tot B3 (één octaaf hoger)
            freq = NOTE_FREQUENCIES[midi_note]
            self.piano_sounds[col] = generate_piano_sound(freq)
            # Lange sustain (niet loopen) om hoorbare herhaling te voorkomen
            self.piano_sustain_sounds[col] = generate_piano_sustain_sound(freq, duration=SUSTAIN_SOUND_SECONDS, volume=0.18)
            # Korte sustain snippet voor cheats-auto playback
            self.piano_auto_sustain_sounds[col] = generate_piano_sustain_sound(freq, duration=0.35, volume=0.18)
        print("Piano sounds ready!")

        # Metronoom geluid generatie
        self.metronome_click_high = generate_metronome_click(1000.0, duration=0.05, volume=0.35)  # Beat 1 (accent)
        self.metronome_click_low = generate_metronome_click(600.0, duration=0.05, volume=0.25)   # Other beats
        
        # Metronoom state
        self.metronome_enabled = False
        self.metronome_current_beat = 0
        self.metronome_frame_counter = 0
        self.metronome_bpm = 120.0
        self.metronome_time_signature = (4, 4)
        self.metronome_frame_duration = 0  # Frames tot de volgende beat
        self.metronome_channel = None
        
        # Init MIDI device input (optioneel)
        self._init_midi_input_system()

        # Init MIDI output (optioneel) - gebruikt system synth (beste piano klank)
        self._init_midi_output_system()

    def _play_sound_on_channel(self, preferred_col: int, snd: pygame.mixer.Sound, volume: float | None = None):
        if snd is None:
            return
        ch = None
        try:
            ch = self.note_channels.get(int(preferred_col))
        except Exception:
            ch = None
        if ch is None:
            try:
                ch = pygame.mixer.find_channel(True)
            except Exception:
                ch = None
        else:
            try:
                if ch.get_busy():
                    # Gebruik extra channel om overlappende noten toe te laten
                    ch = pygame.mixer.find_channel(True) or ch
            except Exception:
                pass
        if ch is None:
            # Last resort
            try:
                snd.play()
            except Exception:
                pass
            return

        try:
            if volume is not None:
                ch.set_volume(float(volume))
        except Exception:
            pass
        ch.play(snd, loops=0)

    def _init_midi_input_system(self):
        if not self.midi_input_supported:
            self.midi_input_last_error = "pygame.midi niet beschikbaar"
            return
        try:
            if not self.midi_input_initialized:
                pgmidi.init()
                self.midi_input_initialized = True
        except Exception as e:
            self.midi_input_last_error = f"MIDI init fout: {e}"
            self.midi_input_initialized = False

    def _close_midi_input(self):
        try:
            if self.midi_input is not None:
                self.midi_input.close()
        except Exception:
            pass
        self.midi_input = None
        self.midi_input_device_id = None
        self.midi_input_device_name = None
        self.midi_input_connected = False

        # Als Note Names automatisch aan is gezet door MIDI: zet terug naar Keys bij disconnect.
        if getattr(self, 'midi_forced_note_names', False):
            self.midi_forced_note_names = False
            self.show_note_names = False
            try:
                self.label_mode_button.text = "Keys"
            except Exception:
                pass
            try:
                self.update_bar_labels()
            except Exception:
                pass

    def _init_midi_output_system(self):
        """Probeer te verbinden met een MIDI output device.

        Op Windows is dit vaak 'Microsoft GS Wavetable Synth' en klinkt veel meer als een piano
        dan een zelf-gegenereerde sinus/synth wave.
        """
        if not self.midi_output_supported:
            self.midi_output_last_error = "pygame.midi niet beschikbaar"
            self.midi_output_connected = False
            return

        if not self.use_midi_piano_output:
            self.midi_output_connected = False
            return

        # Zorg dat pgmidi.init() gedaan is (wordt ook door input gebruikt)
        self._init_midi_input_system()
        if not self.midi_input_initialized:
            self.midi_output_last_error = "MIDI init niet gelukt"
            self.midi_output_connected = False
            return

        # Als er al een output is, laat die staan.
        if self.midi_output_connected and self.midi_output is not None:
            return

        dev_id = -1
        try:
            dev_id = int(pgmidi.get_default_output_id())
        except Exception:
            dev_id = -1

        # Fallback: zoek de eerste output
        if dev_id < 0:
            try:
                count = int(pgmidi.get_count())
            except Exception:
                count = 0
            for i in range(count):
                try:
                    info = pgmidi.get_device_info(i)
                    if not info:
                        continue
                    # info = (interf, name, input, output, opened)
                    if int(info[3]) == 1:
                        dev_id = int(i)
                        break
                except Exception:
                    continue

        if dev_id < 0:
            self.midi_output_last_error = "Geen MIDI output device gevonden"
            self.midi_output_connected = False
            return

        try:
            self.midi_output = pgmidi.Output(dev_id)
            self.midi_output_device_id = int(dev_id)
            try:
                info = pgmidi.get_device_info(int(dev_id))
                if info and len(info) >= 2:
                    self.midi_output_device_name = info[1].decode(errors='ignore') if isinstance(info[1], (bytes, bytearray)) else str(info[1])
            except Exception:
                self.midi_output_device_name = None

            # GM program 0 = Acoustic Grand Piano
            try:
                self.midi_output.set_instrument(0, 0)
            except Exception:
                try:
                    self.midi_output.set_instrument(0)
                except Exception:
                    pass

            self._midi_all_notes_off()
            self.midi_output_connected = True
            self.midi_output_last_error = ""
        except Exception as e:
            self.midi_output = None
            self.midi_output_device_id = None
            self.midi_output_connected = False
            self.midi_output_last_error = f"MIDI output connect fout: {e}"

    def _close_midi_output(self):
        try:
            self._midi_all_notes_off()
        except Exception:
            pass
        try:
            if self.midi_output is not None:
                self.midi_output.close()
        except Exception:
            pass
        self.midi_output = None
        self.midi_output_device_id = None
        self.midi_output_device_name = None
        self.midi_output_connected = False
        self._pending_midi_note_off.clear()
        self._active_midi_hold_notes.clear()

    def _midi_all_notes_off(self):
        if not (self.midi_output_connected and self.midi_output is not None):
            return
        for ch in range(16):
            try:
                # CC 123 = All Notes Off, CC 120 = All Sound Off
                self.midi_output.write_short(0xB0 + int(ch), 123, 0)
                self.midi_output.write_short(0xB0 + int(ch), 120, 0)
            except Exception:
                pass

    def _midi_note_on(self, note_num: int, velocity: int = 96, channel: int = 0) -> bool:
        if not (self.use_midi_piano_output and self.midi_output_connected and self.midi_output is not None):
            return False
        try:
            n = int(note_num)
            v = int(max(0, min(127, int(velocity))))
            ch = int(max(0, min(15, int(channel))))
            self.midi_output.note_on(n, v, ch)
            return True
        except Exception:
            return False

    def _midi_note_off(self, note_num: int, channel: int = 0) -> bool:
        if not (self.use_midi_piano_output and self.midi_output_connected and self.midi_output is not None):
            return False
        try:
            n = int(note_num)
            ch = int(max(0, min(15, int(channel))))
            self.midi_output.note_off(n, 0, ch)
            return True
        except Exception:
            return False

    def _schedule_midi_note_off(self, note_num: int, delay_ms: int, channel: int = 0) -> None:
        try:
            now = int(pygame.time.get_ticks())
        except Exception:
            now = 0
        off_at = now + int(max(0, delay_ms))
        self._pending_midi_note_off.append((off_at, int(note_num), int(channel)))

    def _process_pending_midi_note_off(self) -> None:
        if not self._pending_midi_note_off:
            return
        try:
            now = int(pygame.time.get_ticks())
        except Exception:
            now = 0

        remaining = []
        for off_at, note_num, ch in self._pending_midi_note_off:
            if int(off_at) <= now:
                self._midi_note_off(int(note_num), int(ch))
            else:
                remaining.append((off_at, note_num, ch))
        self._pending_midi_note_off = remaining

    def refresh_midi_device_connection(self):
        """Check of de verbonden MIDI device nog steeds beschikbaar is."""
        self._init_midi_input_system()
        if not self.midi_input_connected or self.midi_input is None:
            self.midi_input_last_search = "Niet verbonden"
            return
        if not self.midi_input_supported or not self.midi_input_initialized:
            self.midi_input_last_search = "MIDI input niet beschikbaar"
            return

        # 1) Bestaat device_id nog?
        try:
            count = int(pgmidi.get_count())
        except Exception as e:
            self.midi_input_last_error = f"Refresh fout: {e}"
            self.midi_input_last_search = "Refresh mislukt"
            return

        did = self.midi_input_device_id
        if did is None or did < 0 or did >= count:
            self.midi_input_last_search = "Niet meer connected"
            self._close_midi_input()
            return

        try:
            info = pgmidi.get_device_info(int(did))
        except Exception as e:
            self.midi_input_last_error = f"Refresh info fout: {e}"
            self.midi_input_last_search = "Refresh mislukt"
            self._close_midi_input()
            return

        if not info:
            self.midi_input_last_search = "Niet meer connected"
            self._close_midi_input()
            return

        # Check of het nog steeds een input device is en dezelfde naam heeft
        try:
            _interface, name, is_input, _is_output, _opened = info
            if int(is_input) != 1:
                self.midi_input_last_search = "Device is geen input meer"
                self._close_midi_input()
                return
            
            # Verifieer dat het dezelfde device is door de naam te checken
            try:
                dev_name = name.decode(errors='ignore') if isinstance(name, (bytes, bytearray)) else str(name)
            except Exception:
                dev_name = str(name)
            
            if self.midi_input_device_name and dev_name != self.midi_input_device_name:
                self.midi_input_last_search = "Device naam veranderd"
                self._close_midi_input()
                return
        except Exception:
            self.midi_input_last_search = "Device info ongeldig"
            self._close_midi_input()
            return

        # RX-check: als er sinds de vorige refresh geen enkele RX is geweest, behandel als disconnect
        try:
            prev_rx = self.midi_rx_last_refresh_count
        except Exception:
            prev_rx = None
        current_rx = int(self.midi_rx_count)
        self.midi_rx_last_refresh_count = current_rx
        if prev_rx is not None and current_rx == int(prev_rx):
            try:
                if not self.midi_input.poll():
                    self.midi_input_last_search = "Geen MIDI input (RX stil)"
                    self._close_midi_input()
                    return
            except Exception as e:
                self.midi_input_last_error = f"Device verloren: {e}"
                self.midi_input_last_search = "Niet meer connected"
                self._close_midi_input()
                return

        # 2) Test of de handle nog echt werkt door actief te proberen te lezen
        try:
            # Probeer te pollen. Als het device is losgekoppeld, geeft dit meestal een error.
            _ = self.midi_input.poll()
            # Als poll() werkt, probeer ook de device info opnieuw op te halen als extra check
            recheck = pgmidi.get_device_info(int(did))
            if not recheck:
                self.midi_input_last_search = "Device verdwenen"
                self._close_midi_input()
                return
            
            self.midi_input_last_search = "Refresh OK"
            self.midi_input_last_error = ""
        except Exception as e:
            self.midi_input_last_error = f"Refresh poll fout: {e}"
            self.midi_input_last_search = "Niet meer connected"
            self._close_midi_input()
            return

    def disconnect_midi_device(self):
        """Manually disconnect from MIDI device."""
        self._close_midi_input()
        self.midi_input_last_search = "Verbroken"
        self.midi_input_last_error = ""

    def _midi_note_to_column(self, note_num: int) -> int:
        # Match de bestaande piano mapping: col 0..11 komt overeen met MIDI noten 48..59.
        base = 48
        n = int(note_num)
        if 0 <= (n - base) < int(NUM_COLUMNS):
            return int(n - base)
        return int(n % int(NUM_COLUMNS))

    def _midi_note_to_name(self, note_num: int) -> str:
        try:
            return get_midi_note_name(int(note_num))
        except Exception:
            return str(note_num)

    def _midi_press_column(self, column: int):
        if self.state != STATE_PLAYING:
            return
        col = int(column)
        if col < 0 or col >= int(NUM_COLUMNS):
            return
        # Voorkom repeat note_on events
        if col in self.pressed_columns:
            return
        self.pressed_columns.add(col)
        self._spawn_smoke_for_column(col, burst=True)
        # Speel altijd geluid op MIDI press (duur = zolang ingedrukt)
        self.start_sustain_sound(col)
        self.check_tile_hit(col)

    def _midi_release_column(self, column: int):
        col = int(column)
        self.pressed_columns.discard(col)

        # Stop sustain sound wanneer toets loslaat
        self.stop_sustain_sound(col, fade_ms=150)

        # Hold-note release scoring (zelfde als KEYUP)
        if self.state == STATE_PLAYING and col in self.active_holds and (not self.easy_hold_enabled):
            tile = self.active_holds.get(col)
            if tile is not None and (not tile.hit) and (not tile.missed):
                self.score_hold_on_release(tile, col)

    def _poll_midi_input(self):
        if not self.midi_input_connected or self.midi_input is None:
            return
        if not self.midi_input_supported or not self.midi_input_initialized:
            return

        try:
            # Lees meerdere events per frame voor snelle passages
            while self.midi_input.poll():
                events = self.midi_input.read(32)
                for data, _timestamp in events:
                    if not data or len(data) < 3:
                        continue
                    status = int(data[0]) & 0xF0
                    note = int(data[1])
                    vel = int(data[2])

                    # Diagnostics
                    self.midi_rx_count += 1
                    self.midi_last_note_num = int(note)
                    self.midi_last_velocity = int(vel)

                    # note_on (0x90) met vel>0 = press, note_off (0x80) of note_on vel==0 = release
                    if status == 0x90 and vel > 0:
                        col = self._midi_note_to_column(note)
                        # Update "Last:" alleen bij note_on (nieuwe toets indrukken)
                        self.midi_last_note_name = self._midi_note_to_name(note)
                        self._midi_press_column(col)
                    elif status == 0x80 or (status == 0x90 and vel == 0):
                        col = self._midi_note_to_column(note)
                        self._midi_release_column(col)
        except Exception as e:
            # Device is waarschijnlijk losgekoppeld
            self.midi_input_last_error = f"MIDI input fout: {e}"
            self.midi_input_last_search = "Device verloren"
            self._close_midi_input()

    def search_and_connect_midi_device(self):
        """Zoek een MIDI input device en verbind het met de game."""
        # Forceer re-init zodat nieuwe devices na game start zichtbaar worden
        try:
            if self.midi_input_initialized and pgmidi is not None:
                pgmidi.quit()
                self.midi_input_initialized = False
        except Exception:
            pass
        self._init_midi_input_system()
        if not self.midi_input_supported or not self.midi_input_initialized:
            self.midi_input_last_search = "MIDI input niet beschikbaar"
            return

        # Als er al een device verbonden is: niet opnieuw zoeken/verbinden (voorkomt errors).
        if self.midi_input_connected and self.midi_input is not None:
            self.midi_input_last_search = "Al verbonden"
            return

        # Sluit eventuele bestaande verbinding
        self._close_midi_input()

        try:
            count = int(pgmidi.get_count())
        except Exception as e:
            self.midi_input_last_search = "Zoeken mislukt"
            self.midi_input_last_error = f"Device count fout: {e}"
            return

        inputs = []  # list[(id:int, name:str)]
        try:
            for device_id in range(count):
                info = pgmidi.get_device_info(device_id)
                if not info:
                    continue
                _interface, name, is_input, _is_output, _opened = info
                if int(is_input) != 1:
                    continue
                try:
                    dev_name = name.decode(errors='ignore') if isinstance(name, (bytes, bytearray)) else str(name)
                except Exception:
                    dev_name = str(name)
                inputs.append((int(device_id), str(dev_name)))
        except Exception as e:
            self.midi_input_last_search = "Zoeken mislukt"
            self.midi_input_last_error = f"Device info fout: {e}"
            return

        if not inputs:
            self.midi_input_last_search = "Geen MIDI input device gevonden"
            self.midi_input_device_name = None
            self.midi_input_last_error = "Het herstarten van de game kan helpen"
            return

        # Heuristiek: kies liever een echte controller (Launchkey, AKAI, etc.)
        preferred_keywords = [
            "launchkey", "novation", "akai", "mpk", "arturia", "keystation",
            "keyboard", "controller", "midi"
        ]
        found_id, found_name = inputs[-1]
        for kw in preferred_keywords:
            for did, nm in inputs:
                if kw in nm.lower():
                    found_id, found_name = did, nm
                    break
            else:
                continue
            break

        self.midi_input_device_id = int(found_id)
        self.midi_input_device_name = str(found_name)

        # Reset diagnostics bij nieuwe connect
        self.midi_rx_count = 0
        self.midi_last_note_num = None
        self.midi_last_note_name = None
        self.midi_last_velocity = None

        try:
            # Bigger buffer helpt bij snelle input en voorkomt dropped events
            self.midi_input = pgmidi.Input(self.midi_input_device_id, buffer_size=4096)
            self.midi_input_connected = True
            self.midi_input_last_search = "Device gevonden en verbonden"
            self.midi_input_last_error = ""

            # Als MIDI input connected is: zet automatisch Note Names aan i.p.v. Keys.
            self.show_note_names = True
            self.midi_forced_note_names = True
            try:
                self.label_mode_button.text = "Note names"
            except Exception:
                pass
            self.update_bar_labels()
        except Exception as e:
            self.midi_input = None
            self.midi_input_connected = False
            self.midi_input_last_search = "Device gevonden, maar verbinden faalde"
            self.midi_input_last_error = f"Connect fout: {e}"

    def enter_game_over(self):
        # Wanneer je af gaat: stop alle geluiden direct.
        if self.state == STATE_GAME_OVER:
            return
        self.state = STATE_GAME_OVER
        self.miss_pending = False
        self.miss_timer = 0
        self.wrong_key_column = None
        
        # Stop metronoom
        self.stop_metronome()

        # Level rating niet tonen op Game Over
        self.level_stars = None
        self.level_note_count = 0
        self.level_score_thresholds = None

        # Snapshot + blur voor smooth game-over overlay
        self.game_over_frame = 0
        try:
            self.game_over_bg = self.screen.copy()
        except Exception:
            self.game_over_bg = None
        if self.game_over_bg is not None:
            self.game_over_bg_blur = self._blur_surface(self.game_over_bg)
        else:
            self.game_over_bg_blur = None

        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None

        # Clear holds/pressed, en stop alle mixer-kanalen.
        self.active_holds.clear()
        self.pressed_columns.clear()
        self.keys_currently_pressed.clear()
        pygame.mixer.stop()

        # Stop ook eventuele MIDI noten (system synth kan anders blijven hangen)
        self._pending_midi_note_off.clear()
        self._active_midi_hold_notes.clear()
        self._midi_all_notes_off()
        self.sustain_channel_by_column.clear()

    def start_sustain_sound(self, column, original_note_num=None):
        # Prefer: echte piano via MIDI output
        # (Momenteel uitgeschakeld - mixer is stabieler)
        note_list = None
        if original_note_num is None:
            note_list = [48 + int(column)]
        elif isinstance(original_note_num, (list, tuple, set)):
            note_list = [int(n) for n in original_note_num if n is not None]
        else:
            note_list = [int(original_note_num)]

        if self.use_midi_piano_output:
            played = []
            for midi_note in note_list:
                if self._midi_note_on(int(midi_note), velocity=92, channel=0):
                    played.append((int(midi_note), 0))
            if played:
                self._active_midi_hold_notes[int(column)] = played
                return

        # Mixer sustain: Genereer geluid met originele toonhoogte indien beschikbaar
        # Stop bestaande sustain op dit column
        prev_ch = self.sustain_channel_by_column.get(int(column))
        if prev_ch is not None:
            try:
                if isinstance(prev_ch, (list, tuple)):
                    for ch_item in prev_ch:
                        try:
                            ch_item.fadeout(60)
                        except Exception:
                            try:
                                ch_item.stop()
                            except Exception:
                                pass
                else:
                    prev_ch.fadeout(60)
            except Exception:
                try:
                    if isinstance(prev_ch, (list, tuple)):
                        for ch_item in prev_ch:
                            try:
                                ch_item.stop()
                            except Exception:
                                pass
                    else:
                        prev_ch.stop()
                except Exception:
                    pass

        # Gebruik een vrije channel zodat polyfonie mogelijk is
        try:
            ch = pygame.mixer.find_channel(True)
        except Exception:
            ch = None
        
        # Gebruik cache voor sustain sounds om lag spikes te voorkomen
        channels_used = []
        if original_note_num is not None:
            for midi_note in note_list:
                snd = self.midi_note_sustain_cache.get(int(midi_note))
                if snd is None:
                    freq = get_frequency_for_midi_note(int(midi_note))
                    snd = generate_piano_sustain_sound(freq, duration=SUSTAIN_SOUND_SECONDS, volume=0.18)
                    self.midi_note_sustain_cache[int(midi_note)] = snd
                if snd is None:
                    continue
                # Gebruik aparte channel per noot
                try:
                    note_ch = pygame.mixer.find_channel(True)
                except Exception:
                    note_ch = ch
                if note_ch is None:
                    continue
                try:
                    note_ch.set_volume(0.9)
                except Exception:
                    pass
                note_ch.play(snd, loops=0)
                channels_used.append(note_ch)
        else:
            snd = self.piano_sustain_sounds.get(column)
            if ch is None or snd is None:
                return
            try:
                ch.set_volume(0.9)
            except Exception:
                pass
            ch.play(snd, loops=0)
            channels_used.append(ch)

        if channels_used:
            self.sustain_channel_by_column[int(column)] = channels_used if len(channels_used) > 1 else channels_used[0]

    def stop_sustain_sound(self, column, fade_ms=110):
        # Stop MIDI hold noot als die actief is
        pair = self._active_midi_hold_notes.pop(int(column), None)
        if pair is not None:
            if isinstance(pair, list):
                for note_num, chn in pair:
                    self._midi_note_off(int(note_num), int(chn))
            else:
                note_num, chn = pair
                self._midi_note_off(int(note_num), int(chn))
            return

        # Mixer: fade-out in plaats van hard stoppen
        # Dit geeft het effect van een piano die geleidelijk uitdempt
        ch = self.sustain_channel_by_column.pop(int(column), None)
        if ch is None:
            return
        
        fade_ms = max(0, int(fade_ms))
        try:
            # Pygame mixer channel.fadeout() fade-out en stopt dan
            if isinstance(ch, (list, tuple)):
                for ch_item in ch:
                    try:
                        ch_item.fadeout(fade_ms)
                    except Exception:
                        try:
                            ch_item.stop()
                        except Exception:
                            pass
            else:
                ch.fadeout(fade_ms)
        except Exception:
            # Fallback: gewoon stoppen
            try:
                if isinstance(ch, (list, tuple)):
                    for ch_item in ch:
                        try:
                            ch_item.stop()
                        except Exception:
                            pass
                else:
                    ch.stop()
            except Exception:
                pass

    def get_label_for_column(self, column):
        # keys_carousel_current_index: 0=Keys, 1=Note names, 2=Straight keys
        if self.keys_carousel_current_index == 1:  # Note names
            return NOTE_LABELS.get(column, '?')
        elif self.keys_carousel_current_index == 2:  # Straight keys
            return STRAIGHT_KEY_LABELS.get(column, '?')
        else:  # Keys (default)
            return KEY_LABELS.get(column, '?')

    def update_bar_labels(self):
        self.bars = [Bar(i, self.get_bar_font_for_column(i), label_text=self.get_label_for_column(i)) for i in range(NUM_COLUMNS)]

    def get_bar_font_for_column(self, column):
        # In note-name modus zijn de labels langer, maak ze kleiner op zwarte toetsen
        if self.show_note_names and column in BLACK_INDICES:
            return self.font_bar_small
        return self.font_bar

    def get_tile_font_for_column(self, column):
        # In note-name modus op zwarte kolommen kleinere tekst gebruiken
        if self.show_note_names and column in BLACK_INDICES:
            return self.font_medium_small
        return self.font_medium
    
    def validate_midi(self, file_path):
        try:
            if not file_path.lower().endswith('.mid') and not file_path.lower().endswith('.midi'):
                return False
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'MThd':
                    return False
            return True
        except Exception as e:
            print(f"MIDI validatie fout: {e}")
            return False
    
    def analyze_midi(self, file_path):
        try:
            midi = MidiFile(file_path)
            tempo = 500000
            time_signature = (4, 4)  # Default (numerator, denominator)
            
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    elif msg.type == 'time_signature':
                        time_signature = (msg.numerator, msg.denominator)
            
            ticks_per_beat = midi.ticks_per_beat

            notes_data = []
            # Verzamel alle noten uit alle tracks tegelijk
            all_msgs = []
            for track_idx, track in enumerate(midi.tracks):
                current_time = 0
                for msg in track:
                    current_time += msg.time
                    if msg.type in ['note_on', 'note_off']:
                        all_msgs.append({
                            'type': msg.type,
                            'note': msg.note,
                            'velocity': getattr(msg, 'velocity', 0),
                            'time': current_time,
                            'track': track_idx
                        })
            
            # Sorteer alle messages op tijd
            all_msgs.sort(key=lambda x: (x['time'], x['track']))
            
            # Verwerk alle messages in één pass
            active = {}  # key = (note, track), value = start_time
            for msg in all_msgs:
                if msg['type'] == 'note_on' and msg['velocity'] > 0:
                    key = (msg['note'], msg['track'])
                    active[key] = msg['time']
                elif msg['type'] == 'note_off' or (msg['type'] == 'note_on' and msg['velocity'] == 0):
                    key = (msg['note'], msg['track'])
                    if key in active:
                        start = active.pop(key)
                        duration = msg['time'] - start
                        transposed = transpose_to_range(msg['note'])
                        # Controleer niet of deze noot al bestaat - voeg toe
                        notes_data.append({
                            'note': transposed,
                            'original_note': msg['note'],  # Behoud originele MIDI-noot voor audio
                            'start_tick': start,
                            'duration_ticks': duration
                        })
            
            # Combineer noten met dezelfde start_tick + transposed note.
            # Dit behoudt akkoorden (meerdere originele noten) in één kolom.
            unique = {}
            for n in notes_data:
                key = (n['start_tick'], n['note'])
                if key not in unique:
                    unique[key] = {
                        'note': n['note'],
                        'original_notes': [n.get('original_note', n['note'])],
                        'start_tick': n['start_tick'],
                        'duration_ticks': n['duration_ticks']
                    }
                else:
                    entry = unique[key]
                    orig = n.get('original_note', n['note'])
                    if orig not in entry['original_notes']:
                        entry['original_notes'].append(orig)
                    # Bewaar de langste duur voor de tile-lengte
                    entry['duration_ticks'] = max(entry.get('duration_ticks', 0), n['duration_ticks'])
            
            notes = sorted(unique.values(), key=lambda x: x['start_tick'])

            notes_frames = []
            removed_notes = []  # Track verwijderde noten voor debug output
            sec_per_tick = tempo / 1_000_000.0 / ticks_per_beat
            
            for n in notes:
                start_seconds = n['start_tick'] * sec_per_tick
                dur_seconds = n['duration_ticks'] * sec_per_tick
                
                # Verwijder extreem korte noten (<= 0.03s): te lastig om te spelen
                if dur_seconds <= 0.03:
                    removed_notes.append({
                        'reason': 'Too short (≤0.03s)',
                        'note': n['note'],
                        'original_notes': n.get('original_notes', []),
                        'start_tick': n['start_tick'],
                        'duration_seconds': dur_seconds
                    })
                    continue
                
                # Gebruik afronden i.p.v. floor, zodat korte afstanden tussen noten
                # minder vaak per ongeluk in dezelfde frame terechtkomen.
                start_frames = int(round(start_seconds * FPS))
                dur_frames = max(1, int(round(dur_seconds * FPS)))
                original_notes = n.get('original_notes')
                if not original_notes:
                    original_notes = [n.get('original_note', n['note'])]
                notes_frames.append({
                    'note': n['note'],
                    'original_notes': list(original_notes),  # Bewaar alle originele MIDI-noten
                    'start_frame': start_frames,
                    'duration_frames': dur_frames
                })

            # Als dezelfde noot opnieuw begint voordat de vorige is afgelopen,
            # verklein dan de duur van de eerste zodat hij net vóór de volgende eindigt.
            # Dit voorkomt overlappende holds bij herhaalde noten.
            gap_seconds = 0.1
            gap_frames = max(0, int(round(gap_seconds * FPS)))
            if gap_frames > 0 and notes_frames:
                indices_by_note = {}
                for i, nf in enumerate(notes_frames):
                    indices_by_note.setdefault(nf['note'], []).append(i)

                for _, idxs in indices_by_note.items():
                    idxs.sort(key=lambda i: notes_frames[i]['start_frame'])
                    for prev_i, next_i in zip(idxs, idxs[1:]):
                        prev = notes_frames[prev_i]
                        nxt = notes_frames[next_i]

                        prev_start = int(prev.get('start_frame', 0))
                        prev_dur = int(prev.get('duration_frames', 1))
                        next_start = int(nxt.get('start_frame', 0))

                        prev_end = prev_start + max(1, prev_dur)
                        cutoff = next_start - gap_frames

                        if prev_end > cutoff:
                            new_dur = max(1, cutoff - prev_start)
                            prev['duration_frames'] = new_dur
            
            # Verwijder duplicate octaven: als twee noten dezelfde toonsoort hebben maar
            # verschillende octaven en binnen 0.03s van elkaar vallen, behoud de hoogste
            filtered_notes_frames = []
            for i, nf in enumerate(notes_frames):
                original_notes = nf.get('original_notes', [])
                if not original_notes:
                    filtered_notes_frames.append(nf)
                    continue
                
                # Check of er andere noten zijn binnen 0.03s met dezelfde toonsoort
                start_time = nf['start_frame'] / float(FPS)
                time_window = 0.03  # 30ms
                
                # Groepeer noten per toonsoort (C, D, E, etc.)
                notes_by_pitch_class = {}
                for note_num in original_notes:
                    pitch_class = note_num % 12  # C=0, C#=1, D=2, etc.
                    if pitch_class not in notes_by_pitch_class:
                        notes_by_pitch_class[pitch_class] = []
                    notes_by_pitch_class[pitch_class].append(note_num)
                
                # Voor elke pitch class met meerdere octaven, behoud alleen de hoogste
                kept_notes = []
                for pitch_class, note_list in notes_by_pitch_class.items():
                    if len(note_list) > 1:
                        # Meerdere octaven gevonden, behoud de hoogste
                        highest = max(note_list)
                        kept_notes.append(highest)
                        # Track de verwijderde lagere octaven
                        for lower_note in note_list:
                            if lower_note != highest:
                                removed_notes.append({
                                    'reason': f'Duplicate octave (keep {get_midi_note_name(highest)})',
                                    'note': nf['note'],
                                    'original_notes': [lower_note],
                                    'start_frame': nf['start_frame'],
                                    'duration_frames': nf['duration_frames']
                                })
                    else:
                        kept_notes.extend(note_list)
                
                if kept_notes:
                    nf['original_notes'] = kept_notes
                    filtered_notes_frames.append(nf)
            
            notes_frames = filtered_notes_frames

            # Scan behouden noten opnieuw: als door aanpassingen een noot < 0.03s is,
            # voeg toe aan verwijderde noten en verwijder uit de lijst.
            rechecked_notes = []
            for nf in notes_frames:
                duration_frames = int(nf.get('duration_frames', 0))
                duration_seconds = duration_frames / float(FPS)
                if duration_seconds < 0.03:
                    removed_notes.append({
                        'reason': 'Too short after filtering (<0.03s)',
                        'note': nf.get('note'),
                        'original_notes': list(nf.get('original_notes', [])),
                        'start_frame': nf.get('start_frame', 0),
                        'duration_frames': duration_frames,
                        'duration_seconds': duration_seconds
                    })
                else:
                    rechecked_notes.append(nf)

            notes_frames = rechecked_notes

            # Bereken BPM uit tempo (tempo is in microseconds per beat)
            bpm = 60_000_000 / tempo
            
            # Sla analyse info op voor weergave in UI (niet printen naar console)
            self.midi_analysis_info = {
                'total_raw': len(notes_data),
                'after_duplicate': len(notes),
                'after_filter': len(notes_frames),
                'removed_count': len(removed_notes),
                'removed_notes': removed_notes,
                'sec_per_tick': sec_per_tick,
                'bpm': bpm,
                'time_signature': time_signature
            }

            self.midi_notes = notes_frames
            self.midi_tempo = tempo
            self.midi_tpb = ticks_per_beat
            
            # Pre-warm cache: genereer sustain sounds voor alle unieke noten in dit MIDI-bestand
            # Dit voorkomt lag spikes tijdens gameplay
            unique_notes = set()
            for n in notes_frames:
                for note_num in n.get('original_notes', []):
                    unique_notes.add(note_num)
            if unique_notes:
                # Cache laden (geen console output)
                for note_num in unique_notes:
                    if note_num not in self.midi_note_sustain_cache:
                        freq = get_frequency_for_midi_note(note_num)
                        self.midi_note_sustain_cache[note_num] = generate_piano_sustain_sound(freq, duration=SUSTAIN_SOUND_SECONDS, volume=0.18)
            
            return True
        except Exception as e:
            # Sla error op voor weergave in UI
            self.midi_analysis_info = {'error': str(e)}
            self.midi_notes = []
            return False

    def open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Selecteer MIDI bestand",
            filetypes=[("MIDI bestanden", "*.mid *.midi"), ("Alle bestanden", "*.*")]
        )
        if file_path:
            self.midi_file_path = file_path
            self.midi_valid = self.validate_midi(file_path)
            if self.midi_valid:
                print(f"MIDI bestand geldig: {os.path.basename(file_path)}")
                # Toon eerst "Loading"; analyse starten we in update (na 1 frame).
                self.midi_loading = True
                self.midi_analysis_pending = True
                self.midi_loading_frames = 0
            else:
                print(f"Geen geldig MIDI bestand: {os.path.basename(file_path)}")
                self.midi_loading = False
                self.midi_analysis_pending = False
                self.midi_loading_frames = 0
        root.destroy()

    def update_metronome(self):
        """Update metronoom timing en speel clicks op juiste momenten."""
        if not self.metronome_enabled:
            return
        
        # Metronoom uitschakelen wanneer cheats aan staan
        if self.cheats_enabled:
            return
        
        # Bereken frames tot volgende beat
        if self.metronome_frame_duration <= 0:
            # Volgende beat berekenen
            beats_per_minute = self.metronome_bpm
            seconds_per_beat = 60.0 / beats_per_minute
            self.metronome_frame_duration = int(seconds_per_beat * FPS)
        
        self.metronome_frame_counter += 1
        
        if self.metronome_frame_counter >= self.metronome_frame_duration:
            # Volgende beat
            self.metronome_frame_counter = 0
            
            # Bepaal welke click
            numerator = self.metronome_time_signature[0]
            is_first_beat = (self.metronome_current_beat % numerator == 0)
            
            # Speel juiste geluid
            click_sound = self.metronome_click_high if is_first_beat else self.metronome_click_low
            try:
                ch = pygame.mixer.find_channel(True)
                if ch:
                    ch.play(click_sound)
            except Exception:
                pass
            
            # Update beat
            self.metronome_current_beat += 1
            if self.metronome_current_beat >= numerator:
                self.metronome_current_beat = 0
    
    def start_metronome(self, bpm, time_signature=(4, 4)):
        """Start metronoom met gegeven BPM en maatsoort."""
        self.metronome_bpm = bpm
        self.metronome_time_signature = time_signature
        self.metronome_current_beat = 0
        self.metronome_frame_counter = 0
        self.metronome_frame_duration = 0
        self.metronome_enabled = True
    
    def stop_metronome(self):
        """Stop metronoom."""
        self.metronome_enabled = False
        self.metronome_current_beat = 0
        self.metronome_frame_counter = 0

    def start_level_1(self):
        file_path = self.level1_midi_path
        if (not file_path) or (not os.path.exists(file_path)):
            return

        self.current_level_id = "level1"

        self.midi_file_path = file_path
        self.midi_valid = self.validate_midi(file_path)
        if not self.midi_valid:
            return

        # Start direct: geen aparte MIDI-info / 'Check Code' stap.
        self.analyze_midi(file_path)
        self.start_game_from_midi()

    def start_level_2(self):
        file_path = self.level2_midi_path
        if (not file_path) or (not os.path.exists(file_path)):
            return

        self.current_level_id = "level2"

        self.midi_file_path = file_path
        self.midi_valid = self.validate_midi(file_path)
        if not self.midi_valid:
            return

        # Start direct: geen aparte MIDI-info / 'Check Code' stap.
        self.analyze_midi(file_path)
        self.start_game_from_midi()
    
    def clear_uploaded_midi(self):
        """Reset de geüploade MIDI file state wanneer je teruggaat naar levels/menu."""
        self.midi_file_path = None
        self.midi_valid = False
        self.midi_notes = []
        self.midi_loading = False
        self.midi_analysis_pending = False
        self.midi_loading_frames = 0
    
    def start_game(self):
        self.state = STATE_PLAYING
        self.score = 0
        self.spawn_counter = 0
        self.tiles = []
        self.miss_pending = False
        self.miss_timer = 0
        self.wrong_key_column = None
        self.game_over_frame = 0
        self.game_over_bg = None
        self.game_over_bg_blur = None
        self.use_midi_schedule = False
        self.scheduled_notes = []
        self.scheduled_notes_original = []
        self.game_frame = 0
        self.play_time = 0
        self.pressed_columns.clear()
        self.keys_currently_pressed.clear()
        self.active_holds.clear()
        self.played_notes_history = []
        self.celebration_tiles = []
        self.level_complete_frame = 0
        self.level_complete_bg = None
        self.level_complete_bg_blur = None
        self.auto_replay_done = False
        self.auto_replay_time = 0.0
        self.auto_replay_frame = 0
        self.auto_replay_bg = None
        self.auto_replay_bg_blur = None
        self.level_complete_from_auto_replay = False

        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None
        self.tutorial_mode = False
        self.tutorial_complete_bg = None
        self.tutorial_complete_bg_blur = None
        self.tutorial_complete_frame = 0
        self.smoke_particles.clear()
        self.level_stars = None
        self.level_note_count = 0
        self.level_score_thresholds = None
        self.current_level_id = None
        self.current_level_id = None
        # Stop eventuele sustain loops
        for col in range(NUM_COLUMNS):
            self.stop_sustain_sound(col, fade_ms=0)

    def start_tutorial(self):
        # Ga eerst naar tutorial welcome screen
        self.state = STATE_TUTORIAL_WELCOME
        self.tutorial_welcome_frame = 0
        self.tutorial_mode = True
        
    def _start_tutorial_gameplay(self):
        # Echte tutorial gameplay (na welcome screen)
        self.state = STATE_PLAYING
        self.score = 0
        self.spawn_counter = 0
        self.tiles = []
        self.miss_pending = False
        self.miss_timer = 0
        self.wrong_key_column = None
        self.game_over_frame = 0
        self.game_over_bg = None
        self.game_over_bg_blur = None
        self.use_midi_schedule = True
        self.scheduled_notes = []
        self.scheduled_notes_original = []
        self.game_frame = 0
        self.play_time = 0
        self.pressed_columns.clear()
        self.keys_currently_pressed.clear()
        self.active_holds.clear()
        self.played_notes_history = []

        # Geen auto replay / level complete flow in tutorial
        self.auto_replay_done = True
        self.auto_replay_time = 0.0
        self.auto_replay_frame = 0
        self.auto_replay_bg = None
        self.auto_replay_bg_blur = None
        self.level_complete_from_auto_replay = False

        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None

        self.tutorial_mode = True
        self.tutorial_slowmo_active = False
        self.tutorial_slowmo_done = False
        self.tutorial_overlay_active = False
        self.tutorial_overlay_step = 1
        self.tutorial_overlay_start_frame = None
        self.tutorial_complete_bg = None
        self.tutorial_complete_bg_blur = None
        self.tutorial_complete_frame = 0
        self.smoke_particles.clear()
        self.level_stars = None
        self.level_note_count = 0
        self.level_score_thresholds = None

        # Stop sustain loops
        for col in range(NUM_COLUMNS):
            self.stop_sustain_sound(col, fade_ms=0)

        # Build a simple scripted schedule (hit times in frames)
        # Tutorial: alleen C-E-G akkoord (kolommen 0, 4, 7)
        # Alle noten moeten in beeld zijn vóór de "This is your piano" message (na 1s)
        travel_frames = int(BAR_Y / TILE_SPEED)
        
        # Spawn alle noten vroeg zodat ze zichtbaar zijn bij de tutorial start
        scheduled = [
            {'spawn_frame': 0, 'column': 0, 'length_pixels': None, 'note': None},   # C
            {'spawn_frame': 15, 'column': 4, 'length_pixels': None, 'note': None},  # E
            {'spawn_frame': 30, 'column': 7, 'length_pixels': None, 'note': None},  # G
        ]
        
        scheduled.sort(key=lambda x: x['spawn_frame'])
        self.scheduled_notes = scheduled
        self.scheduled_notes_original = [dict(s) for s in scheduled]

    def retry_level(self):
        # Start dezelfde mode opnieuw (random of MIDI) zonder terug naar menu.
        if self.use_midi_schedule:
            if self.scheduled_notes_original:
                self.scheduled_notes = [dict(s) for s in self.scheduled_notes_original]
                self.state = STATE_PLAYING
                self.score = 0
                self.spawn_counter = 0
                self.tiles = []
                self.miss_pending = False
                self.miss_timer = 0
                self.wrong_key_column = None
                self.game_over_frame = 0
                self.game_over_bg = None
                self.game_over_bg_blur = None
                self.game_frame = 0
                self.play_time = 0
                self.pressed_columns.clear()
                self.keys_currently_pressed.clear()
                self.active_holds.clear()
                self.played_notes_history = []
                self.celebration_tiles = []
                self.level_complete_frame = 0
                self.level_complete_bg = None
                self.level_complete_bg_blur = None
                self.auto_replay_done = False
                self.auto_replay_time = 0.0
                self.auto_replay_frame = 0
                self.auto_replay_bg = None
                self.auto_replay_bg_blur = None
                self.level_complete_from_auto_replay = False

                self.last_judgement = ""
                self.last_judgement_timer = 0
                self.judgement_history.clear()
                self.combo_streak = 0
                self.combo_kind = None

                self.level_stars = None
                self.level_note_count = 0
                self.level_score_thresholds = None
                return
            return self.start_game_from_midi()

        return self.start_game()

    def start_game_from_midi(self):
        if not self.midi_notes:
            return self.start_game()

        # Bepaal level-id voor rating (Level 1 of custom midi)
        try:
            if getattr(self, 'midi_file_path', None) and os.path.abspath(self.midi_file_path) == os.path.abspath(self.level1_midi_path):
                self.current_level_id = "level1"
            elif getattr(self, 'midi_file_path', None) and os.path.abspath(self.midi_file_path) == os.path.abspath(self.level2_midi_path):
                self.current_level_id = "level2"
            elif getattr(self, 'midi_file_path', None):
                self.current_level_id = "midi:" + os.path.abspath(self.midi_file_path).lower()
            else:
                self.current_level_id = None
        except Exception:
            self.current_level_id = None

        travel_frames = int(BAR_Y / TILE_SPEED)
        # Bouw een schema op en dedupliceer exact gelijke spawns per kolom
        scheduled_map = {}
        for n in self.midi_notes:
            note = n['note']
            original_notes = n.get('original_notes') or [n.get('original_note', note)]
            start_frame = n['start_frame']
            duration_frames = n.get('duration_frames', 1)
            col = note - 36
            if col < 0 or col >= NUM_COLUMNS:
                continue
            spawn_frame = start_frame - travel_frames

            # Maak de tile-lengte gebaseerd op de echte nootduur:
            # duur (frames) * TILE_SPEED (px/frame) => px.
            # Minimum zodat korte noten zichtbaar blijven.
            min_len = 30
            max_len = SCREEN_HEIGHT * 3
            length_pixels = int(round(duration_frames * TILE_SPEED))
            length_pixels = max(min_len, min(max_len, length_pixels))

            key = (spawn_frame, col)
            existing = scheduled_map.get(key)
            # Als er al een entry is op exact zelfde tijd/kolom, bewaar de langste tile
            if existing is None or length_pixels > existing['length_pixels']:
                scheduled_map[key] = {
                    'spawn_frame': spawn_frame,
                    'column': col,
                    'length_pixels': length_pixels,
                    'note': note,
                    'original_notes': list(original_notes)  # Bewaar originele voor audio
                }
            elif existing is not None:
                # Merge eventuele extra originele noten (akkoorden)
                existing_list = existing.get('original_notes', [])
                for on in original_notes:
                    if on not in existing_list:
                        existing_list.append(on)
                existing['original_notes'] = existing_list

        scheduled = sorted(scheduled_map.values(), key=lambda x: x['spawn_frame'])

        # Als de eerste noten te vroeg starten t.o.v. de valtijd, worden spawn_frames negatief.
        # Zonder correctie zouden meerdere vroege noten tegelijk verschijnen bij game start.
        if scheduled:
            min_spawn = min(s['spawn_frame'] for s in scheduled)
            if min_spawn < 0:
                shift = -min_spawn
                for s in scheduled:
                    s['spawn_frame'] += shift

        self.scheduled_notes = scheduled
        # Bewaar een kopie zodat we instant kunnen retryn (scheduled_notes wordt leeg gepopt).
        self.scheduled_notes_original = [dict(s) for s in scheduled]
        self.use_midi_schedule = True
        
        # Start metronoom met BPM en maatsoort uit MIDI
        analysis = getattr(self, 'midi_analysis_info', {})
        bpm = analysis.get('bpm', 120.0)
        time_signature = analysis.get('time_signature', (4, 4))
        self.start_metronome(bpm, time_signature)
        
        self.state = STATE_PLAYING
        self.score = 0
        self.spawn_counter = 0
        self.tiles = []
        self.miss_pending = False
        self.miss_timer = 0
        self.game_frame = 0
        self.play_time = 0
        self.pressed_columns.clear()
        self.keys_currently_pressed.clear()
        self.active_holds.clear()
        self.played_notes_history = []
        self.celebration_tiles = []
        self.level_complete_frame = 0
        self.level_complete_bg = None
        self.level_complete_bg_blur = None
        self.auto_replay_done = False
        self.auto_replay_time = 0.0
        self.auto_replay_frame = 0
        self.auto_replay_bg = None
        self.auto_replay_bg_blur = None
        self.level_complete_from_auto_replay = False

        # Voorkom dat judgement van vorige run nog even zichtbaar is.
        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None

    def start_auto_replay(self):
        if not self.scheduled_notes_original:
            self.start_level_completed()
            return

        # Reset en speel dezelfde schedule opnieuw af, zonder input/fail.
        self.state = STATE_AUTO_REPLAY
        self.auto_replay_done = True
        self.auto_replay_speed_mult = 2.0
        self.auto_replay_time = 0.0
        self.auto_replay_frame = 0

        # Precompute blur background zodat we niet elke frame hoeven te blurren.
        self.auto_replay_bg = self._render_playfield_background()
        self.auto_replay_bg_blur = self._blur_surface(self.auto_replay_bg)

        self.use_midi_schedule = True
        self.scheduled_notes = [dict(s) for s in self.scheduled_notes_original]
        self.tiles = []
        self.active_holds.clear()
        self.pressed_columns.clear()
        self.miss_pending = False
        self.miss_timer = 0
        self.game_frame = 0

        # Geen judgement overnemen van vorige run.
        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None

        # Bereken stars op basis van de score van de net gespeelde run.
        self._compute_level_star_rating()

        # Stop alle sustain loops
        for col in range(NUM_COLUMNS):
            self.stop_sustain_sound(col, fade_ms=0)

    def _record_played_tile(self, tile):
        # Bewaar een snapshot voor het level-complete effect.
        try:
            self.played_notes_history.append({
                'column': int(tile.column),
                'height': int(getattr(tile, 'height', 0)),
                'y': float(getattr(tile, 'y', 0.0)),
                'note': getattr(tile, 'note_num', None)
            })
        except Exception:
            return

    def _render_playfield_background(self) -> pygame.Surface:
        """Render alleen de playfield achtergrond (geen tiles/UI) voor overlays."""
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)).convert()
        surf.blit(self.background, (0, 0))

        # Bars in dezelfde volgorde als tijdens spelen (wit -> zwart).
        for bar in self.bars:
            if bar.column not in BLACK_INDICES:
                bar.draw(surf)
        for bar in self.bars:
            if bar.column in BLACK_INDICES:
                bar.draw(surf)

        return surf

    def start_level_completed(self):
        # Maak een snapshot ZONDER tiles, zodat je geen split-second gele noten ziet.
        # Als we uit AUTO_REPLAY komen, hergebruik dan de auto-replay blur zodat de blur
        # niet ineens "uit" gaat bij de overgang.
        self.level_complete_from_auto_replay = (self.state == STATE_AUTO_REPLAY)
        panel_drop_frames = max(1, int(FPS * 0.6))
        # Komt level-complete uit auto replay, dan staat het paneel al in beeld.
        # Start dus direct op de eindpositie zodat het niet nog een keer "valt".
        self.level_complete_frame = int(panel_drop_frames) if self.level_complete_from_auto_replay else 0
        
        # Stop metronoom
        self.stop_metronome()

        if self.level_complete_from_auto_replay and self.auto_replay_bg is not None:
            self.level_complete_bg = self.auto_replay_bg
            self.level_complete_bg_blur = self.auto_replay_bg_blur
            # Auto replay had al "alle noten" effect; geen extra celebration-tiles meer.
            self.celebration_tiles = []
        else:
            self.level_complete_bg = self._render_playfield_background()
            self.level_complete_bg_blur = self._blur_surface(self.level_complete_bg)

        # Clear judgement/combo zodat dit niet blijft hangen bij volgend level.
        self.last_judgement = ""
        self.last_judgement_timer = 0
        self.judgement_history.clear()
        self.combo_streak = 0
        self.combo_kind = None

        # Bereken stars voor dit level (score is al definitief bij completion).
        self._compute_level_star_rating()

        # Maak "vallende" tiles van gespeelde noten (geen input nodig).
        # Alleen wanneer we NIET uit auto replay komen.
        if not self.level_complete_from_auto_replay:
            self.celebration_tiles = []
            font = self.get_tile_font_for_column(0)
            for rec in self.played_notes_history[-250:]:  # cap voor performance
                col = int(rec.get('column', 0))
                h = int(rec.get('height', 0))
                if h <= 0:
                    continue
                t = Tile(col, length_pixels=h, note_num=rec.get('note'), font=font, label_text=self.get_label_for_column(col))
                t.y = float(rec.get('y', -h))
                t.speed = TILE_SPEED * 12
                t.hit = False
                t.missed = False
                t.holding = False
                t.latched = False
                self.celebration_tiles.append(t)

        self.state = STATE_LEVEL_COMPLETED
    
    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Tijdens overgang: alleen quit afhandelen.
            if self.state == STATE_TRANSITION:
                continue
            
            if event.type == pygame.KEYDOWN:
                # Game volledig afsluiten (nieuwe quit key)
                if event.key == pygame.K_DELETE:
                    self.running = False
                    continue

                if event.key == pygame.K_r and (self.state == STATE_PLAYING or self.state == STATE_GAME_OVER):
                    self.retry_level()
                    continue

                if event.key == pygame.K_ESCAPE:
                    if self.state == STATE_LEVELS:
                        self.clear_uploaded_midi()
                        self.start_transition(STATE_MENU, direction='right')
                    elif self.state == STATE_SETTINGS:
                        if getattr(self, 'settings_return_state', STATE_MENU) == STATE_PAUSED:
                            self.settings_return_state = STATE_MENU
                            self.state = STATE_PAUSED
                        else:
                            self.start_transition(STATE_MENU, direction='down')
                    elif self.state == STATE_SHOP:
                        # Als je in instruments bent, start reverse animatie
                        if self.shop_tab == "instruments":
                            self.shop_swipe_frame = self.shop_swipe_duration  # Start van einde
                            self.shop_swipe_direction = -1  # Reverse
                        else:
                            # Pas dan ga je naar menu
                            self.state = STATE_MENU
                    elif self.state == STATE_MIDI_INFO:
                        self.state = STATE_LEVELS
                    elif self.state == STATE_PLAYING:
                        self.enter_pause()
                    elif self.state == STATE_PAUSED:
                        self.start_countdown()
                    elif self.state == STATE_COUNTDOWN:
                        self.enter_pause()
                    elif self.state == STATE_GAME_OVER:
                        self.return_to_main_from_game()
                    else:
                        # ESC sluit de game niet meer af; gebruik DELETE om te quitten.
                        pass
                
                if self.state == STATE_MENU and event.key == pygame.K_SPACE:
                    self.state = STATE_PLAY_SELECT

                if self.state == STATE_PLAY_SELECT:
                    if event.key == pygame.K_ESCAPE:
                        self.state = STATE_MENU
                
                if self.state == STATE_PLAYING:
                    # Game key handling gebeurt volledig via polling in update()
                    # om 4+ simultane toetsen te ondersteunen zonder conflicten
                    if event.key == pygame.K_ESCAPE and self.tutorial_mode:
                        self.state = STATE_PLAY_SELECT

                if self.state == STATE_AUTO_REPLAY:
                    # Geen input nodig; allow skip/continue.
                    # SPACE = meteen door (zoals de hint), ESC = ook skip.
                    if event.key == pygame.K_SPACE:
                        self.clear_uploaded_midi()
                        self.state = STATE_LEVELS
                    elif event.key == pygame.K_ESCAPE:
                        self.clear_uploaded_midi()
                        self.state = STATE_LEVELS

                if self.state == STATE_LEVEL_COMPLETED:
                    # Geen input nodig; allow exit.
                    if event.key == pygame.K_SPACE:
                        self.clear_uploaded_midi()
                        self.state = STATE_LEVELS
                    elif event.key == pygame.K_ESCAPE:
                        self.clear_uploaded_midi()
                        self.state = STATE_LEVELS
                
                if self.state == STATE_GAME_OVER and event.key == pygame.K_SPACE:
                    self.clear_uploaded_midi()
                    self.state = STATE_MENU

                if self.state == STATE_TUTORIAL_COMPLETED:
                    if event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                        self.state = STATE_PLAY_SELECT

                if self.state == STATE_TUTORIAL_WELCOME:
                    if event.key == pygame.K_SPACE:
                        self._start_tutorial_gameplay()

                if self.state == STATE_PLAYING and self.tutorial_mode and self.tutorial_overlay_active:
                    if event.key == pygame.K_SPACE:
                        if self.tutorial_overlay_step == 1:
                            self.tutorial_overlay_step = 2
                            self.tutorial_overlay_start_frame = int(self.play_time)
                        else:
                            self.tutorial_overlay_active = False
                            self.tutorial_slowmo_active = False
                            self.tutorial_slowmo_done = True
                            self.tutorial_slowmo_start_frame = None
                            self.tutorial_overlay_start_frame = None
                        continue

            if event.type == pygame.KEYUP:
                # Key release handling gebeurt volledig via polling in update()
                # om conflicten te voorkomen bij 4+ simultane toetsen
                pass

            if event.type == pygame.MOUSEWHEEL:
                if self.state == STATE_MIDI_INFO and getattr(self, 'midi_info_show_all', False):
                    try:
                        delta = int(event.y) * 30
                    except Exception:
                        delta = 0
                    self.midi_info_scroll = max(0, min(
                        self.midi_info_scroll - delta,
                        int(getattr(self, 'midi_info_max_scroll', 0))
                    ))

            if event.type == pygame.MOUSEBUTTONUP:
                if self.state == STATE_MIDI_INFO:
                    self.midi_info_scroll_dragging = False

            if event.type == pygame.MOUSEMOTION:
                if self.state == STATE_MIDI_INFO and getattr(self, 'midi_info_show_all', False):
                    if getattr(self, 'midi_info_scroll_dragging', False):
                        track_rect = getattr(self, 'midi_info_scrollbar_rect', None)
                        handle_rect = getattr(self, 'midi_info_scrollbar_handle_rect', None)
                        max_scroll = int(getattr(self, 'midi_info_max_scroll', 0))
                        if track_rect and handle_rect and max_scroll > 0:
                            handle_h = handle_rect.height
                            track_h = track_rect.height
                            usable = max(1, track_h - handle_h)
                            new_handle_y = event.pos[1] - int(getattr(self, 'midi_info_scrollbar_drag_offset', 0))
                            new_handle_y = max(track_rect.y, min(track_rect.y + usable, new_handle_y))
                            ratio = float(new_handle_y - track_rect.y) / float(usable)
                            self.midi_info_scroll = int(round(ratio * max_scroll))
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Belangrijk: behandel maar één state per klik-event.
                # Anders kan een klik die een state opent (bv. Settings) meteen ook
                # onderliggende knoppen in de nieuwe state triggeren.
                if self.state == STATE_MENU:
                    if self.play_button.is_clicked(mouse_pos):
                        self.state = STATE_PLAY_SELECT
                    elif self.levels_button.is_clicked(mouse_pos):
                        self.start_transition(STATE_LEVELS, direction='left')
                    elif self.shop_button.is_clicked(mouse_pos):
                        self.state = STATE_SHOP
                    elif self.settings_button.is_clicked(mouse_pos):
                        self.start_transition(STATE_SETTINGS, direction='up')

                elif self.state == STATE_PLAY_SELECT:
                    if self.free_play_button.is_clicked(mouse_pos):
                        self.start_game()
                    elif self.tutorial_button.is_clicked(mouse_pos):
                        self.start_tutorial()
                
                elif self.state == STATE_LEVELS:
                    if self.level1_button.is_clicked(mouse_pos):
                        self.start_level_1()
                    elif self.level2_button.is_clicked(mouse_pos):
                        self.start_level_2()
                    elif self.upload_button.is_clicked(mouse_pos):
                        self.open_file_dialog()
                    elif self.midi_valid and self.start_button.is_clicked(mouse_pos):
                        self.midi_info_show_all = False
                        self.midi_info_scroll = 0
                        self.state = STATE_MIDI_INFO
                
                elif self.state == STATE_MIDI_INFO:
                    if self.midi_valid and self.play_now_button.is_clicked(mouse_pos):
                        self.start_game_from_midi()
                    else:
                        rect_show_less = getattr(self, 'midi_info_show_less_rect', None)
                        if rect_show_less and rect_show_less.collidepoint(mouse_pos):
                            self.midi_info_show_all = False
                            self.midi_info_scroll = 0
                            self.midi_info_scroll_dragging = False
                            return
                        rect_removed = getattr(self, 'midi_info_more_rect_removed', None)
                        rect_kept = getattr(self, 'midi_info_more_rect_kept', None)
                        if rect_removed and rect_removed.collidepoint(mouse_pos):
                            self.midi_info_show_all = True
                        if rect_kept and rect_kept.collidepoint(mouse_pos):
                            self.midi_info_show_all = True
                        if getattr(self, 'midi_info_show_all', False):
                            handle_rect = getattr(self, 'midi_info_scrollbar_handle_rect', None)
                            track_rect = getattr(self, 'midi_info_scrollbar_rect', None)
                            max_scroll = int(getattr(self, 'midi_info_max_scroll', 0))
                            if handle_rect and handle_rect.collidepoint(mouse_pos):
                                self.midi_info_scroll_dragging = True
                                self.midi_info_scrollbar_drag_offset = mouse_pos[1] - handle_rect.y
                            elif track_rect and track_rect.collidepoint(mouse_pos) and max_scroll > 0:
                                handle_h = handle_rect.height if handle_rect else 30
                                usable = max(1, track_rect.height - handle_h)
                                new_handle_y = mouse_pos[1] - int(handle_h / 2)
                                new_handle_y = max(track_rect.y, min(track_rect.y + usable, new_handle_y))
                                ratio = float(new_handle_y - track_rect.y) / float(usable)
                                self.midi_info_scroll = int(round(ratio * max_scroll))

                elif self.state == STATE_SETTINGS:
                    if self.settings_gameplay_tab_button.is_clicked(mouse_pos):
                        self.settings_tab = "gameplay"
                        continue
                    if self.settings_visuals_tab_button.is_clicked(mouse_pos):
                        self.settings_tab = "visuals"
                        continue
                    if self.settings_audio_tab_button.is_clicked(mouse_pos):
                        self.settings_tab = "audio"
                        continue
                    if self.settings_midi_tab_button.is_clicked(mouse_pos):
                        self.settings_tab = "midi"
                        continue

                    if self.settings_tab == "midi":
                        # Enable/disable knoppen op basis van connectie
                        try:
                            self.midi_search_device_button.enabled = (not bool(self.midi_input_connected))
                        except Exception:
                            self.midi_search_device_button.enabled = True
                        try:
                            self.midi_test_device_button.enabled = bool(self.midi_input_connected)
                        except Exception:
                            self.midi_test_device_button.enabled = False
                        try:
                            self.midi_refresh_device_button.enabled = bool(self.midi_input_connected)
                        except Exception:
                            self.midi_refresh_device_button.enabled = False
                        try:
                            self.midi_disconnect_device_button.enabled = bool(self.midi_input_connected)
                        except Exception:
                            self.midi_disconnect_device_button.enabled = False

                        if self.midi_search_device_button.is_clicked(mouse_pos):
                            self.search_and_connect_midi_device()
                            continue

                        if self.midi_refresh_device_button.is_clicked(mouse_pos):
                            self.refresh_midi_device_connection()
                            continue

                        if self.midi_disconnect_device_button.is_clicked(mouse_pos):
                            self.disconnect_midi_device()
                            continue

                        if self.midi_test_device_button.is_clicked(mouse_pos):
                            # Nog leeg zoals gevraagd (placeholder)
                            continue

                    # Keys carousel pijltjes (alleen in Gameplay tab)
                    if self.settings_tab == "gameplay":
                        if self.keys_carousel_left_arrow.is_clicked(mouse_pos):
                            # Naar links: vorige optie, animatie van links
                            self.keys_carousel_previous_index = self.keys_carousel_current_index
                            self.keys_carousel_current_index = (self.keys_carousel_current_index - 1) % len(self.keys_carousel_options)
                            self.keys_carousel_slide_frames = 0
                            self.keys_carousel_slide_direction = 1  # Van links
                            self.show_note_names = (self.keys_carousel_current_index == 1)
                            self.update_bar_labels()
                            continue

                        if self.keys_carousel_right_arrow.is_clicked(mouse_pos):
                            # Naar rechts: volgende optie, animatie van rechts
                            self.keys_carousel_previous_index = self.keys_carousel_current_index
                            self.keys_carousel_current_index = (self.keys_carousel_current_index + 1) % len(self.keys_carousel_options)
                            self.keys_carousel_slide_frames = 0
                            self.keys_carousel_slide_direction = -1  # Van rechts
                            self.show_note_names = (self.keys_carousel_current_index == 1)
                            self.update_bar_labels()
                            continue

                    if self.label_mode_button.is_clicked(mouse_pos):
                        self.show_note_names = not self.show_note_names
                        self.label_mode_button.text = "Note names" if self.show_note_names else "Keys"
                        self.update_bar_labels()

                    elif self.cheats_button.is_clicked(mouse_pos):
                        self.cheats_enabled = not self.cheats_enabled
                        self.cheats_button.text = "Cheats: ON" if self.cheats_enabled else "Cheats: OFF"

                    elif self.easy_hold_button.is_clicked(mouse_pos):
                        self.easy_hold_enabled = not self.easy_hold_enabled
                        self.easy_hold_button.text = "Easy: ON" if self.easy_hold_enabled else "Easy: OFF"

                    elif self.green_note_button.is_clicked(mouse_pos):
                        self.green_note_enabled = not self.green_note_enabled
                        self.green_note_button.text = "Blue hit: ON" if self.green_note_enabled else "Blue hit: OFF"

                    elif self.cheat_sustain_button.is_clicked(mouse_pos):
                        self.cheat_auto_sustain_enabled = not self.cheat_auto_sustain_enabled
                        self.cheat_sustain_button.text = "Cheat sustain: ON" if self.cheat_auto_sustain_enabled else "Cheat sustain: OFF"

                elif self.state == STATE_SHOP:
                    # Themes en Background buttons worden genegeerd wanneer instruments tabje actief is
                    if self.shop_tab != "instruments":
                        if self.shop_themes_button.is_clicked(mouse_pos):
                            self.shop_tab = "themes"
                            self.shop_swipe_frame = 0
                            self.shop_swipe_direction = 0
                        elif self.shop_background_button.is_clicked(mouse_pos):
                            self.shop_tab = "background"
                            self.shop_swipe_frame = 0
                            self.shop_swipe_direction = 0
                    
                    if self.shop_tab != "instruments" and self.shop_instruments_button.is_clicked(mouse_pos):
                        self.shop_tab = "instruments"
                        self.shop_swipe_frame = 1  # Start animatie
                        self.shop_swipe_direction = 1  # Forward
                    
                    # Click detection voor instrument vakjes (3x3 grid)
                    if self.shop_tab == "instruments":
                        # Bereken grid parameters (moet hetzelfde zijn als in render)
                        grid_size = 3
                        base_square_size = 110
                        gap = 15
                        
                        swipe_progress = min(1.0, float(self.shop_swipe_frame) / float(self.shop_swipe_duration)) if self.shop_swipe_direction == 1 else 0.0
                        min_scale = 0.3
                        current_scale = min_scale + (1.0 - min_scale) * swipe_progress
                        square_size = int(base_square_size * current_scale)
                        
                        grid_width = grid_size * square_size + (grid_size - 1) * gap
                        grid_height = grid_size * square_size + (grid_size - 1) * gap
                        
                        horizontal_offset = 0
                        if self.shop_swipe_direction == -1:
                            reverse_progress = 1.0 - swipe_progress
                            horizontal_offset = int(reverse_progress * 300)
                        
                        start_x = (SCREEN_WIDTH - grid_width) // 2 + horizontal_offset
                        start_y = (SCREEN_HEIGHT - grid_height) // 2
                        
                        # Check welk vakje geklikt is
                        for row in range(grid_size):
                            for col in range(grid_size):
                                x = start_x + col * (square_size + gap)
                                y = start_y + row * (square_size + gap)
                                rect = pygame.Rect(x, y, square_size, square_size)
                                
                                if rect.collidepoint(mouse_pos):
                                    self.shop_selected_instrument = (row, col)

                elif self.state == STATE_PAUSED:
                    if self.pause_continue_button.is_clicked(mouse_pos):
                        self.start_countdown()
                    elif self.pause_settings_button.is_clicked(mouse_pos):
                        self.settings_return_state = STATE_PAUSED
                        self.state = STATE_SETTINGS
                    elif self.pause_return_button.is_clicked(mouse_pos):
                        self.return_to_main_from_game()
        
        if self.state == STATE_MENU:
            self.play_button.update_hover(mouse_pos)
            self.levels_button.update_hover(mouse_pos)
            self.settings_button.update_hover(mouse_pos)
            self.shop_button.update_hover(mouse_pos)

        if self.state == STATE_PLAY_SELECT:
            self.free_play_button.update_hover(mouse_pos)
            self.tutorial_button.update_hover(mouse_pos)
        
        if self.state == STATE_LEVELS:
            self.level1_button.update_hover(mouse_pos)
            self.level2_button.update_hover(mouse_pos)
            self.upload_button.update_hover(mouse_pos)
            if self.midi_valid:
                self.start_button.update_hover(mouse_pos)

        if self.state == STATE_SETTINGS:
            self.label_mode_button.update_hover(mouse_pos)
            self.cheats_button.update_hover(mouse_pos)
            self.easy_hold_button.update_hover(mouse_pos)
            self.settings_gameplay_tab_button.update_hover(mouse_pos)
            self.settings_visuals_tab_button.update_hover(mouse_pos)
            self.settings_audio_tab_button.update_hover(mouse_pos)
            self.settings_midi_tab_button.update_hover(mouse_pos)
            self.green_note_button.update_hover(mouse_pos)
            self.cheat_sustain_button.update_hover(mouse_pos)
            # Keys carousel pijltjes hover (alleen in Gameplay tab)
            if self.settings_tab == "gameplay":
                self.keys_carousel_left_arrow.update_hover(mouse_pos)
                self.keys_carousel_right_arrow.update_hover(mouse_pos)
            if self.settings_tab == "midi":
                # Refresh enabled state ook voor hover visuals
                try:
                    self.midi_search_device_button.enabled = (not bool(self.midi_input_connected))
                except Exception:
                    self.midi_search_device_button.enabled = True
                try:
                    self.midi_test_device_button.enabled = bool(self.midi_input_connected)
                except Exception:
                    self.midi_test_device_button.enabled = False

                try:
                    self.midi_refresh_device_button.enabled = bool(self.midi_input_connected)
                except Exception:
                    self.midi_refresh_device_button.enabled = False

                try:
                    self.midi_disconnect_device_button.enabled = bool(self.midi_input_connected)
                except Exception:
                    self.midi_disconnect_device_button.enabled = False

                self.midi_search_device_button.update_hover(mouse_pos)
                self.midi_refresh_device_button.update_hover(mouse_pos)
                self.midi_disconnect_device_button.update_hover(mouse_pos)
                if self.midi_input_connected:
                    self.midi_test_device_button.update_hover(mouse_pos)

        if self.state == STATE_SHOP:
            self.shop_themes_button.update_hover(mouse_pos)
            self.shop_background_button.update_hover(mouse_pos)
            self.shop_instruments_button.update_hover(mouse_pos)

        if self.state == STATE_PAUSED:
            self.pause_settings_button.update_hover(mouse_pos)
            self.pause_continue_button.update_hover(mouse_pos)
            self.pause_return_button.update_hover(mouse_pos)

    def start_transition(self, target_state, direction):
        # Render current and target state to surfaces, then animate.
        from_state = self.state
        from_surf = self.render_state_surface(from_state)
        to_surf = self.render_state_surface(target_state)

        self.transition_from = from_state
        self.transition_to = target_state
        self.transition_dir = direction  # 'down' | 'up' | 'right' | 'left'
        self.transition_frame = 0
        self.transition_from_surf = from_surf
        self.transition_to_surf = to_surf
        self.state = STATE_TRANSITION

    def render_state_surface(self, state):
        surf = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)).convert()
        if state == STATE_MENU:
            self._render_menu_to(surf)
        elif state == STATE_SETTINGS:
            self._render_settings_to(surf)
        elif state == STATE_SHOP:
            self._render_shop_to(surf)
        elif state == STATE_LEVELS:
            self._render_levels_to(surf)
        elif state == STATE_PLAYING:
            # Fallback; should not be used for requested transitions.
            surf.fill(DARK_GRAY)
        else:
            surf.fill(DARK_GRAY)
        return surf

    def _render_menu_to(self, surface):
        surface.fill(DARK_GRAY)
        title_text = self.font_xlarge.render("Fallin' Keys", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        surface.blit(title_text, title_rect)

        instruction_text = self.font_medium.render("Press SPACE to play", True, WHITE)
        instruction_rect = instruction_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
        surface.blit(instruction_text, instruction_rect)

        # Buttons
        self.play_button.draw(surface)
        self.levels_button.draw(surface)
        self.settings_button.draw(surface)
        self.shop_button.draw(surface)

    def _render_shop_to(self, surface):
        self._render_shop_screen(surface)

    def _render_shop_screen(self, surface):
        surface.fill(DARK_GRAY)

        title_text = self.font_xlarge.render("Shop", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 80))
        surface.blit(title_text, title_rect)

        # Bereken swipe animatie progress
        swipe_progress = 0.0
        if self.shop_swipe_frame > 0 and self.shop_tab == "instruments":
            swipe_progress = min(1.0, float(self.shop_swipe_frame) / float(self.shop_swipe_duration))
            # Ease-out effect
            swipe_progress = 1.0 - (1.0 - swipe_progress) * (1.0 - swipe_progress)

        # Left-side category buttons (selected category gets the same highlight feel)
        t_prev = self.shop_themes_button.hovered
        b_prev = self.shop_background_button.hovered
        i_prev = self.shop_instruments_button.hovered

        self.shop_themes_button.hovered = (self.shop_tab == "themes") or t_prev
        self.shop_background_button.hovered = (self.shop_tab == "background") or b_prev
        self.shop_instruments_button.hovered = (self.shop_tab == "instruments") or i_prev

        # Swipe animatie: buttons bewegen naar links en verdwijnen uit beeld
        swipe_offset = int(swipe_progress * -250)  # 250 pixels naar links (negatief)
        
        # Save originele posities
        themes_orig_x = self.shop_themes_button.rect.x
        bg_orig_x = self.shop_background_button.rect.x
        instr_orig_x = self.shop_instruments_button.rect.x
        
        # Apply swipe offset
        self.shop_themes_button.rect.x = int(themes_orig_x + swipe_offset)
        self.shop_background_button.rect.x = int(bg_orig_x + swipe_offset)
        self.shop_instruments_button.rect.x = int(instr_orig_x + swipe_offset)
        
        # Alleen tekenen als ze nog zichtbaar zijn (x > -200 om buiten beeld uit te gaan)
        if self.shop_themes_button.rect.x > -200:
            self.shop_themes_button.draw(surface)
        if self.shop_background_button.rect.x > -200:
            self.shop_background_button.draw(surface)
        if self.shop_instruments_button.rect.x > -200:
            self.shop_instruments_button.draw(surface)

        # Restore originele posities
        self.shop_themes_button.rect.x = themes_orig_x
        self.shop_background_button.rect.x = bg_orig_x
        self.shop_instruments_button.rect.x = instr_orig_x

        self.shop_themes_button.hovered = t_prev
        self.shop_background_button.hovered = b_prev
        self.shop_instruments_button.hovered = i_prev

        # Content area (rechts) – afhankelijk van geselecteerde tab
        if self.shop_tab == "instruments":
            # 3x3 grid van vierkantjes met groeiende animatie
            grid_size = 3
            base_square_size = 110  # Nog groter
            gap = 15
            
            # Schaal de vierkanten op basis van swipe progress (groeien van klein naar groot)
            min_scale = 0.3
            current_scale = min_scale + (1.0 - min_scale) * swipe_progress
            square_size = int(base_square_size * current_scale)
            
            # Bereken alpha voor fade in/out
            alpha = int(swipe_progress * 255)
            
            # Horizontale offset: bij reverse gaat het naar rechts uit beeld
            horizontal_offset = 0
            if self.shop_swipe_direction == -1:
                # Reverse: het rooster verdwijnt naar rechts
                reverse_progress = 1.0 - swipe_progress  # Omgekeerde progress
                horizontal_offset = int(reverse_progress * 300)  # 300 pixels naar rechts
            
            grid_width = grid_size * square_size + (grid_size - 1) * gap
            grid_height = grid_size * square_size + (grid_size - 1) * gap
            
            # Centeren in het scherm (met horizontale offset)
            start_x = (SCREEN_WIDTH - grid_width) // 2 + horizontal_offset
            start_y = (SCREEN_HEIGHT - grid_height) // 2
            
            # Draw 3x3 grid met fade effect
            for row in range(grid_size):
                for col in range(grid_size):
                    x = start_x + col * (square_size + gap)
                    y = start_y + row * (square_size + gap)
                    rect = pygame.Rect(x, y, square_size, square_size)
                    
                    # Controleer of dit vakje geselecteerd is
                    is_selected = (row, col) == self.shop_selected_instrument
                    
                    # Maak een surface met alpha
                    square_surf = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    pygame.draw.rect(square_surf, (100, 150, 200, alpha), (0, 0, square_size, square_size))  # Lichtblauwe vierkanten met alpha
                    
                    # Border dikte: dikker wanneer geselecteerd
                    border_width = 4 if is_selected else 2
                    border_color = (0, 255, 200, alpha) if is_selected else (255, 255, 255, alpha)
                    pygame.draw.rect(square_surf, border_color, (0, 0, square_size, square_size), border_width)
                    
                    surface.blit(square_surf, (x, y))
                    
                    # Pixel art elektrische piano in het eerste vakje
                    if row == 0 and col == 0:
                        # Piano tekenen met pixel art stijl
                        piano_x = x + 5
                        piano_y = y + 5
                        piano_w = square_size - 10
                        piano_h = square_size - 10
                        
                        # Piano body (donkere blauwe/grijze kleur met neon accenten)
                        # Maak een surface voor de piano
                        piano_surf = pygame.Surface((piano_w, piano_h), pygame.SRCALPHA)
                        
                        # Hoofd piano lichaam
                        body_h = int(piano_h * 0.6)
                        pygame.draw.rect(piano_surf, (20, 20, 50, alpha), (0, 0, piano_w, body_h))
                        pygame.draw.rect(piano_surf, (0, 200, 255, alpha), (0, 0, piano_w, body_h), 1)  # Neon rand
                        
                        # Display/screen area (bovenaan)
                        screen_h = int(body_h * 0.35)
                        pygame.draw.rect(piano_surf, (10, 30, 60, alpha), (4, 4, piano_w - 8, screen_h - 4))
                        # Neon glow voor screen
                        pygame.draw.rect(piano_surf, (200, 100, 255, int(alpha * 0.8)), (4, 4, piano_w - 8, screen_h - 4), 1)
                        
                        # Toetsenbord (onderkant)
                        keys_y = body_h - int(body_h * 0.4)
                        keys_h = int(body_h * 0.4)
                        keys_w = piano_w - 8
                        
                        # Zwarte toetsen en witte toetsen pattern
                        num_keys = 8
                        key_w = keys_w // num_keys
                        
                        for i in range(num_keys):
                            key_x = 4 + i * key_w
                            # Witte toets
                            pygame.draw.rect(piano_surf, (200, 200, 200, alpha), (key_x, keys_y, key_w - 1, keys_h))
                            pygame.draw.rect(piano_surf, (100, 100, 100, alpha), (key_x, keys_y, key_w - 1, keys_h), 1)
                            
                            # Zwarte toets (niet op elke toets)
                            if i % 3 != 2 and i < num_keys - 1:
                                black_w = int(key_w * 0.6)
                                black_x = key_x + int(key_w * 0.7)
                                black_h = int(keys_h * 0.6)
                                pygame.draw.rect(piano_surf, (30, 30, 30, alpha), (black_x, keys_y, black_w, black_h))
                                pygame.draw.rect(piano_surf, (100, 200, 255, alpha), (black_x, keys_y, black_w, black_h), 1)
                        
                        # Pads (cyaan) in het paarse scherm
                        pad_size = 4
                        pad_spacing = 2
                        num_pads = 8
                        total_pad_width = num_pads * pad_size + (num_pads - 1) * pad_spacing
                        pad_start_x = 6 + (piano_w - 12 - total_pad_width) // 2
                        pad_y = 6
                        
                        for pad in range(num_pads):
                            pad_x = pad_start_x + pad * (pad_size + pad_spacing)
                            pygame.draw.rect(piano_surf, (0, 200, 255, alpha), (pad_x, pad_y, pad_size, pad_size))
                            pygame.draw.rect(piano_surf, (100, 255, 255, alpha), (pad_x, pad_y, pad_size, pad_size), 1)
                        
                        # Blanco strip onderkant (waar neon glow zou zijn)
                        pygame.draw.rect(piano_surf, (0, 255, 200, int(alpha * 0.6)), (2, piano_h - 3, piano_w - 4, 3))
                        
                        surface.blit(piano_surf, (piano_x, piano_y))
                        
                        # "E-piano" label onder de piano (binnen het vakje)
                        text_font = self.font_small
                        text_surf = text_font.render("E-piano", True, WHITE)
                        
                        # Rechthoekig vakje rond de tekst
                        label_padding = 4
                        label_width = text_surf.get_width() + label_padding * 2
                        label_height = text_surf.get_height() + label_padding * 2
                        
                        # Plaats het label gecentreerd horizontaal en onderaan het vakje (met 10px padding)
                        label_x = x + (square_size - label_width) // 2
                        label_y = y + square_size - label_height - 10  # Onderaan het vakje met 10px padding
                        label_rect = pygame.Rect(label_x, label_y, label_width, label_height)
                        
                        # Maak een surface voor het label met alpha
                        label_surf = pygame.Surface((label_width, label_height), pygame.SRCALPHA)
                        pygame.draw.rect(label_surf, (0, 0, 0, int(alpha * 0.6)), (0, 0, label_width, label_height))  # Donkere achtergrond
                        pygame.draw.rect(label_surf, (255, 255, 255, alpha), (0, 0, label_width, label_height), 1)  # Witte rand
                        
                        surface.blit(label_surf, (label_x, label_y))
                        
                        # Tekst in het midden van het vakje
                        text_x = label_x + (label_width - text_surf.get_width()) // 2
                        text_y = label_y + (label_height - text_surf.get_height()) // 2
                        surface.blit(text_surf, (text_x, text_y))
                    
                    # Pixel art trompet in het tweede vakje (rechtsboven, row=0, col=1)
                    if row == 0 and col == 1:
                        # Trompet tekenen met pixel art stijl
                        trumpet_x = x + 5
                        trumpet_y = y + 5
                        trumpet_w = square_size - 10
                        trumpet_h = square_size - 10
                        
                        # Trompet surface
                        trumpet_surf = pygame.Surface((trumpet_w, trumpet_h), pygame.SRCALPHA)
                        
                        # Basis kleur: goud/geel met neon accenten
                        mid_y = int(trumpet_h * 0.45)
                        tube_h = max(6, int(trumpet_h * 0.18))
                        tube_y = mid_y - tube_h // 2
                        tube_x = int(trumpet_w * 0.08)
                        tube_w = int(trumpet_w * 0.62)
                        
                        # Hoofd buis (lange body)
                        pygame.draw.rect(trumpet_surf, (200, 160, 50, alpha), (tube_x, tube_y, tube_w, tube_h))
                        pygame.draw.rect(trumpet_surf, (255, 200, 0, alpha), (tube_x, tube_y, tube_w, tube_h), 1)
                        
                        # Bel (grote uitgang) rechts
                        bell_w = int(trumpet_w * 0.28)
                        bell_h = int(trumpet_h * 0.35)
                        bell_x = tube_x + tube_w - int(bell_w * 0.2)
                        bell_y = mid_y - bell_h // 2
                        pygame.draw.ellipse(trumpet_surf, (180, 140, 30, alpha), (bell_x, bell_y, bell_w, bell_h))
                        pygame.draw.ellipse(trumpet_surf, (220, 180, 80, alpha), (bell_x, bell_y, bell_w, bell_h), 1)
                        
                        # Buis flare naar bel (onder)
                        flare = [
                            (tube_x + int(tube_w * 0.55), tube_y + tube_h),
                            (bell_x, mid_y + bell_h // 2),
                            (bell_x - int(bell_w * 0.25), mid_y + int(bell_h * 0.35)),
                            (tube_x + int(tube_w * 0.35), tube_y + tube_h)
                        ]
                        pygame.draw.polygon(trumpet_surf, (180, 140, 30, alpha), flare)
                        pygame.draw.polygon(trumpet_surf, (220, 180, 80, alpha), flare, 1)
                        
                        # Kleppen/ventielen (drie cylinders) boven de buis
                        valve_w = max(4, int(trumpet_w * 0.06))
                        valve_h = max(8, int(trumpet_h * 0.18))
                        valve_y = tube_y - valve_h + 2
                        valve_start_x = tube_x + int(trumpet_w * 0.18)
                        valve_gap = valve_w + max(3, int(trumpet_w * 0.04))
                        for valve in range(3):
                            valve_x = valve_start_x + valve * valve_gap
                            pygame.draw.rect(trumpet_surf, (160, 120, 20, alpha), (valve_x, valve_y, valve_w, valve_h))
                            pygame.draw.rect(trumpet_surf, (255, 200, 0, alpha), (valve_x, valve_y, valve_w, valve_h), 1)
                        
                        # Mondstuk links
                        mouth_w = max(4, int(trumpet_w * 0.06))
                        mouth_h = max(8, int(trumpet_h * 0.22))
                        mouth_x = tube_x - mouth_w - 2
                        mouth_y = mid_y - mouth_h // 2
                        pygame.draw.rect(trumpet_surf, (100, 80, 20, alpha), (mouth_x, mouth_y, mouth_w, mouth_h))
                        pygame.draw.rect(trumpet_surf, (150, 120, 50, alpha), (mouth_x, mouth_y, mouth_w, mouth_h), 1)
                        
                        surface.blit(trumpet_surf, (trumpet_x, trumpet_y))
                        
                        # "Trumpet" label onder de trompet (binnen het vakje)
                        text_font = self.font_small
                        text_surf = text_font.render("Trumpet", True, WHITE)
                        
                        # Rechthoekig vakje rond de tekst
                        label_padding = 4
                        label_width = text_surf.get_width() + label_padding * 2
                        label_height = text_surf.get_height() + label_padding * 2
                        
                        # Plaats het label gecentreerd horizontaal en onderaan het vakje (met 10px padding)
                        label_x = x + (square_size - label_width) // 2
                        label_y = y + square_size - label_height - 10
                        label_rect = pygame.Rect(label_x, label_y, label_width, label_height)
                        
                        # Maak een surface voor het label met alpha
                        label_surf = pygame.Surface((label_width, label_height), pygame.SRCALPHA)
                        pygame.draw.rect(label_surf, (0, 0, 0, int(alpha * 0.6)), (0, 0, label_width, label_height))  # Donkere achtergrond
                        pygame.draw.rect(label_surf, (255, 255, 255, alpha), (0, 0, label_width, label_height), 1)  # Witte rand
                        
                        surface.blit(label_surf, (label_x, label_y))
                        
                        # Tekst in het midden van het vakje
                        text_x = label_x + (label_width - text_surf.get_width()) // 2
                        text_y = label_y + (label_height - text_surf.get_height()) // 2
                        surface.blit(text_surf, (text_x, text_y))
        else:
            info = "Select: Themes" if self.shop_tab == "themes" else "Select: Background"
            info_surf = self.font_medium.render(info, True, WHITE)
            info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2 + 60, 220))
            surface.blit(info_surf, info_rect)

        back_text = self.font_medium.render("Press ESC to go back", True, WHITE)
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
        surface.blit(back_text, back_rect)

    def _render_settings_to(self, surface):
        self._render_settings_screen(surface)

    def _render_settings_screen(self, surface):
        surface.fill(DARK_GRAY)

        title_text = self.font_xlarge.render("Settings", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 80))
        surface.blit(title_text, title_rect)

        # Tabs (selected tab is drawn as "hovered")
        g_prev = self.settings_gameplay_tab_button.hovered
        v_prev = self.settings_visuals_tab_button.hovered
        a_prev = self.settings_audio_tab_button.hovered
        m_prev = self.settings_midi_tab_button.hovered

        self.settings_gameplay_tab_button.hovered = (self.settings_tab == "gameplay") or g_prev
        self.settings_visuals_tab_button.hovered = (self.settings_tab == "visuals") or v_prev
        self.settings_audio_tab_button.hovered = (self.settings_tab == "audio") or a_prev
        self.settings_midi_tab_button.hovered = (self.settings_tab == "midi") or m_prev

        self.settings_gameplay_tab_button.draw(surface)
        self.settings_visuals_tab_button.draw(surface)
        self.settings_audio_tab_button.draw(surface)
        self.settings_midi_tab_button.draw(surface)

        self.settings_gameplay_tab_button.hovered = g_prev
        self.settings_visuals_tab_button.hovered = v_prev
        self.settings_audio_tab_button.hovered = a_prev
        self.settings_midi_tab_button.hovered = m_prev

        if self.settings_tab == "visuals":
            info_text = self.font_medium.render("Blue hit:", True, WHITE)
            info_rect = info_text.get_rect(center=(SCREEN_WIDTH // 2, 230))
            surface.blit(info_text, info_rect)
            self.green_note_button.draw(surface)

            mode_text = "Aan: noot wordt blauw bij goed" if self.green_note_enabled else "Uit: noot blijft geel"
            mode_surf = self.font_small.render(mode_text, True, WHITE)
            mode_rect = mode_surf.get_rect(center=(SCREEN_WIDTH // 2, 340))
            surface.blit(mode_surf, mode_rect)

        elif self.settings_tab == "audio":
            info_text = self.font_medium.render("Cheats auto-sustain:", True, WHITE)
            info_rect = info_text.get_rect(center=(SCREEN_WIDTH // 2, 210))
            surface.blit(info_text, info_rect)

            self.cheat_sustain_button.draw(surface)

            mode_text = (
                f"Aan: lange noten klinken ~{self.cheat_auto_sustain_seconds:.1f}s bij verdwijnen"
                if self.cheat_auto_sustain_enabled
                else "Uit: bij verdwijnen alleen korte tik"
            )
            mode_surf = self.font_small.render(mode_text, True, WHITE)
            mode_rect = mode_surf.get_rect(center=(SCREEN_WIDTH // 2, 330))
            surface.blit(mode_surf, mode_rect)

        elif self.settings_tab == "midi":
            title = self.font_medium.render("MIDI Input Device:", True, WHITE)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 210))
            surface.blit(title, title_rect)

            # Knoppen: wanneer connected -> 'Connected' + 'Refresh' naast elkaar.
            base_x = (SCREEN_WIDTH - 260) // 2
            if self.midi_input_connected:
                self.midi_search_device_button.rect = pygame.Rect(base_x, 220, 130, 50)
                self.midi_search_device_button.text = "Connected"
                self.midi_search_device_button.enabled = False
                self.midi_search_device_button.draw(surface)

                # Refresh icon button: vierkant, 50x50, met ruimte rechts
                refresh_x = base_x + 140
                self.midi_refresh_device_button.x = refresh_x
                self.midi_refresh_device_button.y = 220
                self.midi_refresh_device_button.rect = pygame.Rect(refresh_x, 220, 50, 50)
                self.midi_refresh_device_button.size = 50
                self.midi_refresh_device_button.enabled = True
                self.midi_refresh_device_button.draw(surface)

                # Disconnect icon button: vierkant, 50x50, met kruis
                disconnect_x = base_x + 200
                self.midi_disconnect_device_button.x = disconnect_x
                self.midi_disconnect_device_button.y = 220
                self.midi_disconnect_device_button.rect = pygame.Rect(disconnect_x, 220, 50, 50)
                self.midi_disconnect_device_button.size = 50
                self.midi_disconnect_device_button.enabled = True
                self.midi_disconnect_device_button.draw(surface)
            else:
                self.midi_search_device_button.rect = pygame.Rect(base_x, 220, 260, 50)
                self.midi_search_device_button.text = "Search Device"
                self.midi_search_device_button.enabled = True
                self.midi_search_device_button.draw(surface)

            # Test button: pas tonen/actief als er een device connected is
            if self.midi_input_connected:
                dev_name = self.midi_input_device_name or "Device"
                self.midi_test_device_button.text = f"Test MIDI {dev_name}"
                self.midi_test_device_button.enabled = True
                self.midi_test_device_button.draw(surface)

            name = self.midi_input_device_name or "(geen)"
            connected = "JA" if self.midi_input_connected else "NEE"
            status_line = f"Device: {name}"
            conn_line = f"Connected met game: {connected}"
            search_line = f"Status: {self.midi_input_last_search}"

            s1 = self.font_small.render(status_line, True, WHITE)
            s2 = self.font_small.render(conn_line, True, WHITE)
            s3 = self.font_small.render(search_line, True, WHITE)
            y0 = 360 if self.midi_input_connected else 320
            surface.blit(s1, s1.get_rect(center=(SCREEN_WIDTH // 2, y0)))
            surface.blit(s2, s2.get_rect(center=(SCREEN_WIDTH // 2, y0 + 30)))
            surface.blit(s3, s3.get_rect(center=(SCREEN_WIDTH // 2, y0 + 60)))

            if self.midi_input_last_error:
                err = self.font_small.render(self.midi_input_last_error, True, (255, 120, 120))
                surface.blit(err, err.get_rect(center=(SCREEN_WIDTH // 2, y0 + 100)))

            # Toon RX info zodat je ziet of de game echt MIDI events ontvangt
            if self.midi_input_connected:
                last = self.midi_last_note_name or "(nog niets)"
                rx_line = f"RX: {int(self.midi_rx_count)}  Last: {last}"
                rx_s = self.font_small.render(rx_line, True, WHITE)
                surface.blit(rx_s, rx_s.get_rect(center=(SCREEN_WIDTH // 2, y0 + 130)))

            hint_y = 510 if self.midi_input_connected else 470
            hint = self.font_small.render("Tip: speel mee door MIDI noten te drukken", True, WHITE)
            surface.blit(hint, hint.get_rect(center=(SCREEN_WIDTH // 2, hint_y)))

        else:  # Gameplay tab
            info_text = self.font_medium.render("Tile labels:", True, WHITE)
            info_rect = info_text.get_rect(center=(SCREEN_WIDTH // 2, 210))
            surface.blit(info_text, info_rect)

            # Bepaal de huidige en vorige optie-tekst
            current_label = self.keys_carousel_options[self.keys_carousel_current_index]
            previous_label = self.keys_carousel_options[self.keys_carousel_previous_index]
            
            # Animatie: bereken offset op basis van slide-progress
            if self.keys_carousel_slide_frames > 0 and self.keys_carousel_slide_direction != 0:
                progress = float(self.keys_carousel_slide_frames) / float(self.KEYS_CAROUSEL_SLIDE_DURATION)
                # Smooth easing
                progress = progress * progress * (3 - 2 * progress)
                slide_offset = progress * SCREEN_WIDTH  # Van 0 naar volledige breedte
                
                # Teken BEIDE buttons: de oude die verdwijnt en de nieuwe die binnenkomt
                original_x = self.label_mode_button.rect.x
                
                if self.keys_carousel_slide_direction == 1:  # Van links
                    # Oude tekst gaat naar rechts
                    old_offset = slide_offset
                    # Nieuwe tekst komt van links
                    new_offset = -SCREEN_WIDTH + slide_offset
                elif self.keys_carousel_slide_direction == -1:  # Van rechts
                    # Oude tekst gaat naar links
                    old_offset = -slide_offset
                    # Nieuwe tekst komt van rechts
                    new_offset = SCREEN_WIDTH - slide_offset
                else:
                    old_offset = 0
                    new_offset = 0
                
                # Teken de oude button die verdwijnt
                self.label_mode_button.text = previous_label
                self.label_mode_button.rect.x = int(original_x + old_offset)
                self.label_mode_button.draw(surface)
                
                # Teken de nieuwe button die binnenkomt
                self.label_mode_button.text = current_label
                self.label_mode_button.rect.x = int(original_x + new_offset)
                self.label_mode_button.draw(surface)
                
                # Reset de positie terug
                self.label_mode_button.rect.x = original_x
            else:
                # Geen animatie: teken gewoon de huidige button
                self.label_mode_button.text = current_label
                self.label_mode_button.draw(surface)

            # Teken pijltjes NA de buttons zodat ze vooraan blijven (labels gaan erachter langs)
            self.keys_carousel_left_arrow.draw(surface)
            self.keys_carousel_right_arrow.draw(surface)

            # Toon beschrijving gebaseerd op huidige optie
            descriptions = {
                "Keys": "Toon toetsenbord-letters",
                "Note names": "Toon notenaam van de noot",
                "Straight keys": "Piano toetsen zonder kleuren"
            }
            mode_text = descriptions.get(current_label, "")
            if mode_text:
                mode_surf = self.font_small.render(mode_text, True, WHITE)
                mode_rect = mode_surf.get_rect(center=(SCREEN_WIDTH // 2, 300))
                surface.blit(mode_surf, mode_rect)

            cheats_label = self.font_medium.render("Cheats:", True, WHITE)
            cheats_rect = cheats_label.get_rect(center=(SCREEN_WIDTH // 2, 330))
            surface.blit(cheats_label, cheats_rect)

            self.cheats_button.draw(surface)

            easy_label = self.font_medium.render("Easy:", True, WHITE)
            easy_rect = easy_label.get_rect(center=(SCREEN_WIDTH // 2, 420))
            surface.blit(easy_label, easy_rect)

            self.easy_hold_button.draw(surface)

            # Extra uitlegtekst onder Easy knop
            easy_text = "Aan: 1x tikken start de noot (vasthouden niet nodig)" if self.easy_hold_enabled else "Uit: vasthouden nodig voor lange noten"
            easy_surf = self.font_small.render(easy_text, True, WHITE)
            easy_y = int(self.easy_hold_button.rect.bottom + 16)
            easy_rect2 = easy_surf.get_rect(center=(SCREEN_WIDTH // 2, easy_y))
            surface.blit(easy_surf, easy_rect2)

        back_text = self.font_medium.render("Press ESC to go back", True, WHITE)
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
        surface.blit(back_text, back_rect)

    def _render_levels_to(self, surface):
        surface.fill(DARK_GRAY)

        title_text = self.font_xlarge.render("Levels", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        surface.blit(title_text, title_rect)

        # Level 1 (vast)
        self.level1_button.draw(surface)
        # Level 2 (vast)
        self.level2_button.draw(surface)

        # Sterren naast elk level (leeg/zwart als nog niet gehaald)
        lvl1_stars = int(self.level_ratings.get("level1", 0))
        self._draw_level_stars_next_to_button(surface, self.level1_button.rect, lvl1_stars)
        lvl2_stars = int(self.level_ratings.get("level2", 0))
        self._draw_level_stars_next_to_button(surface, self.level2_button.rect, lvl2_stars)

        # Upload (custom level)
        self.upload_button.draw(surface)

        # Loading tekst onder het plusje
        if getattr(self, 'midi_loading', False):
            loading_text = self.font_small.render("Loading", True, WHITE)
            loading_rect = loading_text.get_rect(center=(self.upload_button.rect.centerx, self.upload_button.rect.bottom + 18))
            surface.blit(loading_text, loading_rect)

        if self.midi_file_path:
            file_name = os.path.basename(self.midi_file_path)
            file_text = self.font_small.render(f"Bestand: {file_name}", True, WHITE)
            file_rect = file_text.get_rect(center=(SCREEN_WIDTH // 2, 390))
            surface.blit(file_text, file_rect)

            if getattr(self, 'midi_loading', False):
                status_text = self.font_small.render("Loading", True, WHITE)
                status_rect = status_text.get_rect(center=(SCREEN_WIDTH // 2, 500))
                surface.blit(status_text, status_rect)
            else:
                if self.midi_valid:
                    status_text = self.font_small.render("MIDI bestand geldig!", True, (0, 255, 0))
                    self.start_button.draw(surface)
                else:
                    status_text = self.font_small.render("Geen geldig MIDI bestand", True, (255, 0, 0))

                status_rect = status_text.get_rect(center=(SCREEN_WIDTH // 2, 440))
                surface.blit(status_text, status_rect)

        back_text = self.font_medium.render("Press ESC to go back", True, WHITE)
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, 540))
        surface.blit(back_text, back_rect)

    def _draw_loading_spinner(self, surface, center, radius=10, color=WHITE):
        try:
            base_color = color
            ticks = pygame.time.get_ticks()
            angle_base = (ticks % 2000) / 2000.0 * 2.0 * math.pi
            dot_count = 12
            for i in range(dot_count):
                angle = angle_base + (i * 2.0 * math.pi / dot_count)
                factor = (i + 1) / float(dot_count)
                c = (
                    int(base_color[0] * factor),
                    int(base_color[1] * factor),
                    int(base_color[2] * factor)
                )
                x = int(center[0] + math.cos(angle) * radius)
                y = int(center[1] + math.sin(angle) * radius)
                pygame.draw.circle(surface, c, (x, y), 2)
        except Exception:
            pass

    def return_to_main_from_game(self):
        # Stop de huidige run en ga terug naar hoofdmenu
 
        self.clear_uploaded_midi()
        self.tiles = []
        self.miss_pending = False
        self.miss_timer = 0
        self.use_midi_schedule = False
        self.scheduled_notes = []
        self.game_frame = 0
        self.pause_background = None
        self.pause_background_blurred = None
        self.game_over_frame = 0
        self.game_over_bg = None
        self.game_over_bg_blur = None
        self.level_stars = None
        self.level_note_count = 0
        self.level_score_thresholds = None
        self.current_level_id = None
        self.pressed_columns.clear()
        self.keys_currently_pressed.clear()
        self.active_holds.clear()
        self.sustain_channel_by_column.clear()
        for col in range(NUM_COLUMNS):
            self.stop_sustain_sound(col, fade_ms=0)
        self.state = STATE_MENU

    def _blur_surface(self, surface):
        w, h = surface.get_size()
        scale = 0.12
        sw = max(1, int(w * scale))
        sh = max(1, int(h * scale))
        small = pygame.transform.smoothscale(surface, (sw, sh))
        blurred = pygame.transform.smoothscale(small, (w, h))
        # Extra blur pass voor iets sterker effect
        small2 = pygame.transform.smoothscale(blurred, (sw, sh))
        blurred2 = pygame.transform.smoothscale(small2, (w, h))
        return blurred2

    def enter_pause(self):
        # Freeze: neem screenshot van huidige frame en blur dit
        self.pause_background = self.screen.copy()
        self.pause_background_blurred = self._blur_surface(self.pause_background)
        self.state = STATE_PAUSED

    def start_countdown(self):
        self.state = STATE_COUNTDOWN
        self.countdown_value = 3
        self.countdown_timer = 0
        # Clear eventuele keys die nog ingedrukt zijn
        self.keys_currently_pressed.clear()
    
    def check_tile_hit(self, column):
        # Tap-notes (korte blokjes): direct hit op keydown.
        # Hold-notes (lange balken): keydown start de hold, score pas na volledige hold.
        if column in self.active_holds:
            return

        # Kies alleen tiles die nog niet geraakt/ gemist zijn.
        # Pak de tile die het dichtst bij de bar zit.
        if self.cheats_enabled:
            # Cheats: te vroeg/te laat maakt niet uit.
            candidates = [
                t for t in self.tiles
                if t.column == column and (not t.hit) and (not t.missed) and (not t.holding)
            ]
        else:
            candidates = [
                t for t in self.tiles
                if t.column == column and (not t.hit) and (not t.missed) and (not t.holding) and t.is_head_on_bar()
            ]

        if candidates:
            tile = max(candidates, key=lambda t: t.y)
            # Korte noot? Behandel als "tap" zodat hij niet eerst verdwijnt
            # (latched hold rendering) en daarna weer zichtbaar wordt bij loslaten.
            if tile.height <= BAR_HEIGHT:
                if self.use_midi_schedule:
                    self._record_played_tile(tile)
                tile.mark_as_hit()
                # Judgement voor tap-notes (ook bij random Play): afstand van de timing-lijn (BAR_Y)
                head_y = float(tile.y + tile.height)
                dist = abs(head_y - float(BAR_Y))
                if dist <= float(PERFECT_RELEASE_MARGIN_PX):
                    self.score += 3
                    self._set_judgement("Perfect")
                elif dist <= float(HOLD_RELEASE_MARGIN_PX):
                    self.score += 2
                    self._set_judgement("Great")
                else:
                    self.score += 1
                    self._set_judgement("Allright")
                return

            tile.holding = True
            self.active_holds[column] = tile
            # Start sustain loop: blijft spelen zolang je ingedrukt houdt
            # Gebruik originele MIDI-noot voor juiste toonhoogte
            self.start_sustain_sound(column, original_note_num=tile.note_num)
            return

        if not self.cheats_enabled:
            # Niet meteen game over: eerst key rood tonen, dan game over.
            self._reset_combo()
            self.wrong_key_column = int(column)
            self.miss_pending = True
            self.miss_timer = 0

    def _draw_wrong_key_highlight(self, column: int, strength: float = 1.0) -> None:
        bar = self.bars[column]

        strength = max(0.0, min(1.0, float(strength)))
        a_fill = int(180 * strength)
        a_border = int(220 * strength)
        fill = (255, 60, 60, a_fill)
        border = (255, 180, 180, a_border)

        overlay = pygame.Surface((bar.width, bar.height), pygame.SRCALPHA)
        overlay.fill(fill)
        pygame.draw.rect(overlay, border, overlay.get_rect(), 3)
        self.screen.blit(overlay, (bar.x, bar.y))

    def _set_judgement(self, text):
        # Als er al een judgement zichtbaar is en er komt een volgende,
        # laat de vorige naar beneden schuiven en vervagen.
        if self.last_judgement:
            in_frames = max(1, int(JUDGEMENT_IN_FRAMES))
            t_in = min(1.0, float(self.last_judgement_age) / float(in_frames))
            # Ease-out (zelfde als draw)
            t_in = 1.0 - (1.0 - t_in) * (1.0 - t_in) * (1.0 - t_in)
            alpha0 = int(255 * t_in)
            if alpha0 > 0:
                y_off = int((1.0 - t_in) * -float(JUDGEMENT_SLIDE_PX))
                y0 = int(10 + y_off)
                self.judgement_history.append({
                    'text': str(self.last_judgement),
                    'age': 0,
                    'alpha0': int(alpha0),
                    'y0': int(y0),
                })
                if len(self.judgement_history) > int(JUDGEMENT_HISTORY_MAX):
                    self.judgement_history = self.judgement_history[-int(JUDGEMENT_HISTORY_MAX):]

        self.last_judgement = str(text)
        self.last_judgement_timer = int(self.last_judgement_duration)
        self.last_judgement_age = 0

        # Combo telt alleen op bij dezelfde judgement achter elkaar.
        if self.last_judgement in ("Perfect", "Great", "Allright"):
            if self.combo_kind == self.last_judgement:
                self.combo_streak += 1
            else:
                self.combo_kind = self.last_judgement
                self.combo_streak = 1
        else:
            self.combo_streak = 0
            self.combo_kind = None

    def _reset_combo(self):
        self.combo_streak = 0
        self.combo_kind = None

    def _award_hold_score(self, tile, column, points, judgement_text):
        tile.holding = False
        if self.use_midi_schedule:
            self._record_played_tile(tile)
        tile.mark_as_hit()
        self.score += int(points)
        self._set_judgement(judgement_text)
        if self.active_holds.get(column) is tile:
            self.active_holds.pop(column, None)

    def score_hold_on_release(self, tile, column):
        """Bepaal Perfect/Great/Allright op basis van hoe laat je loslaat.

        - Perfect: binnen 20px van het einde (+3)
        - Great: binnen de bestaande marge (+2)
        - Allright: eerder loslaten (+1, geen penalty)
        """
        if tile is None or tile.hit or tile.missed:
            return

        # Als hij nog niet latched is, tel dit als een vroege release.
        if not getattr(tile, 'latched', False):
            self._award_hold_score(tile, column, points=1, judgement_text="Allright")
            return

        elapsed = int(getattr(tile, 'hold_frames_elapsed', 0))
        perfect_req = int(getattr(tile, 'hold_frames_required_perfect', 10**9))
        great_req = int(getattr(tile, 'hold_frames_required_great', getattr(tile, 'hold_frames_required', 10**9)))

        if elapsed >= perfect_req:
            self._award_hold_score(tile, column, points=3, judgement_text="Perfect")
        elif elapsed >= great_req:
            self._award_hold_score(tile, column, points=2, judgement_text="Great")
        else:
            self._award_hold_score(tile, column, points=1, judgement_text="Allright")

    def complete_hold(self, tile, column):
        # Volledig vasthouden = beste resultaat.
        self._award_hold_score(tile, column, points=3, judgement_text="Perfect")

    def fail_hold(self, tile, column):
        tile.holding = False
        self.stop_sustain_sound(column)
        if self.active_holds.get(column) is tile:
            self.active_holds.pop(column, None)

        tile.missed = True

        # Hold fail = combo breken
        self._reset_combo()

        if self.cheats_enabled:
            # Cheats: geen game over bij vroeg loslaten.
            return

        self.miss_pending = True
        self.miss_timer = 0
    
    def spawn_tiles(self):
        self.spawn_counter += 1
        if self.spawn_counter >= SPAWN_RATE:
            # Na 10 seconden (600 frames): 10% kans op dubbele spawn
            if self.play_time > 600 and random.random() < 0.1:
                # Spawn 2 tiles in verschillende kolommen
                column1 = random.randint(0, NUM_COLUMNS - 1)
                column2 = random.randint(0, NUM_COLUMNS - 1)
                while column2 == column1:
                    column2 = random.randint(0, NUM_COLUMNS - 1)
                self.tiles.append(Tile(column1, font=self.get_tile_font_for_column(column1), label_text=self.get_label_for_column(column1)))
                self.tiles.append(Tile(column2, font=self.get_tile_font_for_column(column2), label_text=self.get_label_for_column(column2)))
            else:
                column = random.randint(0, NUM_COLUMNS - 1)
                self.tiles.append(Tile(column, font=self.get_tile_font_for_column(column), label_text=self.get_label_for_column(column)))
            self.spawn_counter = 0
    
    def update(self):
        # MIDI input: blijf altijd pollen zodat RX/Last ook in Settings live werkt.
        # Gameplay-acties (hit/hold) worden in _midi_press_column/_midi_release_column al beperkt tot STATE_PLAYING.
        self._poll_midi_input()

        # MIDI output: werk note-offs af (voor tap-notes)
        self._process_pending_midi_note_off()

        # Keys carousel animatie updaten (als in progress)
        if self.keys_carousel_slide_frames < self.KEYS_CAROUSEL_SLIDE_DURATION:
            self.keys_carousel_slide_frames += 1
        else:
            self.keys_carousel_slide_direction = 0  # Animatie klaar

        # Shop swipe animatie updaten
        if self.shop_swipe_direction == 1:
            # Forward animation
            if self.shop_swipe_frame > 0 and self.shop_swipe_frame < self.shop_swipe_duration:
                self.shop_swipe_frame += 1
        elif self.shop_swipe_direction == -1:
            # Reverse animation
            if self.shop_swipe_frame > 0:
                self.shop_swipe_frame -= 1
                if self.shop_swipe_frame <= 0:
                    # Animatie klaar, reset tab
                    self.shop_tab = "themes"
                    self.shop_swipe_frame = 0
                    self.shop_swipe_direction = 0

        # MIDI analyse uitstellen zodat "Loading" eerst zichtbaar is.
        if getattr(self, 'midi_analysis_pending', False) and self.state == STATE_LEVELS:
            try:
                self.midi_loading_frames = int(getattr(self, 'midi_loading_frames', 0)) + 1
            except Exception:
                self.midi_loading_frames = 1
            if self.midi_loading_frames >= 2:
                file_path = getattr(self, 'midi_file_path', None)
                if file_path and self.midi_valid:
                    self.analyze_midi(file_path)
                self.midi_analysis_pending = False
                self.midi_loading = False

        if self.state == STATE_GAME_OVER:
            self.game_over_frame += 1
            return

        if self.state == STATE_LEVEL_COMPLETED:
            self.level_complete_frame += 1
            # Eerst het level-complete paneel tonen, daarna blur + vallende noten.
            if self.level_complete_frame >= LEVEL_COMPLETE_FALL_DELAY_FRAMES:
                # Laat celebration tiles hard naar beneden vallen.
                for t in self.celebration_tiles[:]:
                    t.y += t.speed
                    if t.y > SCREEN_HEIGHT + t.height:
                        self.celebration_tiles.remove(t)
            return

        if self.state == STATE_AUTO_REPLAY:
            self.auto_replay_frame += 1
            # Speed-up effect
            self.auto_replay_speed_mult = min(
                self.auto_replay_speed_cap,
                float(self.auto_replay_speed_mult) * float(self.auto_replay_speed_growth)
            )

            # Gebruik een geschaalde tijdlijn zodat de afstand tussen noten hetzelfde blijft,
            # maar alles in minder "echte" tijd gebeurt.
            self.auto_replay_time += float(self.auto_replay_speed_mult)

            while self.scheduled_notes and self.scheduled_notes[0]['spawn_frame'] <= self.auto_replay_time:
                s = self.scheduled_notes.pop(0)
                col = s['column']
                length = s.get('length_pixels')
                note = s.get('note')
                original_notes = s.get('original_notes') or [s.get('original_note', note)]
                t = Tile(col, length_pixels=length, note_num=list(original_notes), font=self.get_tile_font_for_column(col), label_text=self.get_label_for_column(col))
                t.speed = TILE_SPEED * self.auto_replay_speed_mult
                t.holding = False
                t.latched = False
                self.tiles.append(t)

            for tile in self.tiles[:]:
                tile.speed = TILE_SPEED * self.auto_replay_speed_mult
                tile.update()
                if tile.y > SCREEN_HEIGHT + tile.height:
                    self.tiles.remove(tile)

            if (not self.scheduled_notes) and (not self.tiles):
                self.start_level_completed()
            return

        if self.state == STATE_TUTORIAL_WELCOME:
            # Tutorial welcome scherm valt van boven naar beneden
            self.tutorial_welcome_frame += 1
            return

        if self.state == STATE_TRANSITION:
            self.transition_frame += 1
            if self.transition_frame >= self.transition_duration:
                self.state = self.transition_to
                self.transition_from = None
                self.transition_to = None
                self.transition_dir = None
                self.transition_from_surf = None
                self.transition_to_surf = None
            return

        if self.state == STATE_COUNTDOWN:
            self.countdown_timer += 1
            if self.countdown_timer >= FPS:
                self.countdown_timer = 0
                self.countdown_value -= 1
                if self.countdown_value <= 0:
                    self.state = STATE_PLAYING
                    # Clear keys die tijdens countdown zijn ingedrukt
                    self.keys_currently_pressed.clear()
                    # Na countdown, gebruik live rendering weer
                    self.pause_background = None
                    self.pause_background_blurred = None
            return

        if self.state != STATE_PLAYING:
            return

        # Poll keyboard state om meerdere simultane toetsen te detecteren
        # Dit omzeilt keyboard ghosting/event buffer problemen volledig
        keys = pygame.key.get_pressed()
        
        # Kies de juiste key mapping op basis van de carousel index
        # 0=Keys (A,W,S,...), 1=Note names (A,W,S,...), 2=Straight keys (1,2,3,...)
        if self.keys_carousel_current_index == 2:  # Straight keys
            active_key_mapping = self.straight_key_to_column
        else:  # Keys of Note names (beide gebruiken dezelfde toetsen)
            active_key_mapping = self.key_to_column
        
        for key, column in active_key_mapping.items():
            is_pressed = keys[key]
            was_pressed = key in self.keys_currently_pressed
            
            if is_pressed and not was_pressed:
                # Nieuwe keypress gedetecteerd via polling
                self.keys_currently_pressed.add(key)
                self.pressed_columns.add(column)
                self._spawn_smoke_for_column(column, burst=True)
                self.start_sustain_sound(column)
                self.check_tile_hit(column)
            elif not is_pressed and was_pressed:
                # Key release gedetecteerd via polling
                self.keys_currently_pressed.discard(key)
                self.pressed_columns.discard(column)
                self.stop_sustain_sound(column, fade_ms=150)
                
                # Hold-note release scoring
                if column in self.active_holds and (not self.easy_hold_enabled):
                    tile = self.active_holds.get(column)
                    if tile is not None and (not tile.hit) and (not tile.missed):
                        self.score_hold_on_release(tile, column)

        # Update smoke particles + spawn zolang toetsen ingedrukt zijn
        if self.pressed_columns:
            for col in self.pressed_columns:
                self._spawn_smoke_for_column(int(col), burst=False)
        if self.smoke_particles:
            for p in list(self.smoke_particles):
                p.update()
                if p.dead:
                    try:
                        self.smoke_particles.remove(p)
                    except ValueError:
                        pass

        # Judgement UI: blijf zichtbaar totdat er een nieuwe judgement komt.
        # (De vorige judgement wordt in _set_judgement() naar de history gepusht en fade daar uit.)
        if self.last_judgement:
            self.last_judgement_age += 1
        else:
            self.last_judgement_age = 0

        # Update judgement history (oude meldingen die uitfaden)
        if self.judgement_history:
            out_frames = max(1, int(JUDGEMENT_OUT_FRAMES))
            for item in list(self.judgement_history):
                try:
                    item['age'] = int(item.get('age', 0)) + 1
                except Exception:
                    item['age'] = 0
                if int(item.get('age', 0)) >= out_frames:
                    try:
                        self.judgement_history.remove(item)
                    except ValueError:
                        pass
        
        # Track play time
        self.play_time += 1

        # Tutorial slowmo trigger (na 2 seconden vallen)
        slowmo_mult = 1.0
        if self.tutorial_mode and (self.state == STATE_PLAYING) and (not self.tutorial_slowmo_done) and (self.play_time >= int(self.tutorial_slowmo_after_frames)):
            self.tutorial_slowmo_active = True
            slowmo_mult = float(self.tutorial_slowmo_multiplier)
            if self.tutorial_slowmo_start_frame is None:
                self.tutorial_slowmo_start_frame = int(self.play_time)
            if not self.tutorial_overlay_active:
                self.tutorial_overlay_active = True
                self.tutorial_overlay_step = 1
                self.tutorial_overlay_start_frame = int(self.play_time)
        else:
            if not self.tutorial_overlay_active:
                self.tutorial_slowmo_active = False
                self.tutorial_slowmo_start_frame = None
        
        if self.miss_pending:
            self.miss_timer += 1
            if self.miss_timer >= MISS_DELAY:
                self.enter_game_over()
            return
        
        # Update metronoom wanneer in gameplay
        self.update_metronome()
        
        if self.use_midi_schedule:
            self.game_frame += float(slowmo_mult)
            while self.scheduled_notes and self.scheduled_notes[0]['spawn_frame'] <= self.game_frame:
                s = self.scheduled_notes.pop(0)
                col = s['column']
                length = s.get('length_pixels')
                note = s.get('note')
                original_notes = s.get('original_notes') or [s.get('original_note', note)]
                self.tiles.append(Tile(col, length_pixels=length, note_num=list(original_notes), font=self.get_tile_font_for_column(col), label_text=self.get_label_for_column(col)))
        else:
            self.spawn_tiles()
        
        tutorial_freeze_notes = bool(
            self.tutorial_mode
            and self.tutorial_overlay_active
        )

        for tile in self.tiles[:]:
            # Pas slowmo toe door tijdelijk de speed te schalen
            orig_speed = tile.speed

            # Tutorial: noten vertragen (negatieve versnelling) tot halverwege,
            # daarna volledig stilzetten tot overlay weg is (na beide messages).
            effective_speed = float(orig_speed)
            if tutorial_freeze_notes and (not tile.hit) and (not tile.missed):
                if not hasattr(tile, 'tutorial_speed'):
                    tile.tutorial_speed = float(orig_speed)
                if not hasattr(tile, 'tutorial_base_speed'):
                    tile.tutorial_base_speed = float(orig_speed)

                half_y = float(SCREEN_HEIGHT) * 0.5
                center_y = float(tile.y) + (float(getattr(tile, 'height', 0)) * 0.5)

                if center_y < half_y:
                    decel = max(0.06, float(tile.tutorial_base_speed) * 0.08)
                    tile.tutorial_speed = max(0.0, float(tile.tutorial_speed) - decel)
                    effective_speed = float(tile.tutorial_speed)
                else:
                    tile.tutorial_speed = 0.0
                    tile.tutorial_frozen = True
                    effective_speed = 0.0
            else:
                # Tutorial overlay weg: herstel normale snelheid
                if hasattr(tile, 'tutorial_speed') or hasattr(tile, 'tutorial_frozen'):
                    try:
                        delattr(tile, 'tutorial_speed')
                    except Exception:
                        pass
                    try:
                        delattr(tile, 'tutorial_frozen')
                    except Exception:
                        pass
                    try:
                        delattr(tile, 'tutorial_base_speed')
                    except Exception:
                        pass

            tile.speed = float(effective_speed) * float(slowmo_mult)
            tile.update()
            tile.speed = orig_speed

            # Hold-notes: als je aan het holden bent, moet de toets ingedrukt blijven.
            if tile.holding:
                col = tile.column
                if (not self.easy_hold_enabled) and (col not in self.pressed_columns):
                    # Losgelaten: score op basis van release-window.
                    if self.active_holds.get(col) is tile:
                        self.score_hold_on_release(tile, col)
                    else:
                        tile.holding = False
                        self.stop_sustain_sound(col)
                else:
                    # Zodra de tile de bar "raakt" (kop onderaan), latch hem vast op de bar.
                    # Vanaf dit moment is de tile niet meer zichtbaar en kan hij niet missen.
                    if not tile.latched:
                        if (tile.y + tile.height) >= BAR_Y:
                            tile.latched = True

                            speed = max(1, int(tile.speed))
                            tile.hold_frames_total = max(1, int(math.ceil(tile.height / speed)))

                            # Great window: bestaande marge (HOLD_RELEASE_MARGIN_PX)
                            required_px_great = max(0, int(tile.height - HOLD_RELEASE_MARGIN_PX))
                            tile.hold_frames_required_great = max(0, int(math.ceil(required_px_great / speed)))
                            tile.hold_frames_required = tile.hold_frames_required_great

                            # Perfect window: kleinere marge (PERFECT_RELEASE_MARGIN_PX)
                            required_px_perfect = max(0, int(tile.height - PERFECT_RELEASE_MARGIN_PX))
                            tile.hold_frames_required_perfect = max(0, int(math.ceil(required_px_perfect / speed)))

                            tile.hold_frames_elapsed = 0

                            # Als de noot korter is dan de marge, is hij meteen "af".
                            if tile.hold_frames_total <= 1:
                                self.complete_hold(tile, col)
                    else:
                        tile.hold_frames_elapsed += 1
                        if tile.hold_frames_elapsed >= tile.hold_frames_total:
                            self.complete_hold(tile, col)
            
            if tile.hit and tile.fade_timer >= tile.fade_duration:
                self.tiles.remove(tile)
                continue
            
            # Voorkom dat een al-geraakte tile alsnog als "missed" wordt gezien.
            if (not tile.hit) and (not tile.missed) and (not tile.holding) and tile.is_missed():
                if self.cheats_enabled:
                    # Cheats: geen game over bij miss; ruim de tile wel op.
                    # Speel wel de noot af wanneer hij "verdwijnt" - met de correcte lengte!
                    self._reset_combo()
                    
                    # Bereken de juiste duur op basis van tile hoogte
                    # tile.height pixels / TILE_SPEED pixels per frame * (1/FPS) seconden per frame
                    duration_seconds = float(tile.height) / float(TILE_SPEED) / float(FPS)
                    # Minimale duur voor tap notes, maximale duur voor hold notes
                    duration_seconds = max(0.15, min(duration_seconds, SUSTAIN_SOUND_SECONDS))
                    
                    # Genereer geluid met de originele frequentie en juiste duur
                    # Gebruik caching strategie: afronden naar 0.1s nauwkeurigheid voor cache-hit ratio
                    cache_duration = round(duration_seconds * 10) / 10  # Afronden naar 0.1s
                    note_list = None
                    if isinstance(tile.note_num, (list, tuple, set)):
                        note_list = [int(n) for n in tile.note_num if n is not None]
                    elif tile.note_num is not None:
                        note_list = [int(tile.note_num)]
                    else:
                        note_list = [48 + tile.column]

                    for note_num in note_list:
                        cache_key = (int(note_num), cache_duration)
                        snd = self.midi_note_sustain_cache.get(cache_key)
                        if snd is None:
                            freq = get_frequency_for_midi_note(int(note_num))
                            snd = generate_piano_sustain_sound(freq, duration=cache_duration, volume=0.18)
                            self.midi_note_sustain_cache[cache_key] = snd
                        if snd is not None:
                            self._play_sound_on_channel(tile.column, snd)

                    if tile.holding:
                        self.active_holds.pop(tile.column, None)
                    self.tiles.remove(tile)
                    continue
                self._reset_combo()
                tile.missed = True
                self.miss_pending = True
                self.miss_timer = 0
                print(f"Missed! Score: {self.score}")

        # MIDI-level completion detectie: alle scheduled notes zijn gespawned en alles is opgeruimd.
        if self.use_midi_schedule:
            if (not self.scheduled_notes) and (not self.tiles) and (not self.active_holds) and (not self.miss_pending):
                if self.tutorial_mode:
                    # Tutorial klaar
                    self.tutorial_complete_frame = 0
                    self.tutorial_complete_bg = self._render_playfield_background()
                    self.tutorial_complete_bg_blur = self._blur_surface(self.tutorial_complete_bg)
                    self.state = STATE_TUTORIAL_COMPLETED
                else:
                    # Eerst een auto-replay vanaf het begin (geen input nodig).
                    if not self.auto_replay_done:
                        self.start_auto_replay()
                    else:
                        self.start_level_completed()
    
    def draw(self):
        if self.state == STATE_MENU:
            self.draw_menu()
        elif self.state == STATE_PLAY_SELECT:
            self.draw_play_select()
        elif self.state == STATE_PLAYING:
            self.draw_game()
        elif self.state == STATE_TRANSITION:
            self.draw_transition()
        elif self.state == STATE_PAUSED:
            self.draw_paused()
        elif self.state == STATE_COUNTDOWN:
            self.draw_countdown()
        elif self.state == STATE_GAME_OVER:
            self.draw_game_over()
        elif self.state == STATE_LEVELS:
            self.draw_levels()
        elif self.state == STATE_MIDI_INFO:
            self.draw_midi_info()
        elif self.state == STATE_SETTINGS:
            self.draw_settings()
        elif self.state == STATE_SHOP:
            self.draw_shop()
        elif self.state == STATE_LEVEL_COMPLETED:
            self.draw_level_completed()
        elif self.state == STATE_AUTO_REPLAY:
            self.draw_auto_replay()
        elif self.state == STATE_TUTORIAL_COMPLETED:
            self.draw_tutorial_completed()
        elif self.state == STATE_TUTORIAL_WELCOME:
            self.draw_tutorial_welcome()

    def draw_tutorial_welcome(self):
        self.screen.fill(DARK_GRAY)
        
        # Animatie: text valt van boven naar beneden
        # Progress: 0 tot 1
        progress = min(1.0, float(self.tutorial_welcome_frame) / float(max(1, self.tutorial_welcome_duration)))
        
        # Smooth easing: ease-out
        progress = 1.0 - (1.0 - progress) * (1.0 - progress)
        
        # Start position: -100 (boven scherm), eind: 200 (verticaal centrum area)
        start_y = -100
        end_y = 200
        current_y = int(start_y + (end_y - start_y) * progress)
        
        # Titel: "Welcome to Fallin' Keys!"
        title = self.font_xlarge.render("Welcome to Fallin' Keys!", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, current_y))
        self.screen.blit(title, title_rect)
        
        # Subtitle: "Press space to continue"
        # Dit verschijnt zachter/later
        if progress > 0.3:
            sub_progress = min(1.0, (progress - 0.3) / 0.7)
            alpha = int(255 * sub_progress)
            subtitle = self.font_large.render("Press space to continue", True, WHITE)
            subtitle.set_alpha(alpha)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, current_y + 80))
            self.screen.blit(subtitle, subtitle_rect)
        
        pygame.display.flip()

    def draw_play_select(self):
        self.screen.fill(DARK_GRAY)

        title = self.font_xlarge.render("Play", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 110))
        self.screen.blit(title, title_rect)

        self.free_play_button.draw(self.screen)
        self.tutorial_button.draw(self.screen)

        hint = self.font_small.render("Press ESC to go back", True, WHITE)
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
        self.screen.blit(hint, hint_rect)

        pygame.display.flip()

    def draw_auto_replay(self):
        # Auto replay: tiles vallen snel, en alles (ook de gele balken) wordt geblurred.
        # We doen een snelle single-pass blur per frame (400x600 is klein genoeg).

        # Render scene naar offscreen surface
        scene = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)).convert()
        scene.blit(self.background, (0, 0))

        for bar in self.bars:
            if bar.column not in BLACK_INDICES:
                bar.draw(scene)
        for bar in self.bars:
            if bar.column in BLACK_INDICES:
                bar.draw(scene)

        for tile in self.tiles:
            tile.draw(scene, show_green_hit=self.green_note_enabled)

        # Blur fade-in
        blur_frames = max(1, int(FPS * 0.9))
        p = min(1.0, float(self.auto_replay_frame) / float(blur_frames))
        p = p * p * (3 - 2 * p)

        # Fast blur (single pass downscale/upscale)
        if p > 0.001:
            w, h = SCREEN_WIDTH, SCREEN_HEIGHT
            scale = 0.20
            sw = max(1, int(w * scale))
            sh = max(1, int(h * scale))
            small = pygame.transform.smoothscale(scene, (sw, sh))
            blurred = pygame.transform.smoothscale(small, (w, h))

            # Blend blurred over scene
            out = scene.copy()
            blurred.set_alpha(int(255 * p))
            out.blit(blurred, (0, 0))

            # Extra dim voor betere blur leesbaarheid
            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            dim.set_alpha(int(120 * p))
            dim.fill((0, 0, 0))
            out.blit(dim, (0, 0))

            self.screen.blit(out, (0, 0))
        else:
            self.screen.blit(scene, (0, 0))

        # Level complete paneel komt tegelijk in beeld (voorgrond)
        self._draw_level_complete_panel(self.auto_replay_frame)

        pygame.display.flip()

    def _draw_level_complete_panel(self, frame: int, *, title_text: str = "Level completed!", hint_text: str = "Press SPACE to continue") -> None:
        # Paneel van boven laten vallen
        # Extra hoogte wanneer we sterren tonen.
        stars = getattr(self, 'level_stars', None)
        panel_w, panel_h = 320, (150 if (stars is not None) else 120)
        target_y = 110
        start_y = -panel_h - 10
        drop_frames = max(1, int(FPS * 0.6))
        drop_p = min(1.0, float(frame) / float(drop_frames))
        # ease-out
        drop_p = 1 - (1 - drop_p) * (1 - drop_p)
        panel_y = int(start_y + (target_y - start_y) * drop_p)
        panel_x = (SCREEN_WIDTH - panel_w) // 2

        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 160))
        pygame.draw.rect(panel_surf, (255, 255, 255, 210), panel_surf.get_rect(), 2, border_radius=10)

        title = self.font_large.render(str(title_text), True, WHITE)
        title_rect = title.get_rect(center=(panel_w // 2, 38 if (stars is not None) else 42))
        panel_surf.blit(title, title_rect)

        if stars is not None:
            # Stars: iets verder uit elkaar + 1-voor-1 fade-in
            drop_frames = max(1, int(FPS * 0.6))
            start_f = int(drop_frames * 0.65)
            stagger = max(1, int(FPS * 0.12))
            fade_frames = max(1, int(FPS * 0.22))
            alphas = []
            for i in range(3):
                f = int(frame) - int(start_f) - int(i * stagger)
                if f <= 0:
                    a = 0
                else:
                    t = min(1.0, float(f) / float(fade_frames))
                    # Ease-out
                    t = 1.0 - (1.0 - t) * (1.0 - t)
                    a = int(255 * t)
                alphas.append(max(0, min(255, int(a))))

            self._draw_star_row(
                panel_surf,
                stars_filled=int(stars),
                center_x=(panel_w // 2),
                center_y=78,
                size=16,
                gap_scale=1.15,
                alphas=alphas,
            )

        hint = self.font_small.render(str(hint_text), True, WHITE)
        hint_rect = hint.get_rect(center=(panel_w // 2, 112 if (stars is not None) else 82))
        panel_surf.blit(hint, hint_rect)

        self.screen.blit(panel_surf, (panel_x, panel_y))

    def _compute_level_star_rating(self) -> None:
        """Bereken 1/2/3 sterren op basis van het aantal noten in het level.

        - 3 sterren: score >= (alleen Perfects) = 3 * N
        - 2 sterren: score >= (alleen Greats)   = 2 * N
        - 1 ster   : score >= (alleen Allright) = 1 * N
        """
        n = 0
        if getattr(self, 'scheduled_notes_original', None):
            n = len(self.scheduled_notes_original)
        elif getattr(self, 'midi_notes', None):
            n = len(self.midi_notes)
        elif getattr(self, 'played_notes_history', None):
            n = len(self.played_notes_history)

        self.level_note_count = int(n)
        if n <= 0:
            self.level_score_thresholds = {'allright': 0, 'great': 0, 'perfect': 0}
            self.level_stars = None
            return

        thresholds = {
            'allright': int(n) * 1,
            'great': int(n) * 2,
            'perfect': int(n) * 3,
        }
        self.level_score_thresholds = thresholds

        s = int(getattr(self, 'score', 0))
        if s >= thresholds['perfect']:
            self.level_stars = 3
        elif s >= thresholds['great']:
            self.level_stars = 2
        elif s >= thresholds['allright']:
            self.level_stars = 1
        else:
            self.level_stars = 0

        # Bewaar best rating per level (niet voor tutorial)
        if (not getattr(self, 'tutorial_mode', False)) and getattr(self, 'current_level_id', None):
            lid = str(self.current_level_id)
            best = int(self.level_ratings.get(lid, 0))
            self.level_ratings[lid] = max(best, int(self.level_stars))

    def _draw_level_stars_next_to_button(self, surface: pygame.Surface, button_rect: pygame.Rect, stars: int) -> None:
        # Teken 3 sterren IN de level-balk, onder de level-tekst.
        stars = max(0, min(3, int(stars)))
        size = 11

        cx = int(button_rect.centerx)
        # Net boven de onderrand zodat het duidelijk "onder" de tekst staat.
        cy = int(button_rect.bottom - 16)

        self._draw_star_row(surface, stars_filled=stars, center_x=cx, center_y=cy, size=size)

    def _draw_star_row(self, surface: pygame.Surface, *, stars_filled: int, center_x: int, center_y: int, size: int, gap_scale: float = 0.90, alphas=None) -> None:
        stars_filled = max(0, min(3, int(stars_filled)))
        size = max(8, int(size))

        try:
            gap_scale = float(gap_scale)
        except Exception:
            gap_scale = 0.90
        gap_scale = max(0.20, min(2.50, gap_scale))

        gap = int(size * gap_scale)
        total_w = size * 3 + gap * 2
        left_x = int(center_x - total_w // 2 + size // 2)

        gold = (255, 210, 70)
        gold_outline = (255, 240, 190)
        dark = (85, 85, 85)
        dark_outline = (140, 140, 140)

        for i in range(3):
            cx = int(left_x + i * (size + gap))
            filled = (i < stars_filled)
            fill = gold if filled else dark
            outline = gold_outline if filled else dark_outline
            a = 255
            if alphas is not None:
                try:
                    a = int(alphas[i])
                except Exception:
                    a = 255
            a = max(0, min(255, int(a)))
            self._draw_star(surface, cx, int(center_y), size, fill, outline, alpha=a)

    def _draw_star(self, surface: pygame.Surface, cx: int, cy: int, r: int, fill_color, outline_color, *, alpha: int = 255) -> None:
        # 5-point star polygon
        alpha = max(0, min(255, int(alpha)))
        if alpha <= 0:
            return

        r_outer = float(r)
        r_inner = float(r) * 0.45

        # Draw on a small alpha surface so we can fade per-star.
        w = int(r * 2 + 6)
        h = int(r * 2 + 6)
        star_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        local_cx = w // 2
        local_cy = h // 2

        points = []
        for k in range(10):
            ang = (math.pi / 2.0) + (k * math.pi / 5.0)
            rad = r_outer if (k % 2 == 0) else r_inner
            x = local_cx + math.cos(ang) * rad
            y = local_cy - math.sin(ang) * rad
            points.append((int(x), int(y)))

        fr, fg, fb = fill_color
        or_, og, ob = outline_color
        pygame.draw.polygon(star_surf, (int(fr), int(fg), int(fb), alpha), points)
        pygame.draw.polygon(star_surf, (int(or_), int(og), int(ob), alpha), points, 2)

        surface.blit(star_surf, (int(cx - w // 2), int(cy - h // 2)))

    def _draw_game_scene(self):
        # Tekent de game-scene zonder pygame.display.flip()
        self.screen.blit(self.background, (0, 0))

        # Layering:
        # - Tiles die al geraakt zijn (hit + fade-out) tekenen we ACHTER de piano toetsen.
        # - Overige tiles blijven VOOR de toetsen (zoals normaal).
        for tile in self.tiles:
            if getattr(tile, 'hit', False):
                tile.draw(self.screen, show_green_hit=self.green_note_enabled)

        for bar in self.bars:
            if bar.column not in BLACK_INDICES:
                bar.draw(self.screen)

        for bar in self.bars:
            if bar.column in BLACK_INDICES:
                bar.draw(self.screen)

        # Key highlight: toets licht op terwijl je hem ingedrukt houdt.
        for col in self.pressed_columns:
            icol = int(col)
            if 0 <= icol < len(self.bars):
                if self.miss_pending and (self.wrong_key_column is not None) and (icol == int(self.wrong_key_column)):
                    strength = 1.0
                    if MISS_DELAY > 0:
                        strength = 1.0 - (float(self.miss_timer) / float(MISS_DELAY))
                    self._draw_wrong_key_highlight(icol, strength=strength)
                else:
                    self._draw_key_highlight(icol)

        for tile in self.tiles:
            if not getattr(tile, 'hit', False):
                tile.draw(self.screen, show_green_hit=self.green_note_enabled)

        # Smoke boven de piano (pixelated), NU voor de gele tiles.
        if self.smoke_particles:
            for p in self.smoke_particles:
                p.draw(self.screen)

        # Visual: wave boven de ingedrukte toets
        for col in self.pressed_columns:
            self.draw_wave(col)

        score_text = self.font_large.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        self._draw_judgements()

    def _draw_judgements(self):
        def draw_judgement_toast(text, alpha, top_y):
            a = max(0, min(255, int(alpha)))
            if a <= 0:
                return None

            if text == "Perfect":
                color = (0, 255, 120)
            elif text == "Great":
                color = (120, 200, 255)
            else:
                color = (255, 255, 255)

            strength = float(a) / 255.0
            j_text = self.font_large.render(str(text), True, color).convert_alpha()
            j_text.set_alpha(a)
            j_rect = j_text.get_rect(midtop=(SCREEN_WIDTH // 2, int(top_y)))
            pad_x, pad_y = 10, 6
            bg = pygame.Rect(
                j_rect.left - pad_x,
                j_rect.top - pad_y,
                j_rect.width + pad_x * 2,
                j_rect.height + pad_y * 2
            )

            panel = pygame.Surface((bg.width, bg.height), pygame.SRCALPHA)
            panel.fill((0, 0, 0, int(210 * strength)))
            pygame.draw.rect(panel, (255, 255, 255, int(255 * strength)), panel.get_rect(), 2)
            self.screen.blit(panel, (bg.left, bg.top))
            self.screen.blit(j_text, j_rect)
            return bg

        # Eerst: oude judgements die uitfaden (achter de huidige)
        if self.judgement_history:
            out_frames = max(1, int(JUDGEMENT_OUT_FRAMES))
            for item in list(self.judgement_history):
                t = min(1.0, float(item.get('age', 0)) / float(out_frames))
                # Ease-in-out
                t = t * t * (3.0 - 2.0 * t)
                alpha = int(int(item.get('alpha0', 255)) * (1.0 - t))
                y = int(item.get('y0', 10)) + int(float(JUDGEMENT_OUT_SLIDE_PX) * t)
                draw_judgement_toast(item.get('text', ''), alpha, y)

        # Daarna: huidige judgement (met multiplier)
        if self.last_judgement:
            # Snel & clean invagen + klein slide-je
            in_frames = max(1, int(JUDGEMENT_IN_FRAMES))
            t_in = min(1.0, float(self.last_judgement_age) / float(in_frames))
            # Ease-out
            t_in = 1.0 - (1.0 - t_in) * (1.0 - t_in) * (1.0 - t_in)
            alpha = int(255 * t_in)
            y_off = int((1.0 - t_in) * -float(JUDGEMENT_SLIDE_PX))

            bg = draw_judgement_toast(self.last_judgement, alpha, 10 + y_off)
            if bg is None:
                bg = pygame.Rect(0, 0, 0, 0)

            # Multiplier box naast judgement (x2, x3, ...)
            if self.combo_streak >= 2:
                mult_surf = self.font_large.render(f"x{self.combo_streak}", True, WHITE).convert_alpha()
                mult_surf.set_alpha(alpha)
                mult_w = max(54, mult_surf.get_width() + 18)
                mult_h = bg.height
                gap = 10
                mult_rect = pygame.Rect(bg.right + gap, bg.top, mult_w, mult_h)
                # Als er rechts geen plek is, teken links.
                if mult_rect.right > SCREEN_WIDTH - 8:
                    mult_rect.right = bg.left - gap

                mult_panel = pygame.Surface((mult_rect.width, mult_rect.height), pygame.SRCALPHA)
                mult_panel.fill((0, 0, 0, int(210 * (float(alpha) / 255.0))))
                mult_panel_alpha = max(0, min(255, int(alpha)))
                pygame.draw.rect(mult_panel, (255, 255, 255, mult_panel_alpha), mult_panel.get_rect(), 2)
                self.screen.blit(mult_panel, (mult_rect.left, mult_rect.top))
                mult_text_rect = mult_surf.get_rect(center=mult_rect.center)
                self.screen.blit(mult_surf, mult_text_rect)

    def _draw_piano_overlay(self):
        # Alleen de piano (toetsen + highlights + waves) opnieuw tekenen
        for bar in self.bars:
            if bar.column not in BLACK_INDICES:
                bar.draw(self.screen)

        for bar in self.bars:
            if bar.column in BLACK_INDICES:
                bar.draw(self.screen)

        for col in self.pressed_columns:
            icol = int(col)
            if 0 <= icol < len(self.bars):
                if self.miss_pending and (self.wrong_key_column is not None) and (icol == int(self.wrong_key_column)):
                    strength = 1.0
                    if MISS_DELAY > 0:
                        strength = 1.0 - (float(self.miss_timer) / float(MISS_DELAY))
                    self._draw_wrong_key_highlight(icol, strength=strength)
                else:
                    self._draw_key_highlight(icol)

        for col in self.pressed_columns:
            self.draw_wave(col)
            return bg

        self._draw_judgements()

        retry_hint = self.font_small.render("Press R to retry", True, WHITE)
        self.screen.blit(retry_hint, (10, 46))

        self.current_fps = self.clock.get_fps()
        fps_text = self.font_medium.render(f"FPS: {int(self.current_fps)}", True, WHITE)
        fps_rect = fps_text.get_rect()
        fps_rect.topright = (SCREEN_WIDTH - 10, 10)

        padding = 8
        background_rect = pygame.Rect(
            fps_rect.left - padding,
            fps_rect.top - padding,
            fps_rect.width + padding * 2,
            fps_rect.height + padding * 2
        )
        pygame.draw.rect(self.screen, (0, 0, 0, 180), background_rect)
        pygame.draw.rect(self.screen, WHITE, background_rect, 2)
        self.screen.blit(fps_text, fps_rect)

    def draw_wave(self, column):
        # Kleine "wave" net boven de bar van de ingedrukte toets.
        bar = self.bars[column]
        x0 = bar.x + 6
        x1 = bar.x + bar.width - 6
        if x1 <= x0:
            return

        base_y = BAR_Y - 14
        amp = 6
        t = pygame.time.get_ticks() / 140.0
        points = []
        steps = 14
        for i in range(steps + 1):
            x = x0 + (x1 - x0) * (i / steps)
            phase = (i / steps) * (math.pi * 2)
            y = base_y + math.sin(t + phase) * amp
            points.append((int(x), int(y)))

        color = (255, 255, 255) if column in BLACK_INDICES else (220, 240, 255)
        pygame.draw.lines(self.screen, color, False, points, 3)

    def _draw_key_highlight(self, column: int) -> None:
        bar = self.bars[column]

        # Kleur afhankelijk van zwarte/witte toets.
        is_black = (column in BLACK_INDICES)
        if is_black:
            fill = (120, 200, 255, 70)
            border = (180, 230, 255, 120)
        else:
            fill = (120, 200, 255, 90)
            border = (80, 180, 255, 160)

        overlay = pygame.Surface((bar.width, bar.height), pygame.SRCALPHA)
        overlay.fill(fill)
        pygame.draw.rect(overlay, border, overlay.get_rect(), 2)
        # Kleine extra “glow” bovenaan
        glow_h = max(6, int(bar.height * 0.18))
        glow = pygame.Surface((bar.width, glow_h), pygame.SRCALPHA)
        glow.fill((255, 255, 255, 40 if is_black else 55))
        overlay.blit(glow, (0, 0))

        self.screen.blit(overlay, (bar.x, bar.y))

    def _spawn_smoke_for_column(self, column: int, *, burst: bool) -> None:
        if not (0 <= int(column) < len(self.bars)):
            return
        bar = self.bars[int(column)]
        # Spawn net boven de bar
        base_y = float(bar.y) - 10.0
        n = int(SMOKE_SPAWN_BURST if burst else SMOKE_SPAWN_PER_FRAME)
        if n <= 0:
            return
        for _ in range(n):
            x = float(bar.x) + random.random() * float(max(1, bar.width - 1))
            y = base_y + random.uniform(-2.0, 2.0)
            self.smoke_particles.append(SmokeParticle(x, y))
    
    def draw_menu(self):
        self._render_menu_to(self.screen)
        pygame.display.flip()

    def draw_transition(self):
        if self.transition_from_surf is None or self.transition_to_surf is None:
            self.screen.fill(DARK_GRAY)
            pygame.display.flip()
            return

        p = self.transition_frame / max(1, self.transition_duration)
        p = max(0.0, min(1.0, p))
        # Ease-in-out
        p = p * p * (3 - 2 * p)

        w, h = SCREEN_WIDTH, SCREEN_HEIGHT

        # Push-swipe: beide schermen bewegen mee.
        if self.transition_dir == 'down':
            # Menu -> Settings: huidig scherm naar beneden, nieuw scherm van boven naar binnen.
            y_old = int(p * h)
            y_new = int(-h + p * h)
            self.screen.blit(self.transition_from_surf, (0, y_old))
            self.screen.blit(self.transition_to_surf, (0, y_new))

        elif self.transition_dir == 'up':
            # Settings -> Menu: huidig scherm naar boven, nieuw scherm van onder naar binnen.
            y_old = int(-p * h)
            y_new = int(h - p * h)
            self.screen.blit(self.transition_from_surf, (0, y_old))
            self.screen.blit(self.transition_to_surf, (0, y_new))

        elif self.transition_dir == 'right':
            # Menu -> Levels: huidig scherm naar rechts, nieuw scherm van links naar binnen.
            x_old = int(p * w)
            x_new = int(-w + p * w)
            self.screen.blit(self.transition_from_surf, (x_old, 0))
            self.screen.blit(self.transition_to_surf, (x_new, 0))

        elif self.transition_dir == 'left':
            # Levels -> Menu: huidig scherm naar links, nieuw scherm van rechts naar binnen.
            x_old = int(-p * w)
            x_new = int(w - p * w)
            self.screen.blit(self.transition_from_surf, (x_old, 0))
            self.screen.blit(self.transition_to_surf, (x_new, 0))

        else:
            self.screen.blit(self.transition_to_surf, (0, 0))

        pygame.display.flip()

    def draw_settings(self):
        self._render_settings_screen(self.screen)
        pygame.display.flip()

    def draw_shop(self):
        self._render_shop_screen(self.screen)
        pygame.display.flip()

    def draw_level_completed(self):
        # Achtergrond + blur fade-in (blur start pas nadat de "Level completed" overlay zichtbaar is)
        if self.level_complete_bg is None:
            self.level_complete_bg = self._render_playfield_background()
        self.screen.blit(self.level_complete_bg, (0, 0))

        if self.level_complete_bg_blur is not None:
            # Vanuit auto replay: blur is al volledig actief.
            if getattr(self, 'level_complete_from_auto_replay', False):
                p = 1.0
            else:
                # Laat de achtergrond geleidelijk blurren tijdens het level-complete effect.
                if self.level_complete_frame < LEVEL_COMPLETE_FALL_DELAY_FRAMES:
                    p = 0.0
                else:
                    blur_frames = max(1, int(FPS * 1.0))
                    f = max(0, int(self.level_complete_frame) - int(LEVEL_COMPLETE_FALL_DELAY_FRAMES))
                    p = min(1.0, float(f) / float(blur_frames))
                    p = p * p * (3 - 2 * p)

            blur = self.level_complete_bg_blur.copy()
            blur.set_alpha(int(255 * p))
            self.screen.blit(blur, (0, 0))

            # Extra subtiele dim zodat blur beter zichtbaar is.
            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            dim.set_alpha(int(120 * p))
            dim.fill((0, 0, 0))
            self.screen.blit(dim, (0, 0))

        # Vallende noten (effect) pas na korte delay (achter de overlay)
        if (not getattr(self, 'level_complete_from_auto_replay', False)) and self.level_complete_frame >= LEVEL_COMPLETE_FALL_DELAY_FRAMES:
            for t in self.celebration_tiles:
                t.draw(self.screen, show_green_hit=self.green_note_enabled)

        # Paneel op de voorgrond (met sterren)
        self._draw_level_complete_panel(self.level_complete_frame)

        pygame.display.flip()
    
    def draw_levels(self):
        # Gebruik dezelfde renderer als transitions zodat knoppen niet "verdwijnen".
        self._render_levels_to(self.screen)
        pygame.display.flip()
    
    def draw_midi_info(self):
        self.screen.fill(DARK_GRAY)
        
        title_text = self.font_xlarge.render("MIDI Analysis", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 20))
        self.screen.blit(title_text, title_rect)

        # Show less knop rechtsboven als alle noten zichtbaar zijn
        self.midi_info_show_less_rect = None
        if getattr(self, 'midi_info_show_all', False):
            show_less_text = self.font_small.render("Show less", True, WHITE)
            pad_x, pad_y = 6, 4
            btn_w = show_less_text.get_width() + pad_x * 2
            btn_h = show_less_text.get_height() + pad_y * 2
            # Plaats naast "Start Game" knop
            btn_x = int(self.play_now_button.rect.right + 10)
            btn_y = int(self.play_now_button.rect.centery - btn_h // 2)
            btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
            pygame.draw.rect(self.screen, (70, 70, 70), btn_rect)
            pygame.draw.rect(self.screen, WHITE, btn_rect, 2)
            text_rect = show_less_text.get_rect(center=btn_rect.center)
            self.screen.blit(show_less_text, text_rect)
            self.midi_info_show_less_rect = btn_rect

        scroll_top = 55
        scroll_bottom = SCREEN_HEIGHT - 80
        scroll_height = max(1, scroll_bottom - scroll_top)

        if not getattr(self, 'midi_info_show_all', False):
            self.midi_info_scroll = 0

        self.midi_info_scroll = max(0, min(
            int(getattr(self, 'midi_info_scroll', 0)),
            int(getattr(self, 'midi_info_max_scroll', 0))
        ))

        self.midi_info_more_rect_kept = None
        self.midi_info_more_rect_removed = None

        prev_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(0, scroll_top, SCREEN_WIDTH, scroll_height))

        y_pos = scroll_top - self.midi_info_scroll

        # Toon analyse statistieken
        analysis = getattr(self, 'midi_analysis_info', None)
        if analysis:
            bpm = analysis.get('bpm', 0)
            stats_text = self.font_small.render(
                f"BPM: {bpm:.1f} | Raw notes: {analysis['total_raw']} | After filter: {analysis['after_filter']} | Removed: {analysis['removed_count']}",
                True, (200, 200, 200)
            )
            self.screen.blit(stats_text, (10, y_pos))
            y_pos += 20

            # Toon verwijderde noten per reden
            if analysis['removed_notes']:
                removed_text = self.font_small.render("Removed notes:", True, WHITE)
                self.screen.blit(removed_text, (10, y_pos))
                y_pos += 18

                # Groepeer per reden
                by_reason = {}
                for rn in analysis['removed_notes']:
                    reason = rn['reason']
                    if reason not in by_reason:
                        by_reason[reason] = []
                    by_reason[reason].append(rn)

                # Toon verwijderde noten (scrollbaar bij "toon alles")
                max_removed_lines = 10
                removed_list = analysis['removed_notes']
                show_all = getattr(self, 'midi_info_show_all', False)

                line_budget = max_removed_lines
                if show_all:
                    line_budget = None

                if line_budget is not None:
                    for reason, notes_list in by_reason.items():
                        if line_budget <= 0:
                            break
                        reason_text = self.font_small.render(
                            f"  {reason}: {len(notes_list)} notes",
                            True, (255, 150, 150)
                        )
                        self.screen.blit(reason_text, (15, y_pos))
                        y_pos += 16
                        line_budget -= 1
                else:
                    for reason, notes_list in by_reason.items():
                        reason_text = self.font_small.render(
                            f"  {reason}: {len(notes_list)} notes",
                            True, (255, 150, 150)
                        )
                        self.screen.blit(reason_text, (15, y_pos))
                        y_pos += 16

                if show_all:
                    shown_removed = removed_list
                else:
                    shown_removed = removed_list[:max(0, line_budget)]

                for i, rn in enumerate(shown_removed):
                    original_notes = rn.get('original_notes', [])
                    note_names = [get_midi_note_name(n) for n in original_notes] if original_notes else []
                    duration_sec = round(rn.get('duration_seconds', 0.0), 3)
                    reason = rn.get('reason', 'Onbekend')

                    removed_line = self.font_small.render(
                        f"  {i+1}. {reason}: {', '.join(note_names)} ({duration_sec}s)",
                        True, (255, 120, 120)
                    )
                    self.screen.blit(removed_line, (15, y_pos))
                    y_pos += 16

                if (not show_all):
                    hidden_count = max(0, len(removed_list) - len(shown_removed))
                else:
                    hidden_count = 0

                if (not show_all) and hidden_count > 0 and (line_budget is None or line_budget > 0):
                    more_removed = hidden_count
                    bar_rect = pygame.Rect(10, y_pos, SCREEN_WIDTH - 20, 24)
                    pygame.draw.rect(self.screen, (70, 70, 70), bar_rect)
                    pygame.draw.rect(self.screen, (255, 150, 150), bar_rect, 2)
                    more_text = self.font_small.render(
                        f"... and {more_removed} more notes",
                        True, WHITE
                    )
                    text_rect = more_text.get_rect(center=bar_rect.center)
                    self.screen.blit(more_text, text_rect)
                    self.midi_info_more_rect_removed = bar_rect
                    y_pos += 28
                    if line_budget is not None:
                        line_budget -= 1

                y_pos += 5

        # Toon behouden noten
        info_text = self.font_small.render(f"Kept notes ({len(self.midi_notes)}):", True, WHITE)
        self.screen.blit(info_text, (10, y_pos))
        y_pos += 18

        show_all = getattr(self, 'midi_info_show_all', False)

        # Toon maximaal 10 noten zolang "meer" niet is aangeklikt
        if show_all:
            max_notes = len(self.midi_notes)
        else:
            max_notes = min(10, len(self.midi_notes))

        for i, note_data in enumerate(self.midi_notes[:max_notes]):
            original_notes = note_data.get('original_notes', [note_data['note']])
            note_names = [get_midi_note_name(n) for n in original_notes]
            start_frame = note_data.get('start_frame', 0)
            duration_frames = note_data.get('duration_frames', 0)

            start_sec = round(start_frame / FPS, 2)
            duration_sec = round(duration_frames / FPS, 2)

            col_num = note_data['note']
            note_text = self.font_small.render(
                f"{i+1}. Column {col_num}: {', '.join(note_names)} @ {start_sec}s (length: {duration_sec}s)",
                True,
                (255, 255, 0)
            )

            self.screen.blit(note_text, (15, y_pos))
            y_pos += 17

        if (not show_all) and len(self.midi_notes) > max_notes:
            more_kept = len(self.midi_notes) - max_notes
            bar_rect = pygame.Rect(10, y_pos, SCREEN_WIDTH - 20, 24)
            pygame.draw.rect(self.screen, (70, 70, 70), bar_rect)
            pygame.draw.rect(self.screen, WHITE, bar_rect, 2)
            more_text = self.font_small.render(f"... and {more_kept} more notes", True, WHITE)
            text_rect = more_text.get_rect(center=bar_rect.center)
            self.screen.blit(more_text, text_rect)
            self.midi_info_more_rect_kept = bar_rect
            y_pos += 28

        # Update scroll range
        content_height = max(0, y_pos - scroll_top)
        self.midi_info_max_scroll = max(0, int(content_height - scroll_height))
        if not getattr(self, 'midi_info_show_all', False):
            self.midi_info_max_scroll = 0
            self.midi_info_scroll = 0

        self.screen.set_clip(prev_clip)

        # Scrollbar (alleen bij "toon alles" en als er echt te scrollen valt)
        self.midi_info_scrollbar_rect = None
        self.midi_info_scrollbar_handle_rect = None
        if getattr(self, 'midi_info_show_all', False) and self.midi_info_max_scroll > 0:
            track_w = 12
            track_x = SCREEN_WIDTH - track_w - 6
            track_rect = pygame.Rect(track_x, scroll_top, track_w, scroll_height)
            pygame.draw.rect(self.screen, (60, 60, 60), track_rect)
            pygame.draw.rect(self.screen, (140, 140, 140), track_rect, 2)

            # Handle size is proportional to visible area
            handle_h = max(30, int(scroll_height * (float(scroll_height) / float(scroll_height + self.midi_info_max_scroll))))
            usable = max(1, scroll_height - handle_h)
            ratio = float(self.midi_info_scroll) / float(max(1, self.midi_info_max_scroll))
            handle_y = scroll_top + int(round(ratio * usable))
            handle_rect = pygame.Rect(track_x + 1, handle_y, track_w - 2, handle_h)
            pygame.draw.rect(self.screen, (200, 200, 200), handle_rect)

            self.midi_info_scrollbar_rect = track_rect
            self.midi_info_scrollbar_handle_rect = handle_rect
        
        if self.midi_valid:
            self.play_now_button.draw(self.screen)

        back_text = self.font_medium.render("Press ESC to go back", True, WHITE)
        back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 20))
        self.screen.blit(back_text, back_rect)

        pygame.display.flip()
    
    def draw_game(self):
        self._draw_game_scene()
        # Kleine blur tijdens tutorial slowmo
        if self.tutorial_mode and getattr(self, 'tutorial_slowmo_active', False):
            try:
                current_step = getattr(self, 'tutorial_overlay_step', 1)
                scene = self.screen.copy()
                
                # Bereken blur/dim intensiteit
                elapsed = 0
                if self.tutorial_slowmo_start_frame is not None:
                    elapsed = max(0, int(self.play_time) - int(self.tutorial_slowmo_start_frame))
                ramp_frames = max(1, int(FPS * 0.5))
                t = min(1.0, float(elapsed) / float(ramp_frames))
                # Ease-in-out
                t = t * t * (3 - 2 * t)
                
                # Step 1: piano zichtbaar (minder blur/dim), step 2: piano verborgen (meer blur/dim)
                if current_step == 1:
                    # Lichtere overlay zodat piano zichtbaar blijft
                    target_blur_alpha = getattr(self, 'tutorial_slowmo_blur_alpha', 90)
                    target_dim_alpha = 140
                else:
                    # Sterkere overlay om piano te verbergen bij "falling notes"
                    target_blur_alpha = 180
                    target_dim_alpha = 200
                
                # Smooth transitie bij step-wisseling via overlay_start_frame
                overlay_elapsed = 0
                if self.tutorial_overlay_start_frame is not None:
                    overlay_elapsed = max(0, int(self.play_time) - int(self.tutorial_overlay_start_frame))
                transition_frames = max(1, int(FPS * 0.6))
                step_t = min(1.0, float(overlay_elapsed) / float(transition_frames))
                step_t = step_t * step_t * (3 - 2 * step_t)  # Ease-in-out
                
                # Bij step 2: interpoleer van step1-waarden naar step2-waarden
                if current_step == 2:
                    old_blur = getattr(self, 'tutorial_slowmo_blur_alpha', 90)
                    old_dim = 140
                    blur_alpha_final = int(old_blur + (target_blur_alpha - old_blur) * step_t)
                    dim_alpha_final = int(old_dim + (target_dim_alpha - old_dim) * step_t)
                else:
                    blur_alpha_final = int(target_blur_alpha * t)
                    dim_alpha_final = int(target_dim_alpha * t)

                blur = self._blur_surface(scene)
                blur.set_alpha(blur_alpha_final)
                self.screen.blit(blur, (0, 0))

                dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
                dim.fill((0, 0, 0, dim_alpha_final))
                self.screen.blit(dim, (0, 0))

                # Piano alleen scherp tekenen bij step 1 (bij step 2 blijft piano verborgen)
                if current_step == 1:
                    self._draw_piano_overlay()
                elif current_step == 2:
                    # Teken de vallende noten opnieuw zodat ze niet blurred/donker zijn
                    for tile in self.tiles:
                        if not getattr(tile, 'hit', False):
                            tile.draw(self.screen, show_green_hit=self.green_note_enabled)

                # Text box: step-based tutorial message
                panel_w = int(SCREEN_WIDTH * 0.76)
                panel_h = 90
                panel_target_x = (SCREEN_WIDTH - panel_w) // 2
                panel_y = int(SCREEN_HEIGHT * 0.58)
                # Smooth fly-in from right
                start_x = SCREEN_WIDTH + 10
                elapsed = 0
                if self.tutorial_overlay_start_frame is not None:
                    elapsed = max(0, int(self.play_time) - int(self.tutorial_overlay_start_frame))
                fly_frames = max(1, int(FPS * 0.6))
                t = min(1.0, float(elapsed) / float(fly_frames))
                # Ease-out
                t = 1.0 - (1.0 - t) * (1.0 - t)
                panel_x = int(start_x + (panel_target_x - start_x) * t)
                # Groter tekstvak voor step 2
                if current_step == 2:
                    panel_w = int(SCREEN_WIDTH * 0.84)
                    panel_h = 130
                    panel_target_x = (SCREEN_WIDTH - panel_w) // 2
                    panel_x = int(start_x + (panel_target_x - start_x) * t)

                panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
                panel.fill((0, 0, 0, 180))
                pygame.draw.rect(panel, (255, 255, 255, 220), panel.get_rect(), 2)
                self.screen.blit(panel, (panel_x, panel_y))

                if current_step == 2:
                    title_font = self.font_large
                    line1 = "And these are fallin' notes."
                    line2 = "Try to play them correctly by pressing"
                    line3 = "the corresponding key on your keyboard."

                    txt1 = title_font.render(line1, True, WHITE)
                    txt2 = title_font.render(line2, True, WHITE)
                    txt3 = title_font.render(line3, True, WHITE)

                    # Als te breed: schaal alle regels omlaag met dezelfde factor
                    max_w = int(panel_w * 0.92)
                    max_line_w = max(txt1.get_width(), txt2.get_width(), txt3.get_width())
                    if max_line_w > max_w:
                        scale = max_w / float(max_line_w)
                        txt1 = pygame.transform.smoothscale(txt1, (int(txt1.get_width() * scale), int(txt1.get_height() * scale)))
                        txt2 = pygame.transform.smoothscale(txt2, (int(txt2.get_width() * scale), int(txt2.get_height() * scale)))
                        txt3 = pygame.transform.smoothscale(txt3, (int(txt3.get_width() * scale), int(txt3.get_height() * scale)))

                    txt1_rect = txt1.get_rect(center=(panel_x + panel_w // 2, panel_y + 24))
                    txt2_rect = txt2.get_rect(center=(panel_x + panel_w // 2, panel_y + 52))
                    txt3_rect = txt3.get_rect(center=(panel_x + panel_w // 2, panel_y + 80))
                    self.screen.blit(txt1, txt1_rect)
                    self.screen.blit(txt2, txt2_rect)
                    self.screen.blit(txt3, txt3_rect)
                else:
                    title_text = "This is your piano"
                    # Groter: benut het hele tekstvak
                    title_font = self.font_large

                    txt = title_font.render(title_text, True, WHITE)
                    # Als het te breed is, schaal iets omlaag
                    max_w = int(panel_w * 0.92)
                    if txt.get_width() > max_w:
                        scale = max_w / float(txt.get_width())
                        txt = pygame.transform.smoothscale(txt, (int(txt.get_width() * scale), int(txt.get_height() * scale)))

                    txt_rect = txt.get_rect(center=(panel_x + panel_w // 2, panel_y + 28))
                    self.screen.blit(txt, txt_rect)

                sub = self.font_small.render("Press space to continue", True, WHITE)
                sub_y = panel_y + (panel_h - 16)
                sub_rect = sub.get_rect(center=(panel_x + panel_w // 2, sub_y))
                self.screen.blit(sub, sub_rect)

                # Bouncing arrow pointing to the piano - only in step 1
                arrow_alpha = 0
                if current_step == 1:
                    arrow_delay = int(FPS * 0.35)
                    if overlay_elapsed >= arrow_delay:
                        arrow_t = min(1.0, float(overlay_elapsed - arrow_delay) / float(int(FPS * 0.5)))
                        arrow_alpha = int(255 * arrow_t)

                if arrow_alpha > 0:
                    arrow_x = SCREEN_WIDTH // 2
                    arrow_base_y = panel_y + panel_h + 8
                    bob = int(6 * math.sin(self.play_time * 0.25))
                    arrow_y = arrow_base_y + bob
                    arrow_color = (255, 255, 255, arrow_alpha)
                    arrow_w = 22
                    arrow_h = 14
                    # Draw arrow on alpha surface
                    arrow_surf = pygame.Surface((arrow_w + 4, arrow_h + 16), pygame.SRCALPHA)
                    cx = (arrow_w + 4) // 2
                    pygame.draw.line(arrow_surf, arrow_color, (cx, 0), (cx, 10), 2)
                    pygame.draw.polygon(
                        arrow_surf,
                        arrow_color,
                        [
                            (cx - arrow_w // 2, 10),
                            (cx + arrow_w // 2, 10),
                            (cx, 10 + arrow_h)
                        ]
                    )
                    self.screen.blit(arrow_surf, (arrow_x - (arrow_w + 4) // 2, arrow_y))
            except Exception:
                pass
        pygame.display.flip()

    def draw_tutorial_completed(self):
        # Blurred achtergrond + tutorial-complete paneel
        if self.tutorial_complete_bg is None:
            self.tutorial_complete_bg = self._render_playfield_background()
            self.tutorial_complete_bg_blur = self._blur_surface(self.tutorial_complete_bg)

        self.screen.blit(self.tutorial_complete_bg, (0, 0))
        if self.tutorial_complete_bg_blur is not None:
            blur = self.tutorial_complete_bg_blur.copy()
            blur.set_alpha(255)
            self.screen.blit(blur, (0, 0))

            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            dim.set_alpha(120)
            dim.fill((0, 0, 0))
            self.screen.blit(dim, (0, 0))

        self._draw_level_complete_panel(self.tutorial_complete_frame, title_text="Tutorial completed!", hint_text="Press SPACE to go back")
        self.tutorial_complete_frame += 1
        pygame.display.flip()

    def draw_paused(self):
        # Achtergrond: game zichtbaar maar blurred
        if self.pause_background_blurred is not None:
            self.screen.blit(self.pause_background_blurred, (0, 0))
        else:
            self._draw_game_scene()

        dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (0, 0))

        title = self.font_xlarge.render("Game Paused", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(title, title_rect)

        self.pause_continue_button.draw(self.screen)
        self.pause_settings_button.draw(self.screen)
        self.pause_return_button.draw(self.screen)

        hint = self.font_small.render("Press ESC to continue", True, WHITE)
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
        self.screen.blit(hint, hint_rect)

        pygame.display.flip()

    def draw_countdown(self):
        # Laat de game zien (niet blurred) met countdown overlay
        self._draw_game_scene()

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 90))
        self.screen.blit(overlay, (0, 0))

        num = self.font_xlarge.render(str(self.countdown_value), True, WHITE)
        num_rect = num.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(num, num_rect)

        pygame.display.flip()
    
    def draw_game_over(self):
        # Achtergrond snapshot
        if self.game_over_bg is not None:
            self.screen.blit(self.game_over_bg, (0, 0))
        else:
            self.screen.fill(DARK_GRAY)

        # Smooth blur fade-in
        if self.game_over_bg_blur is not None:
            blur_frames = max(1, int(FPS * 0.9))
            p = min(1.0, float(self.game_over_frame) / float(blur_frames))
            # Ease-in-out
            p = p * p * (3 - 2 * p)

            blur = self.game_over_bg_blur.copy()
            blur.set_alpha(int(255 * p))
            self.screen.blit(blur, (0, 0))

            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            dim.set_alpha(int(140 * p))
            dim.fill((0, 0, 0))
            self.screen.blit(dim, (0, 0))

        # Paneel zoals level complete
        hint = "Press ESC to go to main menu"
        self._draw_level_complete_panel(self.game_over_frame, title_text="Game Over", hint_text=hint)

        # Score pas laten invagen nadat het paneel in beeld is.
        panel_drop_frames = max(1, int(FPS * 0.6))
        score_delay = int(panel_drop_frames + int(FPS * 0.12))
        score_fade_frames = max(1, int(FPS * 0.30))

        f = int(self.game_over_frame) - int(score_delay)
        if f > 0:
            t = min(1.0, float(f) / float(score_fade_frames))
            # Ease-out
            t = 1.0 - (1.0 - t) * (1.0 - t)
            alpha = max(0, min(255, int(255 * t)))

            score_text = self.font_large.render(f"Score: {self.score}", True, WHITE).convert_alpha()
            score_text.set_alpha(alpha)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 280))
            self.screen.blit(score_text, score_rect)

        pygame.display.flip()
    
    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        # Cleanup (ook als er een device verbonden is)
        self._close_midi_input()
        self._close_midi_output()
        try:
            if pgmidi is not None:
                pgmidi.quit()
        except Exception:
            pass

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = Game()
    game.run()