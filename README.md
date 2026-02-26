# 🎹 Fallin' Keys - Full Featured HTML5 Piano Tiles Game

Complete HTML5 conversie van de Python/Pygame Magic Tiles game met alle features.

## 🎮 Hoe te spelen

1. **Open** `index.html` in een moderne webbrowser
2. **Klik** "Play" om te beginnen
3. **Druk** de juiste toets wanneer een tile de witte balk bereikt!

## ✨ Alle Features

### Menu System
- **Play** - Free Play of Tutorial mode
- **Levels** - Voorgedefinieerde levels met sterren rating
- **Shop** - Themes, Backgrounds, Instruments (unlock systeem)
- **Settings** - Gameplay, Visuals, Audio, MIDI configuratie

### Gameplay Features
- **12 kolommen piano layout** (witte en zwarte toetsen)
- **Hold notes** (lange noten vasthouden)
- **Judgement systeem**: Perfect / Great / Allright
- **Combo multiplier**
- **Level completion** met sterren (★★★)
- **Tutorial mode**
- **Pause functie** (ESC of pause knop)

### Settings (4 tabs)
- **Gameplay**: Key labels (Keys/Note names/Straight), Cheats, Easy Hold
- **Visuals**: Blue/Green hit effect
- **Audio**: Sound on/off, Volume, Metronome
- **MIDI**: Connect MIDI keyboard (Web MIDI API)

### Shop (3 tabs)
- **Themes**: Visuele thema's
- **Background**: Achtergrond stijlen  
- **Instruments**: Piano klanken

### Input Methods
- **Keyboard**: A, W, S, E, D, C, U, J, I, K, O, L
- **Straight keys**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -, =
- **Touch**: Tap op piano toetsen (mobile support)
- **MIDI keyboard**: Sluit een externe piano aan

### MIDI File Support
- Upload eigen .mid/.midi bestanden
- Parser voor standaard MIDI format
- Automatische note spawning

## 🎯 Controls

| Modus | Toetsen |
|-------|---------|
| Keys | A W S E D C U J I K O L |
| Note Names | C3 C#3 D3 D#3 E3 F3 F#3 G3 G#3 A3 A#3 B3 |
| Straight | 1 2 3 4 5 6 7 8 9 0 - = |
| Andere | ESC = Pause |

## 🏆 Scoring

- **Perfect**: 20 punten + (combo × 5)
- **Great**: 15 punten + (combo × 5)
- **Allright**: 10 punten + (combo × 5)
- **Hold notes**: Extra punten voor vasthouden

## 📁 Project Structure

```
fallingkeysweb/
├── index.html          # Complete game (HTML + CSS + JavaScript)
├── README.md           # Dit bestand
└── .github/
    └── copilot-instructions.md
```

## 🛠️ Technical Details

- **No Dependencies** - Pure vanilla JavaScript
- **Canvas API** - Rendering
- **Web Audio API** - Piano sounds generatie
- **Web MIDI API** - MIDI keyboard support (optioneel)
- **60 FPS** - Smooth game loop
- **Touch Events** - Mobile support
- **LocalStorage ready** - Voor save data (uitbreidbaar)

## 🚀 Getting Started

Open `index.html` in je browser:

```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

## 📱 Browser Support

- Chrome (aanbevolen - beste MIDI support)
- Firefox
- Safari
- Edge
- Mobile browsers (iOS Safari, Chrome for Android)

## 🎨 Vergelijking met Python versie

| Feature | Python (Pygame) | HTML5 (Web) |
|---------|-----------------|-------------|
| Menu System | ✅ | ✅ |
| Play/Levels/Shop/Settings | ✅ | ✅ |
| 12 Column Piano | ✅ | ✅ |
| Hold Notes | ✅ | ✅ |
| Judgement System | ✅ | ✅ |
| MIDI File Upload | ✅ | ✅ |
| MIDI Device Input | ✅ | ✅ (Web MIDI) |
| Piano Sounds | ✅ (Pygame mixer) | ✅ (Web Audio) |
| Smoke Particles | ✅ | ❌ (performance) |
| System Synth | ✅ (Windows) | ❌ (Web limitation) |

## 📄 License

MIT License - Vrij te gebruiken en aan te passen!

---

Made with pure programming skills - Full featured HTML5 conversion of Fallin' Keys
