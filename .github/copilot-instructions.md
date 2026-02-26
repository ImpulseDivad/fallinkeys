# Fallin' Keys Web Game

Volledige HTML5 conversie van de Python/Pygame Magic Tiles game.

## Project Structure
- `index.html` - Complete game met alle HTML, CSS, en JavaScript
- `README.md` - Project documentatie

## Features (volledig geconverteerd)

### Menu System
- Play → Free Play / Tutorial
- Levels → Level 1 (Sunrise), Level 2 (Green Hills), MIDI Upload
- Shop → Themes / Background / Instruments tabs
- Settings → Gameplay / Visuals / Audio / MIDI tabs

### Gameplay
- 12 kolommen piano-style layout (witte en zwarte toetsen)
- Hold notes (lange noten)
- Judgement systeem: Perfect / Great / Allright
- Combo multiplier met score berekening
- Level completion met sterren rating (★★★)
- Tutorial mode
- Pause functie (ESC)

### Settings Options
- Key labels: Keys / Note names / Straight keys
- Cheats mode
- Easy Hold mode  
- Blue/Green hit effect
- Sound on/off + Volume
- Metronome
- MIDI device connection (Web MIDI API)

### Input Support
- Keyboard: A W S E D C U J I K O L
- Straight keys: 1 2 3 4 5 6 7 8 9 0 - =
- Touch support (mobile)
- MIDI keyboard (Web MIDI API)
- MIDI file upload (.mid/.midi)

## How to Run
Open `index.html` in een moderne webbrowser (Chrome aanbevolen voor MIDI support).

## Technical Details
- Pure vanilla JavaScript (geen dependencies)
- Canvas API voor rendering
- Web Audio API voor piano sounds
- Web MIDI API voor MIDI keyboard support
- 60 FPS game loop
- Touch events voor mobile
- ~1200 regels code
