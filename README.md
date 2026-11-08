# shorts-engine
Python + numpy + Pillow + ffmpeg. No Blender, no GPU, no paid API, no AI calls.

## 5-agent pipeline (python shorts.py auto)
1 Strategist  : picks the next unused pair (ledger.json), ranking = familiarity + ratio sweet spot (unvalidated heuristic)
2 Researcher  : recomputes diameter/volume ratios from data/bodies.json, checks ranges, builds on-screen texts
3 Director    : storyboard (hook / zoom-out / fill / payoff timings, camera scales, palette)
4 Engine      : renders 1080x1920 30fps on CPU (3D-shaded spheres, bloom, audio synthesized in numpy)
5 Publisher   : QC (ffprobe + blank-frame check), YouTube title/description/tags (.txt), ledger update

## Adding bodies
Append to data/bodies.json. Required: name, diam_km, color_hex. Optional: plural, type (rock|gas|ice|earth|star),
familiarity 1-5, vol_km3 (enables a second-source volume check). No code changes needed.
Valid pairs need a diameter ratio of 8-120 (MIN_RATIO/MAX_RATIO in shorts.py).

## Other engines
python shorts.py illusion | physics | all     (independent of the ledger)

## GitHub
Actions tab -> Render Short -> Run workflow. Download the artifact (mp4 + txt). The ledger is committed back automatically.
