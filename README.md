# P4 starter pack

The assignment page is the specification.

```
p4_starter/
├── install.md              set up first
├── scene/                  the A4 scene, a4: the window panel, and the scene of the demo
├── Blender/                four window shapes to test your gap finder on (the page, section 5)
├── Code/                   yours: copy into your own folder and edit
│   ├── Wrapper.py          flies your policy in the simulator (the file we run)
│   ├── policy_template.py  gap finder, state machine and controller: fill in the TODOs
│   └── fpv.py              records what the camera saw, for FPVVideo.mp4
└── tools/                  ours: use them, do not submit them
    ├── check_setup.py      checks your setup
    ├── make_contract.py    writes p4_contract.json: your camera and action
    ├── fly.py              ros_node with the depth taken out: fly with this
    ├── render_views.py     renders the camera's view at poses you pick
    ├── cpu_renderer.py     renders the splat without a GPU, slowly
    └── splat_cpu.py        the maths behind it
```

## Steps

Keep this folder next to your own (`~/rbe595/p4_starter`, `~/rbe595/p4_Group1`).
In this folder:

```bash
python3 tools/make_contract.py    # 1. write p4_contract.json; --width sets the resolution
python3 tools/check_setup.py      # 2. check the setup
```

Copy `Code/` into your folder, fill in the TODOs, and in your folder fly it in
the simulator with one of:

```bash
python3 Code/Wrapper.py --worker-port 7790        # 3. on the lab PC, with the render server running
python3 Code/Wrapper.py --worker-python PYTHON    #    on your own GPU (Turing: install.md)
python3 Code/Wrapper.py --cpu                     #    anywhere, at seconds per frame
```

It prints whether you went through the gap and how long it took, and writes
`run.json`, `run.png` and `fpv/<run>/`.

To fly, in the lab container, with the render server running: first the hover
dry run `check_setup.py` prints, then this from your folder, then again without
`--dry-run` (full procedure: the course site's Deployment page):

```bash
PYTHONPATH=Code:$PYTHONPATH python3 ../p4_starter/tools/fly.py \
  --topic       /vicon/<object>/<object> \
  --contract    ../p4_starter/p4_contract.json \
  --transform   ../p4_starter/scene/vicon_transform.json \
  --gates       ../p4_starter/scene/gates.json \
  --esdf        ../p4_starter/scene/a4_esdf.npy \
  --clearance-m 0.05 \
  --worker-port 7790 \
  --policy      policy_template:GapFlyer \
  --hold-altitude-m 0.7 \
  --log         run.json \
  --dry-run
```

`fly.py` is `ros_node` with one change: your policy gets the RGB image and no
depth, as in `Wrapper.py`. Every flag is `ros_node`'s. A run ends after 2
minutes.

## Notes

- **RGB only.** `obs.rgb` is the camera image and `obs.depth_m` is `None`, in
  `Wrapper.py` and on the drone through `fly.py`. Your contract has no depth
  either: `make_contract.py` writes an RGB one, and both refuse any other.
- **Where the gap is** comes from your images. Do not read `gates.json` or the
  gate fields of `state` (`Wrapper.py` and `fly.py` hand your policy a state
  without them; `--gates` is only there to score the run), and do not hard-code
  where the window or the gap is: this scene is also the demo's.
- **Training data.** Only synthetic images you make: `Blender/` has the window
  scene, four window shapes and four textures to start from. Anything rendered
  from the splat (`Wrapper.py`, `render_views.py`, the FPV frames) is for
  testing.
- **Frames.** `state.position_m` is in scene metres. Our scenes are stored in
  the Vicon frame: origin at the L tape, x down the course, z up from the
  floor. Your velocity goes out in the body frame.
- **Timing.** The loop runs at 15 Hz. On the lab PC, rendering and `act()`
  share each 66 ms frame; `Wrapper.py` prints how long `act()` took.
- **FPV video.** Each run leaves `fpv/<run>/frame_*.png`, also after
  `fly.py`: the camera image, and the same image with your detection drawn on.
  Make the video with
  `ffmpeg -framerate 15 -i fpv/<run>/frame_%05d.png -pix_fmt yuv420p FPVVideo.mp4`.
