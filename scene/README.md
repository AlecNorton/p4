# The A4 scene

`a4`: the net with one window panel standing about 2.9 m in front of the
start and facing it, registered to the Vicon frame and stored in it. It is the
scene you develop in and the scene of the demo: the real window stands where
it is here. Load it with `SceneBundle.load("scene")`; `manifest.json` lists its
files.

| file | what it is |
|---|---|
| `a4_cleaned_splat.splat` | the Gaussian splat: what the camera sees |
| `a4_esdf.npy` | distance field: clearance in metres at any point; scores collisions |
| `a4_flight_pc.ply` | the same geometry as a point cloud, for plots |
| `gates.json` | the gap's opening, to score the run: your flight code may not read it |
| `task.json` | the start, Vicon metres |
| `policy_contract.json` | the scene's camera; `make_contract.py` builds yours from it |
| `vicon_transform.json` | Vicon metres → splat units: here only a scale |
| `anchors.json` | the floor markers the registration was fitted to |
| `dataparser_transforms.json`, `splat_frame.json` | the capture's own guess at its scale, and how its frame was levelled |
| `capture_transforms.json`, `nerfstudio_config.yml` | the capture's cameras and training config, for the GPU renderer |

The window is a foamcore panel with a stone-wall print, standing on two
poles. It has one hole the drone fits through and a few smaller ones it does
not.

The registration is accurate to 29 mm RMS over six floor markers, and to
10–20 mm at the two beside the window. `check()` reports one warning: the
capture's own guess at its scale (`dataparser_transforms.json`) is 3.0% off
the scale measured against Vicon. This scene uses the measured scale, so
nothing needs fixing.
