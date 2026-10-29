# Thesis platform backup

This snapshot is under tracknet-thesis/ on the dedicated backup/tracknet-thesis-platforms-20261030 branch.
The AOI repository main branch and pre-existing files are not replaced.

Includes thesis PC and Nano source code, upstream source, config, documentation, tests, PowerPoint,
root thesis figures/tables, small experiment summaries, Nano original-source archive and compatibility diff.
The Nano's deployed upstream source is intentionally not overwritten by the PC copy.

Excluded: .git, dataset/, raw/, outputs/, weights/, checkpoints, ONNX and TensorRT engines,
generated videos, temporary event images/caches, and text result files larger than 5 MiB.
This is a source/document backup, not a complete dataset or trained-model backup.

PC runtime: python -m thesis.platforms.pc.student_realtime_events --help
Nano runtime: python3 -m thesis.platforms.jetson_nano.runtime.demo_with_gt --help
Run each platform in its existing environment. See docs/platforms.md.

Snapshots were syntax checked; the PC regression suite passed 85 tests.
Nano inference accuracy/FPS was not re-benchmarked during source reorganization.

Use the dedicated backup checkout for commits and pushes; the live working projects retain their upstream Git histories.
No credentials or private keys are included. Never push the live upstream branch into the AOI main branch.
