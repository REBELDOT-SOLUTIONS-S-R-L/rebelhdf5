## Dataset tooling — rebelHDF5

The training pipeline runs entirely on HDF5 files (Isaac Lab + MimicGen output).
Open-source MyHDF5 was a good starting point for raw inspection but didn't fit
robotics workflows, so we forked it into **rebelHDF5** and added the features
we actually needed to iterate on dataset quality.

Project link:
https://github.com/alex-luci/myhdf5/tree/pose-trace-integration

![rebelHDF5 viewer](screenshots/rebelhdf5-viewer.png)

*Main view: opened-files panel on the far left, HDF5 group/demo tree, and a tabular view of the selected dataset (here, an `actions` array). The same chrome wraps every other tool below.*

- **Pose Trace.** A 3D viewer that renders the EEF trajectory as a continuous
  path alongside the garment keypoints over time, so failed grasps, drift, or
  misaligned trajectories are spotted at a glance instead of from raw arrays.
  A second tab plots the supporting metrics — EEF-to-keypoint distances,
  garment fold-form distances, and EEF-vs-keypoint heights over the episode.

  ![Pose Trace — 3D EEF + keypoint paths](screenshots/rebelhdf5-pose-trace-1.png)

  *3D Pose Trace: left-arm (blue) and right-arm (red) EEF paths plotted with the six garment keypoints over time. The bottom slider scrubs through the episode step-by-step.*

  ![Pose Trace — per-step distance/height analytics](screenshots/rebelhdf5-pose-trace-2.png)

  *Pose Trace analytics tab: per-step EEF-to-keypoint distances (top), garment fold-form distances (middle), and EEF-vs-keypoint heights (bottom), broken out for the left and right arms.*

- **Video Converter.** Extracts the camera streams from an HDF5 episode and
  produces an MP4 directly in the browser — no external scripts, click-to-preview
  qualitative review of any sample.

  ![Video Converter](screenshots/rebelhdf5-video-converter.png)

  *Video Converter: top-down camera stream extracted from the selected episode and played back inline at native FPS, with a Save Video button to export an MP4.*

- **Data Processing.** In-browser cut / merge / append operations on episodes,
  with key-level selection so you can keep only the parts of an episode you
  want. Bad rollouts get pruned, multiple generation runs are merged, new
  batches are appended — all without leaving the viewer.

  ![Data Processing](screenshots/rebelhdf5-data-processing.png)

  *Data Processing: Cut operation across an episode range (`demo_0`–`demo_99`) with key-level selection — pick exactly which dataset paths (e.g. `initial_state/articulation/left_arm/joint_position`) survive into the output file.*

- **Cloth Distribution Analysis.** Scatter of init poses (success vs failed
  overlaid on the teleop set) plus failure-rate heatmaps over the (pos_x,
  pos_y) and (rot_x, rot_y) init space. This is what closes the loop with the
  Halton sampler — failure-prone regions become directly visible and can be
  re-sampled. Clicking any generated demo on the scatter draws an arrow back
  to the source teleop demo MimicGen pulled from, so failure clusters can be
  traced to specific source demonstrations.

  ![Init-pose scatter (success vs failed)](screenshots/rebelhdf5-cloth-distribution-xyscatter.png)

  *Init-pose scatter: every generated demo plotted on the (init_pose_x, init_pose_y) plane, color-coded as success (green), failed (red), or teleop source (blue), so coverage and failure clustering are visible at a glance.*

  ![Click-to-trace source demo on scatter](screenshots/rebelhdf5-cloth-distribution-xyscatter-selection.png)

  *Click-to-trace: selecting a generated demo on the scatter draws arrows to the source teleop demo(s) MimicGen pulled from, with a tooltip listing the source path — so a failure cluster can be tied back to specific source demonstrations.*

  ![Failure-rate heatmap over init position](screenshots/rebelhdf5-cloth-distribution-positionmap.png)

  *Failure-rate heatmap over the init-position space (pos_x, pos_y), aggregated from generated demos via nearest-neighbor pooling. Hotter cells = higher failure rate — directly drives where to oversample next.*

  ![Failure-rate heatmap over init rotation](screenshots/rebelhdf5-cloth-distribution-rotationmap.png)

  *Same heatmap projection over the init-rotation space (rot_x, rot_y), exposing orientation regimes the policy struggles with independently of position.*

- **Databricks integration.** Single interface for uploading/downloading
  datasets, managing secrets, and triggering pipelines, so local iteration and
  cloud runs share one workflow.

  ![Databricks page — top](screenshots/rebelhdf5-databricks-1.png)

  *Screenshot 1 (Databricks page, top): the secrets and job-control section, where API tokens, the Databricks workspace URL, and rollout-step settings are pushed as scoped secrets; the same page also uploads the opened HDF5 file and triggers named jobs such as synthetic-data generation, HDF5→LeRobot conversion, and training.*

  ![Databricks page — bottom](screenshots/rebelhdf5-databricks-2.png)

  *Screenshot 2 (Databricks page, bottom): the Volume Browser, which lets you inspect arbitrary Databricks volume paths such as `trained_models/` and pull selected files back to a configurable local download directory with one click.*

Net effect: HDF5s stopped being opaque archives and became interactive dataset
assets, which is what made the synthetic pipeline above tractable to debug at
scale.

## DATASET STRUCTURE

<style>
.tree {
  background: #ffffff00;
  color: #e5e7eb;
  padding: 16px 20px;
  border-radius: 10px;
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 13px;
  line-height: 1.5;
  overflow-x: auto;
  border: 1px solid #ffffff00;
}

.tree .comment {
  color: #6b7280;
  margin-left: 8px;
}

.tree .folder {
  color: #93c5fd;
}

.tree .file {
  color: #e5e7eb;
}
</style>

<pre class="tree">
<span class="folder">data/</span><span class="comment"># attrs: schema_version, fps, env_args, applied_action, num_episodes, total, description</span>
├── <span class="folder">demo_&lt;N&gt;/</span><span class="comment"># attrs: num_samples, success, random_seed</span>
│   ├── <span class="folder">actions/</span>
│   │   ├── <span class="file">pose</span><span class="comment"># (T, A_pose) EEF pose + gripper actions; attrs: frame, format, quat_order, entity_order</span>
│   │   └── <span class="file">joints</span><span class="comment"># (T, total_joints) joint-space actions; attrs: units, joint_order, entity_order</span>
│   │
│   ├── <span class="folder">initial_state/</span>
│   │   ├── <span class="folder">articulation/</span>
│   │   │   ├── <span class="folder">articulation_name/</span>
│   │   │   │   ├── <span class="file">joint_position</span><span class="comment"># (1, J)</span>
│   │   │   │   ├── <span class="file">joint_velocity</span><span class="comment"># (1, J)</span>
│   │   │   │   ├── <span class="file">root_pose</span><span class="comment"># (1, 7)</span>
│   │   │   │   └── <span class="file">root_velocity</span><span class="comment"># (1, 6)</span>
│   │   │
│   │   └── <span class="folder">objects/</span>
│   │       └── <span class="folder">object_name/</span>
│   │           ├── <span class="file">initial_pose</span>
│   │           └── <span class="file">scale</span>
│   │
│   ├── <span class="folder">obs/</span><span class="comment"># measured observations; attrs: sample_phase</span>
│   │   ├── <span class="folder">articulation/</span>
│   │   │   └── <span class="folder">articulation_name/</span>
│   │   │       ├── <span class="file">joint_position</span><span class="comment"># (T, J)</span>
│   │   │       └── <span class="file">joint_velocity</span><span class="comment"># (T, J)</span>
│   │   │
│   │   ├── <span class="folder">end_effectors/</span>
│   │   │   └── <span class="folder">end_effector_name/</span>
│   │   │       └── <span class="file">pose</span><span class="comment"># (T, 4, 4)</span>
│   │   │
│   │   ├── <span class="folder">objects/</span>
│   │   │   └── <span class="folder">object_name/</span>
│   │   │       └── <span class="file">pose</span><span class="comment"># (T, 4, 4)</span>
│   │   │
│   │   ├── <span class="folder">cameras/</span>
│   │   │   └── <span class="file">camera_name</span><span class="comment"># (T, H, W, C)</span>
│   │   │
│   │   ├── <span class="folder">sensors/</span>
│   │   │   └── <span class="folder">sensor_name/</span>
│   │   │       └── <span class="file">field_name</span>
│   │   │
│   │   └── <span class="folder">datagen_info/</span><span class="comment"># MimicGen-compatible; attrs: sample_phase, aligned_to</span>
│   │       ├── <span class="folder">eef_pose/</span>
│   │       │   └── <span class="file">end_effector_name</span><span class="comment"># (T, 4, 4)</span>
│   │       ├── <span class="folder">target_eef_pose/</span>
│   │       │   └── <span class="file">end_effector_name</span><span class="comment"># (T, 4, 4)</span>
│   │       ├── <span class="folder">object_pose/</span>
│   │       │   └── <span class="file">object_name</span><span class="comment"># (T, 4, 4)</span>
│   │       └── <span class="folder">subtask_term_signals/</span>
│   │           └── <span class="file">signal_name</span><span class="comment"># (T, 1)</span>
│   │ 
│   └── <span class="folder">reference_demo_indices/</span><span class="comment"># (num_subtasks,) source demo index MimicGen selected for each subtask</span>
│       └── <span class="file">articulation_name</span>
</pre>
## Attribution

This project is based on work originally developed by the European Synchrotron Radiation Facility.

Original project:
- Copyright (c) 2022 European Synchrotron Radiation Facility
- Licensed under the MIT License

Modifications and extensions in this repository include additional features such as:
- Pose trajectory visualization
- HDF5 to video conversion tools
- Dataset processing utilities (merge, split, append)
- Cloth distribution analysis tools

All original code is used in accordance with the terms of the MIT License. A copy of the original license is included in this repository
