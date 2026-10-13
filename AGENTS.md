# P4 repository rules

- Start by reading this file, docs/project_handoff.md, README.md and the current task package; check pwd and git status. Preserve user work.
- Current scope is one task at a time. Never automatically progress to the next task. A roadmap is not implementation authorization.
- Use Sprint Learning Mode: Problem → Intuition → Concepts → Mathematics → Architecture → Implementation → Experiment → Result → Failure Analysis → Interview Notes.
- Engineering Complete (Code / Experiment / Docs) is separate from Learning Mastered (Run / Modify / Explain). README is the status source. Only explicit user reports can check learning boxes; assistant execution does not count.
- CPU-first on Ryzen 7 7840HS / Windows + WSL2 Ubuntu 24.04, no NVIDIA assumption. Prefer small explicit NumPy or CPU PyTorch implementations. No CUDA, large model training or cloud-first defaults.
- Keep dependencies task-local and minimal in requirements.txt. Use a repository .venv or a separately verified dedicated environment; never install into Conda base or system Python. Check interpreter, dependency versions and actual module paths before execution. Prefer Python 3.11/3.12 for future ML compatibility; record any different tested version.
- Explain dependency repairs and dataset download size/purpose/necessity before downloading. Do not automatically download GB/TB datasets or large model weights.
- No hidden black-box learning: explain modality, tensor shape, dtype, units, coordinate frame, temporal meaning and module/data flow.
- Every learned model needs evaluation. Offline loss != robot success. Report closed-loop success, failure stages and latency where applicable. Paper claims != our experimental evidence. Do not overclaim toy generalization or reproduction fidelity.
- Keep simple readable code; no speculative factory, registry, plugin framework, complex config or layered wrappers.
- The sibling mujoco-robotics-playground is read-only context. Do not modify, copy, submodule or import its internal modules. Connect through versioned dataset/file contracts. Old AGENTS.md applies to the old repository, not this one.
- Preserve raw physical data; fit normalization on training episodes only. Never silently reinterpret action semantics, time alignment, frame or embodiment.
- Major learning packages use the sections in docs/vla/s0_1_data_contract.md, including Expected vs Actual Result, limitations, failure cases, interview capsule and learner verification.
- After meaningful work update docs/project_handoff.md with exact commands/results, environment, limitations, Current Task and Next Task. Keep curated reasoning, not private thought transcripts.
- Run appropriate validation and git diff --check. Do not claim unrun checks or mark mastery from tests. Generated datasets, environments and logs stay ignored. Do not commit/push unless asked.
