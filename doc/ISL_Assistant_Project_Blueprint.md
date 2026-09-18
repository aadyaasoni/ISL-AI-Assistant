**PROJECT BLUEPRINT**

**Multimodal, Multilingual AI**

**Sign-Language Communication Assistant**

*MVP Phase — ISL ↔ Text ↔ AI ↔ 3D Avatar*

**Team Lead:** Aadyaa Soni — architecture, orchestration, meaning layer, avatar integration, code review & verification

**Team Member:** Gauri — data pipeline, ISL recognition model, preprocessing, model training

**Institution:** Greater Noida Institute of Technology (GNIOT)

**Duration:** 6 weeks (assumed — adjust if your actual window differs)

**Document version:** v1.0

# 1. Why This Project Is Resume-Worthy

Sign-language recognition by itself is a saturated space — dozens of public repos already do ISL/ASL alphabet classification with MediaPipe + CNN, or one-directional text-to-avatar conversion. What is NOT common is a system that treats sign language as one modality inside a genuine multi-agent communication architecture. That reframing is what should anchor the resume line and interview narrative.

### The differentiators worth building and highlighting

- Bidirectional loop, not one-way translation — sign user and hearing user both stay in the conversation, mediated live by the AI.
- Orchestrator Agent — a routing layer that decides which sub-system (ISL model, meaning layer, conversation agent) handles each input. This is an agentic systems pattern, not a classifier.
- Common Meaning Layer — a shared semantic representation, closer to an interlingua in machine translation than a label lookup. This is the intellectual core of the project.
- Confidence-aware routing and fallback — the system knows when it doesn't know, and asks for clarification instead of guessing (detailed in Section 5).
- Evaluation discipline — most student sign-language projects report only classification accuracy and stop. Adding round-trip meaning-preservation checks and documented failure analysis is rare and reads as engineering maturity, not just modeling.

Resume framing: describe this as "designed and led a multi-agent orchestration system for cross-modal communication," with ISL recognition as one solved input modality — not as "built a sign language classifier." The architecture is the differentiator; the classifier is a component.

# 2. System Architecture (MVP Scope)

MVP flow: ISL (camera) → Recognition Model → Text (gloss) → Meaning Layer → Conversation Agent → Response Text → ISL Gloss → 3D Avatar.

### Component responsibilities

| **Component**            | **Responsibility**                                                                   | **Owner**                             |
|--------------------------|--------------------------------------------------------------------------------------|---------------------------------------|
| Multimodal Input Layer   | Captures camera frames (MVP: video only); mic/text deferred to later phase           | Gauri                                 |
| ISL Recognition System   | Landmark extraction (MediaPipe) → sequence model → gloss + confidence score          | Gauri                                 |
| Orchestrator Agent       | Routes input to the right pipeline; applies confidence thresholds; triggers fallback | Aadyaa                                |
| Common Meaning Layer     | Converts ISL gloss ↔ shared semantic frame (intent + entities + slots)               | Aadyaa                                |
| Conversation Agent       | Holds dialogue state, decides next system response using an LLM                      | Aadyaa                                |
| Sign-Generation + Avatar | Maps response text/gloss to a pre-built animation sequence and renders it            | Gauri (assets) + Aadyaa (integration) |

# 3. Team Structure & Code Verification Workflow

Two-person team. Clear module ownership avoids overlap; a lightweight review gate keeps you (lead) accountable for what ships, without becoming a bottleneck.

## 3.1 Role split

| **Person**    | **Owns**                                                                                                                                    | **Does not touch without review**                                         |
|---------------|---------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Aadyaa (Lead) | Orchestrator, Meaning Layer, Conversation Agent, integration, final review, repo structure, README/demo narrative                           | Gauri's model training code (reviews it, doesn't rewrite it unilaterally) |
| Gauri         | Dataset collection & preprocessing, ISL recognition model (training, evaluation), landmark extraction pipeline, avatar animation asset prep | Orchestrator/meaning-layer logic (can propose changes via PR)             |

## 3.2 Verification workflow (Git-based, lightweight)

- Repo structure: main branch is protected. Gauri works on feature branches (e.g. gauri/isl-model-v1).
- Every push from Gauri opens a Pull Request into main — no direct commits to main by anyone.
- You (lead) review every PR before merge: check (a) does it run end-to-end, (b) does output format match the interface contract in Section 4, (c) no hardcoded local paths, (d) basic code cleanliness.
- Use a shared CONTRACT.md file from Day 1 defining exact input/output shape for every module boundary (e.g. "ISL model outputs: {gloss: str, confidence: float, timestamp: float}") — this is what makes review fast, since you're checking against a spec, not guessing intent.
- Daily 10-minute async sync (text/voice note is fine) — what got merged, what's blocked, what's next. Avoid long meetings; keep momentum on solo-paced work.
- Tag PRs small — one module or one fix per PR. Large PRs are hard to verify properly and slow you down as reviewer.

# 4. Technical Stack

| **Layer**           | **Tools / Libraries**                                                                        | **Notes**                                                                                              |
|---------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Landmark extraction | MediaPipe Holistic (hands + pose + face)                                                     | Face landmarks matter for ISL non-manual markers (eyebrows, head tilt)                                 |
| Sequence model      | PyTorch — BiLSTM or Transformer encoder over landmark sequences                              | Sequence-level, not single-frame classification — needed for word/phrase-level ISL, not just alphabets |
| Meaning layer       | Rule-based slot-filling to start; optional small LLM-assisted normalization later            | Keep MVP deterministic and debuggable before adding LLM calls here                                     |
| Conversation agent  | Claude/GPT API with a constrained system prompt + conversation state object                  | Context = last N turns + current semantic frame, not raw video                                         |
| Avatar              | Pre-rigged 3D model (Blender/Mixamo) driven by pre-recorded or blended gloss animation clips | Do NOT attempt real-time procedural sign generation for MVP — use a gloss-to-clip lookup + blending    |
| Serving             | FastAPI backend, lightweight web frontend (React) for camera capture + avatar render         | Keep it cross-platform and free of hardcoded paths per your usual standard                             |
| Experiment tracking | joblib / simple CSV logs, or Weights & Biases if you want it on resume                       | W&B is a nice, cheap resume line if you already log metrics anyway                                     |

# 5. Differentiating Features To Build In

These are the additions worth prioritizing because they're rare in existing public repos and each is independently a good interview talking point.

### 5.1 Confidence-based routing (you already flagged this)

- ISL model returns a confidence score alongside every gloss prediction.
- Orchestrator applies a threshold (e.g. \< 0.6): low-confidence predictions don't get passed silently to the meaning layer.
- Below threshold → orchestrator triggers a clarification behavior instead of guessing.

### 5.2 Fallback / clarification loop

- On low confidence or ambiguous meaning-layer output, the Conversation Agent generates a clarifying response, and the Avatar signs a short "can you repeat / do you mean X or Y" prompt.
- This turns failure cases into a visible, demoable feature instead of a silent wrong answer — very effective in a live demo.

### 5.3 New suggestions worth adding

- Round-trip meaning-preservation eval: take N held-out phrases, run them through the full ISL→meaning→response→ISL loop, and manually score whether intent survived. This is the "I evaluated my system, not just my model" story that almost no student repo tells.
  - Concretely: build a 20–30 example test set, log pass/fail + failure category (misrecognition vs. meaning-layer ambiguity vs. avatar mapping gap).
- Ambiguity logging dashboard: a simple table/log of every case where the orchestrator triggered fallback, with the reason. Turns into a great "failure analysis" slide/section.
- Gloss caching layer: cache avatar animations for frequently used gloss sequences instead of regenerating/re-blending each time — a real engineering trade-off (latency vs. storage) you can discuss concretely in interviews.
- Session-level conversation memory: Conversation Agent remembers earlier turns in the session (e.g. a previously established topic or name) so responses aren't stateless — demonstrates agentic context handling, not just single-turn QA.
- Explicit interface contracts (CONTRACT.md, Section 3.2): treating module boundaries like API contracts is itself worth a line on a resume — shows software engineering discipline, not just notebook-style ML.

Recommendation: build 5.1 and 5.2 first (already prioritized), then 5.3's eval harness and ambiguity log next — those two together are the strongest "this isn't just another ISL classifier" evidence for reviewers.

# 6. Dataset Plan

### 6.1 Candidate datasets (verify licensing/availability before committing)

| **Dataset**                    | **Content**                                         | **Use for**                                                                                                           |
|--------------------------------|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| INCLUDE (ISL, IIT Bombay)      | Word-level ISL videos, 260+ signs, multiple signers | Primary — word/phrase-level recognition, closer to your MVP goal than alphabet-only sets                              |
| ISL-CSLRT                      | Continuous ISL sentences with word-level annotation | Optional stretch — continuous signing, useful once alphabet/word MVP works                                            |
| Kaggle ISL Alphabet/Digit sets | Static A–Z, 0–9 images                              | Fallback / bootstrapping only — not enough alone for a "communication assistant" story since it's just fingerspelling |

Recommendation: prioritize INCLUDE or an equivalent word/phrase-level dataset over alphabet-only sets. Fingerspelling-only demos are what every other repo already shows — word/phrase-level signs closer to real conversational ISL is what supports your "communication assistant, not classifier" positioning.

### 6.2 Data pipeline ownership

- Gauri: download, inspect class balance, extract MediaPipe landmarks per video, build train/val/test splits, document any class imbalance handling.
- Aadyaa: define the exact schema landmarks must be saved in (so the orchestrator/meaning layer can consume model output without translation layers), review data split for leakage (same signer in train and test can inflate accuracy — worth explicitly avoiding and mentioning in your writeup).

# 7. Week-by-Week Timeline (6 Weeks)

Assumes ~10–12 hrs/week per person. Each week ends with a merge into main and a working, demoable state — never leave main broken over a week boundary.

| **Week** | **Aadyaa (Lead)**                                                                                                                                          | **Gauri**                                                                                           | **Milestone (end of week)**                                       |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 1        | Repo setup, CONTRACT.md, orchestrator skeleton (routing stub), architecture doc finalized                                                                  | Dataset download, exploration, class balance report, MediaPipe landmark extraction pipeline         | Repo scaffolded; dataset validated and landmark-extractable       |
| 2        | Meaning layer schema design (semantic frame format); mock meaning layer with hardcoded rules                                                               | Train baseline sequence model (BiLSTM) on landmarks; log macro-F1 per class                         | Baseline ISL model trained; meaning layer schema frozen           |
| 3        | Conversation Agent v1 (LLM call + basic dialogue state); connect to mock meaning layer                                                                     | Improve model (data augmentation / class_weight balancing); add confidence score output             | End-to-end mock pipeline runs (fake ISL input → real AI response) |
| 4        | Confidence-based routing + fallback/clarification logic in orchestrator (Section 5.1–5.2)                                                                  | Integrate real trained model into pipeline; fix output schema to match CONTRACT.md                  | Real ISL input flows through orchestrator with fallback working   |
| 5        | Avatar integration: gloss-to-clip mapping + rendering pipeline; gloss caching layer                                                                        | Prepare/rig avatar animation clips for target vocabulary; support Aadyaa on avatar integration bugs | Full loop working: camera → ISL → AI → Avatar, end to end         |
| 6        | Eval harness (round-trip meaning preservation test set), ambiguity log, polish demo, write README + architecture write-up, code review pass on all modules | Run eval set, help fix failure cases surfaced by eval, record demo footage                          | Demo-ready build, documented eval results, resume-ready README    |

# 8. Step-by-Step: What To Actually Do, In Order

## Step 1 — Foundation (before any modeling)

- Create GitHub repo, protected main branch, folder structure (e.g. /orchestrator, /isl_model, /meaning_layer, /conversation_agent, /avatar, /data, /eval).
- Write CONTRACT.md — exact JSON shape for every module's input and output. Do this together before writing code; it's the single highest-leverage document for a 2-person team.
- Set up environment: requirements.txt / pyproject.toml, no hardcoded paths (use relative paths or a config file), README with setup instructions from day one.

## Step 2 — Data (Gauri-led, Aadyaa reviews)

- Download chosen dataset(s); write a data exploration notebook — class distribution, video lengths, signer diversity.
- Build landmark extraction script (MediaPipe → saved arrays), matching the schema in CONTRACT.md.
- Split data by signer (not randomly) to avoid leakage; document this decision explicitly for the report.

## Step 3 — ISL recognition model (Gauri-led)

- Baseline: BiLSTM over landmark sequences, cross-entropy loss, macro-F1 as the primary metric (not raw accuracy, given likely class imbalance).
- Add confidence output (softmax max probability is fine for MVP; don't over-engineer calibration this early).
- PR into main once it hits an agreed baseline threshold; Aadyaa reviews against CONTRACT.md before merge.

## Step 4 — Orchestrator + Meaning Layer (Aadyaa-led)

- Orchestrator: simple router first (single input type → single path) — don't over-build multi-path logic before MVP proves the single path works.
- Meaning layer: start rule-based/deterministic (gloss → semantic frame via a lookup + slot-filling), not LLM-based — easier to debug and demo reliably.
- Add confidence threshold + fallback branch once the happy path works end-to-end.

## Step 5 — Conversation Agent (Aadyaa-led)

- Constrained system prompt: agent should only respond within the scope of the semantic frame it receives, not free-chat — keeps demo outputs predictable.
- Maintain a simple conversation state object (last N turns) passed with each call.

## Step 6 — Avatar integration (joint)

- Gauri prepares a fixed vocabulary of gloss → animation clip mappings (don't attempt full generative sign synthesis for MVP).
- Aadyaa wires orchestrator response → gloss sequence → clip lookup → render/playback in frontend.
- Add caching for repeated gloss sequences (Section 5.3).

## Step 7 — Evaluation & polish (joint, lead-driven)

- Build the 20–30 case round-trip eval set; run it; log failures by category.
- Fix the highest-frequency failure category first, not every failure — document what you didn't fix and why (this is a legitimate, honest part of a good project write-up).
- Record a clean demo video as a fallback for live-demo risk (camera/lighting issues are common in ISL demos).
- Finalize README: architecture diagram, setup steps, known limitations, and the resume-framing pitch from Section 1.

# 9. Risks & Mitigations

| **Risk**                                                       | **Mitigation**                                                                                                                                                       |
|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Word/phrase-level ISL model accuracy is low with limited data  | Start with a smaller, high-frequency vocabulary subset (20–30 signs) rather than the full dataset — a reliable small system beats an unreliable large one for a demo |
| Avatar generation scope creep (trying to animate everything)   | Fixed clip library for MVP vocabulary only; explicitly scope this in the README as a known limitation with a stated expansion path                                   |
| Live demo camera/lighting failure                              | Always have a recorded backup demo video ready                                                                                                                       |
| Two-person bandwidth — one person blocked waiting on the other | CONTRACT.md lets both people build against a spec in parallel from Week 1 instead of waiting on integration                                                          |
| Scope creep into BSL/ASL/multimodal before MVP is solid        | Explicitly timebox MVP to Week 5; Week 6 is polish/eval only, not new scope                                                                                          |

# 10. Resume & Interview Framing (Recap)

When this ships, describe it as: "Led a 2-person team building a multi-agent AI system for real-time, bidirectional ISL communication — designed an orchestrator agent with confidence-based routing and fallback handling, a common semantic meaning layer, and a round-trip evaluation harness to measure intent preservation end-to-end." That sentence leads with system design and evaluation rigor, which is what separates this from the many single-model ISL classifiers already public.
