# PROGRESS LOG

## Verified status — 2026-11-15

The all-X failure is repaired. Tracking-only inference completes and passes the official evaluator. The output exactly matches the published baseline_raw; no improved classifier score is claimed.

### Root causes fixed
- EPTS format is frame_id:player slots:ball, each object containing normalized x,y,speed. The old parser treated the first player as ball and speed as height.
- XML integer track IDs now match parsed player slots; teams use H/A consistently.
- Raw IDs 1..199319 map to submission frames 0..199318.
- Both entrypoints share the loading/classification pipeline; player-team mappings are no longer passed as frame tags.
- Output contains exactly frame,label. Missing ball -> D is explicitly a baseline heuristic; valid ball without players -> X.

### Verification
- Diagnostic: `python More_Than_A_Hack_Task_1/diagnostic_supervised.py` from workspace root.
- Checked all 199319 raw ball observations, frame IDs, 46 tracks, independent nearest-player samples, D/X fallbacks and submission columns.
- Inference: `python More_Than_A_Hack_Task_1/main.py --device cpu --data-dir More_Than_A_Hack_Task_1/data/tactical --output More_Than_A_Hack_Task_1/results/supervised_recovery/recovered_tracking.csv`
- Evaluation: `python More_Than_A_Hack_Task_1/evaluation/run_evaluation.py --results More_Than_A_Hack_Task_1/results/supervised_recovery --out More_Than_A_Hack_Task_1/evaluation/supervised_recovery --no-detail`
- Distribution: H=63964, A=49524, D=85831, X=0. Full dataframe equals evaluation/results/baseline_raw.csv.
- Public macroF1=0.597227; spellF1=0.037838. D recall=0.336283.
- baseline_smooth125: macroF1=0.581185, spellF1=0.424242. Recovered raw output has LOWER spellF1; optimization remains unfinished.

### Corrections and limitations
The former claim that enhanced output tied baseline_smooth125 was not supported by evaluation of its actual prediction files and is retracted. Old all-X CSVs are preserved and are not current verified outputs.
Smart Tagging/homography fusion is explicitly disabled; legacy loaders are not repaired or validated. No smoothing, CUDA acceleration or improved algorithm is claimed. Ground truth is used only by the evaluator, never inference. Do not tune against test labels.

### OpenCode supervision
Old looping session interrupted. Recovery session: ses_e1f36b224ffeFneQ1UkXZbwOe5 (Supervised classifier recovery). Supervisor supplied a concrete recovery script after repeated tool misuse/compaction and a transport error; it was executed through OpenCode session shell. Supervisor independently checked the generated CSV and evaluator results.
Original source files and this progress log have .before-supervised-recovery.bak backups. No datasets or previous prediction files were overwritten.

### Next step
Recovery is complete. For future performance work, design and document temporal possession/dead-ball logic, verify it independently, and score a separate submission. Do not rerun the recovery application script or return to the broken test_fix.py diagnostic.


## Generic input filenames — verified 2026-11-15
- OpenCode clean session ses_e1f20496fffewwU7cohnVDcj5Z (Generic ATD filenames fix) changed load_tactical_data to discover *_atd.txt and same-stem XML, and removed main.py's second hardcoded metadata lookup.
- Folder basename need not match filename. BAR-ATH_broadcast_atd and BAR-ATH_denem_atd accepted; arbitrary match prefix accepted. Missing metadata/no candidate/ambiguous candidates give explicit errors.
- Supervisor corrected generated test fixtures after OpenCode used invalid class names/data; interrupted OpenCode before independent test edits.
- Verified command: python -m unittest discover -s More_Than_A_Hack_Task_1 -p test_generic_inputs.py -v (NO trailing dot). All 8 tests passed, including full main pipeline H/A/D, renamed directories, missing XML, multiple pairs and valid pair plus orphan TXT.
- Real broadcast folder independently loaded: 164488 frames, 51 tracks, 83121 valid ball frames. No datasets/prediction CSVs changed. This validates parsing/loading, not cross-source evaluation alignment or improved scores.
- Current task complete. Sources: main.py, src/enhanced_classifier.py; regression: test_generic_inputs.py.
