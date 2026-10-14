## P11 pitch/Transpose/NR
T1: P11-8 NR std::vector<float> bands(numBands) every process() while engaged (NoiseReduction.h:136)
T1: P11-9 NR reset() ch={} free + exp.resize alloc; process() calls reset on every re-engage; prepare same, LR filters prepare allocs, no reserve (NoiseReduction.h:108-116,292-296,101-105)
T2: P11-10 NR per sample/band/ch decibelsToGain(threshDb) + 20log10(threshold) + exp release ~3 pow+6 log10+1 exp per sample/ch; threshold block-invariant; bandOffsetsDb zeros (NoiseReduction.h:175,180-182,225-226)
T3: P11-14 Transpose at rest: pushDrySamples copy, block.clear, mixWetSamples every block, root node always runs (Transpose.h:95-110)
T1: P11-16 Transpose prepare mixer.prepare (DryWetMixer) can alloc (no-op unless block/channels grow) (Transpose.h:67-77)
G: P11-17 CPUAwarePitchShifter reserve on EVERY compile resets retired/reads pending/live while audio-thread prepare moves them: race (CPUAwarePitchShifter.h:47-65); same PitchShifterRB.h:89-117
T6/mem: P11-18 retired engine stays allocated until next compile; any spec diff incl maximumBlockSize rebuilds whole engine incl RB weight solve (msg thread) (CPUAwarePitchShifter.h:47-62)
T1: P11-20 PitchShifter lever + TS2 pedal prepare -> PitchShifter2 ring.setSize, fine.resize (finec_dsp PitchShifter2.h:229-230), no reserve (hidden pedals)
P11-7 startBlock no-op on private engines (Transpose.h:100, RainDelay.h:1011, Split.h:161, Reverb.h:325/1476) - minor per-block
P11 NotProblem: NR early-out at -80 dB deliberate CPU gate
