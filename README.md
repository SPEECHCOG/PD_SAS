# Notes:
LIU Yuanyuan, TUT, 2021-06-01.

This project is to extract speech attribute score (SAS) features for Parkinson's disease (PD) assessment. Commonly used features eGeMAPS and MFCC are used as baselines.

Speech attribute scores on articulation manner and place are extracted using MULAN-ACCENT implemented in Kaldi (https://github.com/Vanova/MULAN-ACCENT).

Our method was validated on a Finnish PD corpuse (PDSTU) and a Spanish PD corpus (PC-GITA). The predictions of speech/voice disorders, speech intelligibility, overall severity of communication disorder, UPDRS and UPDRS-speech are formed as regression problems.

# Environment:
- Kaldi
- python (opensmile, tensorflow, librosa, pandas...)
# Steps:
- Feature extraction
- - SAS: output from MULAN-ACCENT
- - fbank_pitch: input to MULAN-ACCENT (run copy_fbanks_pitch.sh)
- - egemaps: opensmile (run feature_extraction_opensmile.ipynb)
- Data precessing:
- - run data_preparation_regression.py (data for MFCC will be generated from data for fbank_pitch.)
- Regression:
- - run PD_assessment_regression.py
- Experimental results analysis:
- - run results_analysis_regression.ipynb
