# Notes:
LIU Yuanyuan, TUT, 2021-06-01.

This project is to extract speech attribute score (SAS) features for Parkinson's disease (PD) assessment. Commonly used features eGeMAPS and MFCC are used as baselines.

Speech attribute scores on articulation manner and place are extracted using MULAN-ACCENT implemented in Kaldi (https://github.com/Vanova/MULAN-ACCENT).

Our method was validated on a Finnish PD corpuse (PDSTU) and a Spanish PD corpus (PC-GITA). Both speech corpora contained speech from tasks of sustained-vowels, reading and spontaneous monologues. The predictions of voice disorders, speech intelligibility, overall severity of communication disorder in PDSTU, UPDRS and UPDRS-speech in PC-GITA are formed as regression problems.

# Environment:
- Kaldi
- python (opensmile, tensorflow, librosa, pandas...)
# Steps:
- Feature extraction
- - SAS: output from MULAN-ACCENT (we used the implemented CNN models for articulation manner and place, respectively.)
- - fbank_pitch: input to MULAN-ACCENT (run copy_fbanks_pitch.sh, to copy features of fbanks and pitch from .ark files.)
- - egemaps: opensmile (run feature_extraction_opensmile.ipynb. We used eGeMAPSv02.)
- Data precessing:
- - run data_preparation_regression.py (data for MFCC will be generated from data for fbank_pitch.)
- Regression:
- - run PD_assessment_regression.py
- Experimental results analysis:
- - run results_analysis_regression.ipynb
