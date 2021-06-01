#!/bin/bash
# to copy fbank_pitch features.
# LIU YUANYUAN, TUT, 20/10/2020

datadir=$1
outdir=$1

if [ -f path.sh ]; then . ./path.sh; fi
featlist=$datadir/fbank_pitch.list
# scan files with ext in current folder and all subfolders
find $datadir -type f -name "fbank_pitch.*.ark" > $featlist

if [ ! -f $featlist ]; then
  echo "[error] there are no files in $datadir" && exit 1;  
fi

while read line; do
  full_name=$line
  # check file exist
  name=$(basename $full_name)
  echo $name
  filename=${name%.*}
  [ ! -f "$full_name" ] && echo "[error] no such audio file..." && exit 1;
  copy-feats ark:$full_name ark,t:$outdir/$filename".txt"
  # echo $utterance_id "sox -r 8000 -t raw -e signed-integer -b 16 -c 1 " $full_name " -t wav - |"
done < $featlist;
