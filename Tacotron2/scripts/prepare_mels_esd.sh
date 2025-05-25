#!/usr/bin/env bash

set -e

DATADIR="/home/madusov/vkr/data/ssw_esd_ljspeech_22050"
FILELISTSDIR="filelists_esd"

TRAINLIST="$FILELISTSDIR/audio_text_train.txt"
VALLIST="$FILELISTSDIR/audio_text_val.txt"

TRAINLIST_MEL="$FILELISTSDIR/mel_text_train.txt"
VALLIST_MEL="$FILELISTSDIR/mel_text_val.txt"

mkdir -p "$DATADIR/mels"
if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"	
fi	
