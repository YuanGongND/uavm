These are the raw videos from VGGSound.
Use the following command to extract the videos: for file in vggsound_*; do tar -xvf "$file" --strip=6; done
This will just write everything to the folder: video
The tar files use up 300GB of space, the raw videos use up 318GB of space.
199,176 videos were extracted. 
vggsound_19.tar.gz only has 9,176 videos, instead of 10k as the other tar files contained (this is expected).
