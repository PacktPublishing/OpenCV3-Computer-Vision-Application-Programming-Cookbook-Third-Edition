opencv_createsamples -info stop.txt -vec stop.vec -w 24 -h 24 -num 10
opencv_traincascade -data classifier -vec stop.vec -bg neg.txt  -numPos 9 -numNeg 20 -numStages 20 -minHitRate 0.95 -maxFalseAlarmRate 0.5 -w 24 -h 24 
