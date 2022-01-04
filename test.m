clc
clear
I = imread("class.jpg");
eyeDetector = vision.CascadeObjectDetector('EyePairSmall');
eyeBox = step(eyeDetector, I);

if(~isempty(eyeBox)) 
    Eyes = insertObjectAnnotation(I,'rectangle',eyeBox,'Mouth');   
    imshow(Eyes);
else
    imshow(I);
end
