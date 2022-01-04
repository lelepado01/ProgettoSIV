clc; 
clf('reset');  

right=imread('RIGHT.jpg');
left=imread('LEFT.jpg');
noface=imread('no_face.jpg');
straight=imread('STRAIGHT.jpg');

face_detector = vision.CascadeObjectDetector(); 
eye_detector = vision.CascadeObjectDetector('EyePairSmall'); 

img = imread("imgs/class.jpg"); 
img = rgb2gray(img); 
img = flip(img, 2);

%vid=snapshot(cam);  
%vid = rgb2gray(vid);    
%img = flip(vid, 2); 

bbox = step(face_detector, img);  
  
if ~ isempty(bbox) 
    biggest_box=1;     
    for i=1:rank(bbox)
        if bbox(i,3)>bbox(biggest_box,3)
            biggest_box=i;
        end
    end
    box = regionprops(img,'Area', 'BoundingBox'); 

    faceImage = imcrop(img, bbox(biggest_box,:)); 
    bboxeyes = step(eye_detector, faceImage);

    figure; 
    imshow(img); 
    title("Grayscale image"); 
    for i=1:size(bbox,1)
        rectangle('position', bbox(i, :), 'lineWidth', 2, 'edgeColor', 'y');
    end

    %figure;  
    %imshow(faceImage);    
    %title("Face in image");         

    if ~ isempty(bboxeyes)  
         
       biggest_box_eyes = 1;     
       for i=1:rank(bboxeyes) 
           if bboxeyes(i,3)>bboxeyes(biggest_box_eyes,3)
               biggest_box_eyes=i;
           end
       end
         
       bboxeyeshalf = [
           bboxeyes(biggest_box_eyes,1), ...
           bboxeyes(biggest_box_eyes,2), ...
           bboxeyes(biggest_box_eyes,3)/3, ...
           bboxeyes(biggest_box_eyes,4)
       ];  
       
         
       eyesImage = imcrop(faceImage,bboxeyeshalf(1,:));   
       eyesImage = imadjust(eyesImage);   

       r = bboxeyeshalf(1,4)/4;
       [centers, radii, metric] = imfindcircles(eyesImage, [floor(r-r/4) floor(r+r/2)], 'ObjectPolarity','dark', 'Sensitivity', 0.93);
         
       [M,I] = sort(radii, 'descend');
       disp(radii);
          
       eyesPositions = centers;
            
       %figure; 
       %imshow(eyesImage); 
       %title("Eyes in image"); 

        %viscircles(centers, radii,'EdgeColor','b');
              
        %if ~isempty(centers)
        %    pupil_x=centers(1);
        %    disL=abs(0-pupil_x);    
        %    disR=abs(bboxeyes(1,3)/3-pupil_x);
        %    subplot(2,2,4);
        %    if disL>disR+16
        %        imshow(right);
        %    else if disR > disL
        %        imshow(left);
        %        else
        %           imshow(straight); 
        %        end
        %    end
        %end          
     end
end
set(gca,'XtickLabel',[],'YtickLabel',[]);