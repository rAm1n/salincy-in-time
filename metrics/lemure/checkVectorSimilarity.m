function [sim]=checkVectorSimilarity(vect1,vect2,width,height)

%% convert to latitude and longitude
% diagDist=norm([width height]);
az1=2*pi*(vect1(:,3)/width)-pi;
az2=2*pi*(vect2(:,3)/width)-pi;
el1=pi*(vect1(:,4)/height)-(pi/2);
el2=pi*(vect2(:,4)/height)-(pi/2);

%% vector difference after conversion the globe-spherical to cartesian
% vectSub=norm([vect1(1,3)-vect1(2,3)-vect2(2,3)+vect2(1,3),vect1(1,4)-vect1(2,4)-vect2(2,4)+vect2(1,4)]);
% vectSubNormalised=vectSub/diagDist;
sx1 = (cos(az1) .* cos(el1)) ;
sx2 = (cos(az2) .* cos(el2)) ;
sy1 = (sin(az1) .* cos(el1)) ;
sy2 = (sin(az2) .* cos(el2)) ;
sz1 = (-sin(el1)) ;
sz2 = (-sin(el2)) ;
vectSubEquiRect=norm([(sx1(1)-sx1(2))-(sx2(1)-sx2(2)),(sy1(1)-sy1(2))-(sy2(1)-sy2(2)),(sz1(1)-sz1(2))-(sz2(1)-sz2(2))])/sqrt(4*4+4*4+4*4);
saccLengthDifference=abs(acos(sin(el1(1))*sin(el1(2))+cos(el1(1))*cos(el1(2))*cos(az1(1)-az1(2)))-acos(sin(el2(1))*sin(el2(2))+cos(el2(1))*cos(el2(2))*cos(az2(1)-az2(2))))/pi;

%% Difference in the euclidean distance between start points
% pixDifference=norm([vect1(1,3) - vect2(1,3), vect1(1,4) - vect2(1,4)]);
% centroidDifference=pixDifference/diagDist;
distOrthoVectDiff=abs(acos(sin(el1(1))*sin(el2(1))+cos(el1(1))*cos(el2(1))*cos(az1(1)-az2(1))))/pi; 

%% time difference
deltaT=(abs(vect1(1,2) - vect2(1,2)));
% Made by fitting the data from exploration speed
% Rai, Y., & LeCallet P. & Guiterriez J. (2017). A dataset of head and eye movements
% for 360 degree images. MMSys, Taiwan
timeDifference=1-exp(-0.15*deltaT);  

%% combine them
sim=(vectSubEquiRect+distOrthoVectDiff+2*timeDifference)/4;   % Equal weightage to space and time


