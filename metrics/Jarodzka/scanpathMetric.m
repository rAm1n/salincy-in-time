function [simCostFinal] = scanpathMetric(folderName,scanPath2,imgNum,compareVis1,compareVis2)

% This function computes the similarity between the set of scanpaths given
% by the user and those recorded in the subjective experiment. The
% metric used is the Jarodzka algorithm (http://portal.research.lu.se/ws/files/5608175/1539210.PDF)
% where the similarity criteria was slightly modified to use orthodrominc
% distances in 360 instead of the eucleadean case.Also, the hungarian
% optimizer algorithm is used to then make a 1 to 1 match between the
% scanpaths to get the least possible final cost. The function expects 5
% input arguments and provides a final similarity score. Please note that a
% lower score implies a better match.

% folderName : Root folder where the ground-truth images and scanpaths are stored
% scanPath2: A Nx5 vector having each row in this format 
% [ScanpathNumber(starts at 1 and serially increments),ScanpathArmNumber(starts at 1 
% and serially increments),Time(insecs),XValue(in equirectangular),YValue(in Equirectangular)]
% imgNum: The scalar value of the image number you want to test
% compareVis1: If you want to visualise the matching process for a certain
% test scanpath. USe 0 to turn off.
% compareVis1: If you want to visualise the matching process for a certain
% reference scanpath. Use 0 to turn off.

% an example usage maybe:
% score=scanpathMetric('F:\VR\FTP',myScanPath,29,0,0) OR
% score=scanpathMetric('F:\VR\FTP',myScanPath,29,1,1) for testing a
% visualisation for first reference nd first test image.

%% read the scanpaths from file
colorSet='rgbcmyk';
imgRGB=imread([folderName '\Images\P' num2str(imgNum) '.jpg']);
width=size(imgRGB,2);
height=size(imgRGB,1);
scanFile = fopen([folderName '\Scanpaths\SP' num2str(imgNum) '.txt'], 'r');
scanData = fscanf(scanFile,'%d,%f,%d,%d\n');
fclose(scanFile);
pers=0; startTime=0; idxHere=1;
for idx=1:floor(numel(scanData)/4)
    if scanData(4*(idx-1)+1)==1
        if pers>0
            scanPath(idxHere,1)=pers;
            scanPath(idxHere,2)=prevId;
            scanPath(idxHere,3)=25;
            scanPath(idxHere,4)=prevX;
            scanPath(idxHere,5)=prevY;
            idxHere=idxHere+1;
        end;
        pers=pers+1;
        startTime=scanData(4*(idx-1)+2);
    end;
    scanPath(idxHere,1)=pers;
    prevId=scanData(4*(idx-1)+1);
    scanPath(idxHere,2)=prevId;
    time=scanData(4*(idx-1)+2)-startTime;
    if time<0
        time=60+time;
    end;
    scanPath(idxHere,3)=time;
    prevX=scanData(4*(idx-1)+3);
    prevY=scanData(4*(idx-1)+4);
    scanPath(idxHere,4)=prevX;
    scanPath(idxHere,5)=prevY;
    idxHere=idxHere+1;
end;
numPersons1=pers;
numPersons2=max(scanPath2(:,1));
simMatrix=zeros(numPersons1,numPersons2);

% VECTOR SIMILARITY METRIC
for pe1=1:numPersons1
    for pe2=1:numPersons2
        
        path1=scanPath(scanPath(:,1)==pe1,2:end);
        path2=scanPath2(scanPath2(:,1)==pe2,2:end);
        arms1=size(path1,1); arms2=size(path2,1);
        
        %Plot the graphs
        if pe1==compareVis1 && pe2==compareVis2
            figure; hold on;
            for j=1:(arms1-1)
                arrow([path1(j+1,2),path1(j,3),path1(j,4)],[path1(j+1,2),path1(j+1,3),path1(j+1,4)],'FaceColor',colorSet(1),'EdgeColor',colorSet(1),'Length',0,'Width',1);
                arrow([path1(j,2),path1(j,3),path1(j,4)],[path1(j+1,2),path1(j,3),path1(j,4)],'FaceColor',colorSet(1),'EdgeColor',colorSet(1),'Length',0,'Width',10);
                scatter3(path1(j,2),path1(j,3),path1(j,4),'MarkerEdgeColor','k','MarkerFaceColor','k');
            end;
            for j=1:(size(path2,1)-1)
                arrow([path2(j+1,2),path2(j,3),path2(j,4)],[path2(j+1,2),path2(j+1,3),path2(j+1,4)],'FaceColor',colorSet(2),'EdgeColor',colorSet(2),'Length',0,'Width',1);
                arrow([path2(j,2),path2(j,3),path2(j,4)],[path2(j+1,2),path2(j,3),path2(j,4)],'FaceColor',colorSet(2),'EdgeColor',colorSet(2),'Length',0,'Width',10);
                scatter3(path2(j,2),path2(j,3),path2(j,4),'MarkerEdgeColor','k','MarkerFaceColor','k');
            end;
            hold off;
        end;
        %Generate the distances matrix
        simSaccades=zeros(arms1,arms2);
        for j=1:(arms1-1)
            for k=1:(arms2-1)
            	[simSaccades(j,k)]=checkVectorSimilarity(path1(j:j+1,:),path2(k:k+1,:),width,height);
            end;
        end;
        %Apply Dijkstra algorithm on the matrix
        dijkstraDistances=1000*ones(arms1,arms2); dijkstraDistances(1,1)=simSaccades(1,1);
        dijkstraNodeConsidered=zeros(arms1,arms2);
        dijkstraFromWhere=cell(arms1,arms2);
        while numel(find(dijkstraNodeConsidered==0))>0
            rowNum=0;colNum=0;minVal=10000;
            for r=1:arms1
                for c=1:arms2
                    if dijkstraNodeConsidered(r,c)==0 && dijkstraDistances(r,c)<minVal
                        rowNum=r;
                        colNum=c;
                        minVal=dijkstraDistances(r,c);
                    end;
                end;
            end;
            dijkstraNodeConsidered(rowNum,colNum)=1;
            if rowNum<arms1
                if dijkstraDistances(rowNum+1,colNum) > (dijkstraDistances(rowNum,colNum)+simSaccades(rowNum+1,colNum))
                    dijkstraDistances(rowNum+1,colNum)=dijkstraDistances(rowNum,colNum)+simSaccades(rowNum+1,colNum);
                    dijkstraFromWhere(rowNum+1,colNum)={[rowNum,colNum]};
                end;
            end;
            if colNum<arms2
                if dijkstraDistances(rowNum,colNum+1) > (dijkstraDistances(rowNum,colNum)+simSaccades(rowNum,colNum+1))
                    dijkstraDistances(rowNum,colNum+1)=dijkstraDistances(rowNum,colNum)+simSaccades(rowNum,colNum+1);
                    dijkstraFromWhere(rowNum,colNum+1)={[rowNum,colNum]};
                end;
                if rowNum<arms1
                    if dijkstraDistances(rowNum+1,colNum+1) > (dijkstraDistances(rowNum,colNum)+simSaccades(rowNum+1,colNum+1))
                        dijkstraDistances(rowNum+1,colNum+1)=dijkstraDistances(rowNum,colNum)+simSaccades(rowNum+1,colNum+1);
                        dijkstraFromWhere(rowNum+1,colNum+1)={[rowNum,colNum]};
                    end;
                end;
            end;
            if rowNum==arms1 && colNum==arms2
                break;
            end;
        end;
        maxDist=dijkstraDistances(end,end);
        dijkstraDistances(dijkstraDistances==1000)=maxDist;
        simMatrix(pe1,pe2)=maxDist;
        curX=arms2;curY=arms1; pathLen=0;
        while curX~=1 || curY~=1
            pathLen=pathLen+1;
            curCoor=cell2mat(dijkstraFromWhere(curY,curX));
            curX=curCoor(2);
            curY=curCoor(1);
        end;
        simMatrix(pe1,pe2)=simMatrix(pe1,pe2)/pathLen;   % Normalisation by total pathlength to compare fairly
        %plot path
        if pe1==compareVis1 && pe2==compareVis2
            figure; hold on;
            curX=arms2;curY=arms1;
            while curX~=1 || curY~=1
                curCoor=cell2mat(dijkstraFromWhere(curY,curX));
                arrow([curCoor(2),curCoor(1),dijkstraDistances(curCoor(1),curCoor(2))],[curX,curY,dijkstraDistances(curY,curX)],'FaceColor','m','EdgeColor','m','Length',2,'Width',4);
                curX=curCoor(2);
                curY=curCoor(1);
            end;
            surf(1:arms2,1:arms1,dijkstraDistances);
        end;
        %disp(['Similarity between ' num2str(pe1) ' and ' num2str(pe2) ' is ' num2str(simMatrix(pe1,pe2))]);
    end;
end;

%% Perform a one to one match between the scan-paths using a Hungarian
[assign,simCostFinal]=munkres(simMatrix);
