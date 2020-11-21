Mode = 'Camera';
%Mode = 'MovieFile';

XExpand = 20;
YExpand = 10;

FrameMax = 440;

% 学習済み物体認識オブジェクトを生成
FaceDetector = vision.CascadeObjectDetector();
EyesDetector = vision.CascadeObjectDetector('EyePairSmall', 'UseROI', true);
MouthDetector = vision.CascadeObjectDetector('Mouth', 'UseROI', true);
%追跡オブジェクト生成
EyesPointTracker = vision.PointTracker('MaxBidirectionalError', 2);
MouthPointTracker = vision.PointTracker('MaxBidirectionalError', 2);
switch Mode
    case 'Camera'
        % カメラオブジェクト生成
        cam = webcam();
        VideoFrame = snapshot(cam);
    case 'MovieFile'
        VideoFileReader = VideoReader('./Movie/MyFace.mov');
        VideoFrame      = readFrame(VideoFileReader);
end

% ウェブカメラのフレームサイズを取得
FrameSize = size(VideoFrame);

% ビデオ再生用オブジェクト生成
VideoPlayer = vision.VideoPlayer('Position', [100 100 [FrameSize(2), FrameSize(1)]+30]);

%% ビデオで顔認識
runLoop = true;
EyesnumPts = 0;
MouthnumPts = 0;
frameCount = 0;

while runLoop && frameCount < FrameMax

    % 次のフレームを取得する
    switch Mode
    case 'Camera'
        % カメラからフレーム取得
        VideoFrame = snapshot(cam);
    case 'MovieFile'
        VideoFrame = readFrame(VideoFileReader);
    end
    videoFrameGray = rgb2gray(VideoFrame);
    frameCount = frameCount + 1;

    %追跡オブジェクトの対応点が10個以下になったら物体位置再特定
    if or(EyesnumPts < 10,MouthnumPts < 10)
        % 物体位置特定モード
        %顔の位置特定
        FaceBBox = FaceDetector.step(videoFrameGray);
        
        %顔位置特定が成功したら、顔領域の中から目の位置を特定
        if ~isempty(FaceBBox)
            EyesBBox = EyesDetector.step(videoFrameGray,FaceBBox(1,:));
            %口は顔領域の下半分から探索する
            %UnderFaceBBox = FaceBBox(1,:) + [0,round(FaceBBox(1,4)/2),-round(FaceBBox(1,4)/2),-round(FaceBBox(1,4)/2)];
            MouthBBox = MouthDetector.step(videoFrameGray,FaceBBox(1,:));
        end
        %目の位置が特定できたら　位置情報を画像に重畳
        if ~isempty(EyesBBox)
            % Find corner points inside the detected region.
            EyesPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', EyesBBox(1, :));
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            % ポイントトラッカーの初期化
            EyesxyPoints = EyesPoints.Location;
            EyesnumPts = size(EyesxyPoints,1);
            release(EyesPointTracker);
            initialize(EyesPointTracker, EyesxyPoints, videoFrameGray);
            % Save a copy of the points.
            EoldPoints = EyesxyPoints;
            
            %トラッキング点を生成
             [VideoFrame,EyebboxPoints,EyebboxPolygon] =...
                 DetectTargetRegion(VideoFrame,EyesBBox,EyesxyPoints);
        end
       
        %口の位置が特定できたら　位置情報を画像に重畳
        if ~isempty(MouthBBox)
            % Find corner points inside the detected region.
            MouthPoints = detectMinEigenFeatures(videoFrameGray, 'ROI', MouthBBox(1, :));
            % Convert the rectangle represented as [x, y, w, h] into an
            % M-by-2 matrix of [x,y] coordinates of the four corners. This
            % is needed to be able to transform the bounding box to display
            % the orientation of the face.
            % ポイントトラッカーの初期化
            MouthxyPoints = MouthPoints.Location;
            MouthnumPts = size(MouthxyPoints,1);
            release(MouthPointTracker);
            initialize(MouthPointTracker, MouthxyPoints, videoFrameGray);
            % Save a copy of the points.
            MoldPoints = MouthxyPoints;
            
            %トラッキング点を生成
             [VideoFrame,MouthbboxPoints,MouthbboxPolygon] =...
                 DetectTargetRegion(VideoFrame,MouthBBox,MouthxyPoints);
        end

    else
        % Eye Tracking mode.
        [EyesxyPoints, EyeisFound] = step(EyesPointTracker, videoFrameGray);
        EyevisiblePoints = EyesxyPoints(EyeisFound, :);
        EyesoldInliers = EoldPoints(EyeisFound, :);
        EyesnumPts = size(EyevisiblePoints, 1);

        if EyesnumPts >= 10
            % 一つ前のフレームで作成したポイント位置
            %現在のフレームのポイント位置のズレから2次元の幾何学変換を推定
            [xform, inlierIdx] = estimateGeometricTransform2D(...
                EyesoldInliers, EyevisiblePoints, 'similarity', 'MaxDistance', 4);
            EyesoldInliers    = EyesoldInliers(inlierIdx, :);
            EyevisiblePoints = EyevisiblePoints(inlierIdx, :);

            % 順方向アフィン変換でトラッキング領域を示すポリゴン座標を変換
            EyebboxPoints = transformPointsForward(xform, EyebboxPoints);

            % 特定した目領域を示す四角形の座標
            % 座標形式を[x1 y1 x2 y2 x3 y3 x4 y4]に変換
            EyebboxPolygon = reshape(EyebboxPoints', 1, []);

            %平行四辺形にならないように調整
            EyebboxPolygon(1) = max(EyebboxPolygon(1),EyebboxPolygon(7));%X1
            EyebboxPolygon(2) = max(EyebboxPolygon(2),EyebboxPolygon(4));%Y1
            EyebboxPolygon(3) = max(EyebboxPolygon(3),EyebboxPolygon(5));%X2
            EyebboxPolygon(4) = max(EyebboxPolygon(2),EyebboxPolygon(4));%Y2
            EyebboxPolygon(5) = max(EyebboxPolygon(3),EyebboxPolygon(5));%X3
            EyebboxPolygon(6) = max(EyebboxPolygon(6),EyebboxPolygon(8));%Y3
            EyebboxPolygon(7) = max(EyebboxPolygon(1),EyebboxPolygon(7));%X4
            EyebboxPolygon(8) = max(EyebboxPolygon(6),EyebboxPolygon(8));%Y4
            
            Xmin  = round(max(EyebboxPolygon(1),EyebboxPolygon(7))) - XExpand ;
            Xmax = round(max(EyebboxPolygon(3),EyebboxPolygon(5))) + XExpand ;
            Ymin  = round(max(EyebboxPolygon(2),EyebboxPolygon(4))) - YExpand ;
            Ymax = round(max(EyebboxPolygon(6),EyebboxPolygon(8))) + YExpand ;
            
          % ポリゴンを画面に重畳.
           % VideoFrame = insertShape(VideoFrame, 'Polygon', EyebboxPolygon, 'LineWidth', 3);

            % トラックポイントを描画.
            %videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % ポイント位置を更新
            EoldPoints = EyevisiblePoints;
            setPoints(EyesPointTracker, EoldPoints);
            
            %画像の一部を反転
            VideoFrame(Ymin:Ymax,Xmin:Xmax,:) = flipud(VideoFrame(Ymin:Ymax,Xmin:Xmax,:));
        end
        % Mouth Tracking mode.
        [MouthxyPoints, MouthisFound] = step(MouthPointTracker, videoFrameGray);
        MouthvisiblePoints = MouthxyPoints(MouthisFound, :);
        MoutholdInliers = MoldPoints(MouthisFound, :);
        MouthsnumPts = size(MouthvisiblePoints, 1);

        if MouthsnumPts >= 10
            % 一つ前のフレームで作成したポイント位置
            %現在のフレームのポイント位置のズレから2次元の幾何学変換を推定
            [xform, inlierIdx] = estimateGeometricTransform2D(...
                MoutholdInliers, MouthvisiblePoints, 'similarity', 'MaxDistance', 4);
            MoutholdInliers    = MoutholdInliers(inlierIdx, :);
            MouthvisiblePoints = MouthvisiblePoints(inlierIdx, :);

            % 順方向アフィン変換でトラッキング領域を示すポリゴン座標を変換
            MouthbboxPoints = transformPointsForward(xform, MouthbboxPoints);

            % 特定した口領域を示す四角形の座標
            % 座標形式を[x1 y1 x2 y2 x3 y3 x4 y4]に変換
            MouthbboxPolygon = reshape(MouthbboxPoints', 1, []);
            
            %平行四辺形にならないように調整
            MouthbboxPolygon(1) = max(MouthbboxPolygon(1),MouthbboxPolygon(7));%X1
            MouthbboxPolygon(2) = max(MouthbboxPolygon(2),MouthbboxPolygon(4));%Y1
            MouthbboxPolygon(3) = max(MouthbboxPolygon(3),MouthbboxPolygon(5));%X2
            MouthbboxPolygon(4) = max(MouthbboxPolygon(2),MouthbboxPolygon(4));%Y2
            MouthbboxPolygon(5) = max(MouthbboxPolygon(3),MouthbboxPolygon(5)) ;%X3
            MouthbboxPolygon(6) = max(MouthbboxPolygon(6),MouthbboxPolygon(8));%Y3
            MouthbboxPolygon(7) = max(MouthbboxPolygon(1),MouthbboxPolygon(7));%X4
            MouthbboxPolygon(8) = max(MouthbboxPolygon(6),MouthbboxPolygon(8));%Y4
            
            Xmin  = round(max(MouthbboxPolygon(1),MouthbboxPolygon(7))) - XExpand ;
            Xmax = round(max(MouthbboxPolygon(3),MouthbboxPolygon(5))) + XExpand ;
            Ymin  = round(max(MouthbboxPolygon(2),MouthbboxPolygon(4))) - YExpand ;
            Ymax = round(max(MouthbboxPolygon(6),MouthbboxPolygon(8))) + YExpand ;
            

            % ポリゴンを画面に重畳.
            %VideoFrame = insertShape(VideoFrame, 'Polygon', MouthbboxPolygon, 'LineWidth', 3);

            % トラックポイントを描画.
            %videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % ポイント位置を更新
            MoldPoints = MouthvisiblePoints;
            setPoints(MouthPointTracker, MoldPoints);
            
             %画像の一部を反転
            VideoFrame(Ymin:Ymax,Xmin:Xmax,:) = flipud(VideoFrame(Ymin:Ymax,Xmin:Xmax,:));
        end

    end

    % ビデオプレイヤーフレーム更新
    if frameCount < FrameMax/2
        %前半は逆さまにする
        step(VideoPlayer,  flipud(VideoFrame));
    else
        %後半は正しい向きにする
        step(VideoPlayer,  VideoFrame);
    end
    
    % ビデオプレイヤーウィンドが閉じられていないか確認
    runLoop = isOpen(VideoPlayer);
end

% Clean up.
release(VideoPlayer);
release(EyesPointTracker);
release(FaceDetector);
release(EyesDetector);
release(MouthDetector);
clear all;
disp('OK!');


function [VideoFrame,bboxPoints,bboxPolygon] = DetectTargetRegion(VideoFrame,TargetBBox,xyPoints)
    
    bboxPoints = bbox2points(TargetBBox(1, :));

    % Convert the box corners into the [x1 y1 x2 y2 x3 y3 x4 y4]
    % format required by insertShape.
    bboxPolygon = reshape(bboxPoints', 1, []);

    % Display a bounding box around the detected face.
    VideoFrame = insertShape(VideoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);

    % Display detected corners.
    VideoFrame = insertMarker(VideoFrame, xyPoints, '+', 'Color', 'white');
end