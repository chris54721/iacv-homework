% Image Analysis and Computer Vision - Homework A.Y. 2021/22
% Author: Christian Grasso (10652464)

%% Setup
close all;
clear;
clc;

addpath(genpath('functions'));

img = im2double(imread('villa.png'));
%% F1 - Feature detection
imgGrayscale = rgb2gray(img);

resizeMult = 0.6;
newSize = resizeMult.*size(img, [1 2]);

imgGrayscaleRescaled = imresize(imgGrayscale, newSize, 'bilinear');
imgRescaled = imresize(img, newSize, 'bilinear');

figure; hold all;
imshow(imgGrayscaleRescaled);
edges = edge(imgGrayscaleRescaled, 'canny', [0 0.1]);
imshow(edges);

figure;
imshow(imgRescaled);
hold all;

[H,o_pi,R] = hough(edges);
P = houghpeaks(H, 100, 'threshold', 0.3*max(H(:)));
hlines = houghlines(edges, o_pi, R, P, 'FillGap', 4, 'MinLength', 14);
hold all;
for k = 1:length(hlines)
    seg = [hlines(k).point1; hlines(k).point2];
    segLine = segToLine(seg);
    plot(seg(:,1),seg(:,2),'Color', 'yellow', 'LineWidth', 2);
end

figure;
imshow(imadjust(rescale(H)),'XData',o_pi,'YData',R,'InitialMagnification','fit');
title('Hough transform');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(o_pi(P(:,2)),R(P(:,1)),'s','color','yellow');

points = detectHarrisFeatures(histeq(imgGrayscale));
figure;
imshow(img);
hold all;
plot(points.selectStrongest(5000));
%% G1 - 2D reconstruction of a horizontal section
%% Draw parallel lines for affine rectification
figure;
hold all;
imshow(img);

nFamilies = 2;
parallelLines = drawLineFamilies('Draw parallel lines', nFamilies, 2);

close all;
%% Compute the affine rectification matrix
[H_aff, imHLinf] = buildHaff(parallelLines);
%figure; hold all;
%p = projective2d(H_aff.');
%imshow(imwarp(img, p, 'OutputView', imref2d(size(img)*3)));
%% Select lines to be used as constraints
figure;
hold all;
imshow(img);

cosTheta = cos(atan(3.9)); % from the sun position

orthogonalLines = drawLines('Draw orthogonal lines', 2, 'r');
angledLines = drawLines('Draw an horizontal line and the shadow', 2, 'g');
%% Build the metric rectification matrix
% Compute images of selected lines
a = H_aff.' \ orthogonalLines(1, :).';
b = H_aff.' \ orthogonalLines(2, :).';
l = H_aff.' \ angledLines(1, :).';
m = H_aff.' \ angledLines(2, :).';

% Solve the system using the selected lines
syms x1 x2;
imDCCP = [x1,x2,0;x2,1,0;0,0,0];
orthogonalConstraintEq = a.'*imDCCP*b ==0;
angledConstraintEq = cosTheta*sqrt((l.'*imDCCP*l)*(m.'*imDCCP*m))==(l.'*imDCCP*m);
S = solve([orthogonalConstraintEq, angledConstraintEq], [x1, x2]);

figure; hold all;
fimplicit(orthogonalConstraintEq);
fimplicit(angledConstraintEq);
% Note that, depending on the selected lines, 1 or two solutions
% could be returned. Here, the case with 1 solution is handled.
s1 = double(S.x1(1));
s2 = double(S.x2(1));

% Build the matrix via Cholesky decomposition
K = chol([s1, s2; s2, 1]);
H_metric_inv = [K.',[0;0];0,0,1]; % Don't invert it since we need to multiply it anyway
%% Metric rectification
Hrect = H_metric_inv \ H_aff;

tform = projective2d(Hrect.');
imgRect = im2double(imwarp(img, tform, 'OutputView', imref2d(size(img)*5)));

figure;
imshow(imgRect);
%% Calculate and display the facades ratio
figure;
imshow(img);
hold all;

% Select facade corners and find their image
title('Select the corners of facades 2 and 3 in order, then press Enter');
[x,y] = getpts();
xyRect = Hrect * [x y [1;1;1]].';
xyRect = xyRect ./ xyRect(3, :); 

p2 = xyRect(:,1);
p23 = xyRect(:,2);
p3 = xyRect(:,3);

% Compute and print the ratio
lengthF2 = norm(p23-p2);
lengthF3 = norm(p3-p23);

disp(lengthF2);
disp(lengthF3);

figure;
imshow(imgRect);
hold all;
plot(p2(1), p2(2), 'd');
plot(p23(1), p23(2), 'd');
plot(p3(1), p3(2), 'd');

ratio =  lengthF2 / lengthF3;
fprintf('Ratio of facade 2 over 3: %d\n', ratio);
%% G2 - Calibration
close all;
%% Get the vertical vanishing point
figure;
imshow(img);
hold all;
verticalLines = drawLines('Draw vertical parallel lines', 2, 'red');
vpV = hpnorm(cross(verticalLines(1,:), verticalLines(2,:)));
%% Find two more vanishing points
horizontalLines = drawLines('Draw two more lines on the horizontal plane', 2, 'green');
vp1 = hpnorm(cross(horizontalLines(1,:), imHLinf));
vp2 = hpnorm(cross(horizontalLines(2,:), imHLinf));
%% Compute the IAC
isize = size(img);
Hscaling = diag([1/isize(2), 1/isize(1), 1]);

vpV_scaled = Hscaling * vpV.';
vp1_scaled = Hscaling * vp1.';
vp2_scaled = Hscaling * vp2.';

Hrect_inv_scaled = Hscaling / Hrect;
h1 = Hrect_inv_scaled(:,1);
h2 = Hrect_inv_scaled(:,2);

syms x1 x2 x3 x4;
x = [x1 0 x2; 0 1 x3; x2 x3 x4];

S = solve([
    vp1_scaled.' * x * vpV_scaled == 0, ...
    vp2_scaled.' * x * vpV_scaled == 0, ...
    h1.' * x * h2 == 0, ...
    h1.' * x * h1 - h2.' * x * h2 == 0
], [x1 x2 x3 x4]);

s1 = double(S.x1);
s2 = double(S.x2);
s3 = double(S.x3);
s4 = double(S.x4);
omega = [s1 0 s2; 0 1 s3; s2 s3 s4];
%% Compute K using Cholesky decomposition
K = inv(chol(omega));
K = K ./ K(3, 3);
disp(K);
%% G3 - Reconstruction of a vertical facade
close all;
%% Find the line at infinity for the vertical plane
figure;
hold all;
imshow(img);

parallelLines = drawLineFamilies('Draw two pairs of parallel lines, first vertical then horizontal', 2, 2);
%% Compute the line at infinity of the vertical plane
[~, imVLinf] = buildHaff(parallelLines);
%% Find the intersection between the line at infinity and the IAC
syms x1 x2;
x = [x1 x2 1].';
S = solve([
    imVLinf.' * x == 0, ...
    x.' * omega * x == 0
], [x1,x2]);
s1 = double(S.x1);
s2 = double(S.x2);
%% Compute the rectification matrix for the vertical facade
I = [s1(1) s2(1) 1].';
J = [s1(2) s2(2) 1].';

imDCCP2 = I*(J.')+J*(I.');
imDCCP2 = imDCCP2 ./ norm(imDCCP2);
[U, S2, ~] = svd(imDCCP2);
S2(3,3) = 1;
Hrect2_inv = U * sqrt(S2);
%% Find the rotation of the output image
hLineRot = Hrect2_inv.' * parallelLines{2}(1,:).';
rotAngle = atan(-hLineRot(1)/hLineRot(2)) + pi;
%% Show the rectified vertical facade
figure;
Hrot = buildRotationMatrix(-rotAngle);
Hvrect = (Hrot / Hrect2_inv).';
tform2 = projective2d(Hvrect);
imgRectV = imwarp(img, tform2);
imshow(imgRectV);
imwrite(imgRectV, 'out/G3_final.png');
%% G4 - Localization
close all;
figure; imshow(img);
%% Select the last needed point and baseline points
title('Select the bottom corners clockwise from bottom-left and press Enter')
[xp,yp] = getpts();
%% Fit the homography
points_c = [0 0; 0 ratio; 1 ratio; 1 0];
points_img = [xp yp];
tform_loc = fitgeotrans(points_c, points_img, 'projective');
H_loc = tform_loc.T;
%% Find versors from the homography and the K matrix
c_ref = K \ H_loc;
c_ref = c_ref ./ c_ref(3,:);
R = [c_ref(:,1), c_ref(:,2), cross(c_ref(:,1), c_ref(:,2))];
% Make sure R is orthogonal
[U, ~, V] = svd(R);
R = U * V';

camera_orientation = R.';
camera_pos = -R.' * c_ref(:,3);
%% Use the camera height to find values in m
camera_mult = camera_pos(3) / 1.5;
camera_pos_m = camera_pos ./ camera_mult;
points_c_m = points_c ./ camera_mult;
%% Show camera position / orientation and facade
close all;
figure; hold all;
plotCamera('Location', camera_pos_m, 'Orientation', camera_orientation.');
pcshow([points_c_m, zeros(size(points_c_m,1), 1)],...
'red', 'VerticalAxisDir', 'up', 'MarkerSize', 100);
patch([points_c_m(2:3,1); flip(points_c_m(2:3,1))], [points_c_m(2:3,2); flip(points_c_m(2:3,2))], [0 0 20 20], 'b');
%% Functions
function lines = drawLineFamilies(title, nFamilies, nLinesPerFamily)
    colors = 'rgbm';
    lines = cell(nFamilies, 1); % store lines
    for i = 1:nFamilies
        lines{i} = drawLines([title, ' - family ', num2str(i), ' of ', num2str(nFamilies)], nLinesPerFamily, colors(i));
    end
end

function lines = drawLines(title_, nLines, color)
    count = 1;
    lines = nan(nLines, 3);
    while(count <= nLines)
        figure(gcf);
        title([title_, ' - segment ', num2str(count), ' of ', num2str(nLines)]);
        segment = drawline('Color', color);
        lines(count, :) = segToLine(segment.Position);
        count = count + 1;
    end
end

% Normalize a point in homogeneous coordinates.
function p_norm = hpnorm(p)
    p_norm = p ./ p(3);
end

function [H, imLinf] = buildHaff(lines)
    V = nan(2, length(lines));
    for i = 1:length(lines)
        A = lines{i}(:,1:2);
        B = -lines{i}(:,3);
        V(:,i) = A\B;
    end
    imLinf = fitLine(V);
    H = [eye(2), zeros(2,1); imLinf(:).'];
end

function matrix = buildRotationMatrix(angle)
    ca = cos(angle);
    sa = sin(angle);
    matrix = [ca,-sa,0;sa,ca,0;0,0,1];
end