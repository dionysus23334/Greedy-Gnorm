% Given Data
data = [32,  34,  13,  16,  70,  49,  40,  21,   8,  24,  25,  53;
        123,  66,  48,  60,  88,  91, 132,  71,  93,  87,  89,  86;
        55, 112,  97,  85, 117, 111, 136, 128,  44,  62,  65, 100;
        129, 109, 120, 125, 107,  59,  46,  72,  84,  69,  68, 110;
        137, 114,  64, 124,  47,  39,  76, 119,  54, 131, 103, 115;
        108,  78, 134, 106,  36, 127,  81,  92, 140, 104,  33,  28;
        90,  77,  11,  29, 138, 113,  58,  61,  67,  56,  99,  41;
        22,  27, 133, 139,  12,  15,  38,  96, 121, 105,  17, 118;
        143,  79,   4,   5, 141,  98,  83,  95,   9, 135,   7,  35;
        82, 126, 142,  57,  14,  26,  10,   2,  63,  73, 116,  23;
        45,  75,  37, 102,  20, 144, 130,   3,   6,  51,  74,  31;
        42,  18,  80,  43,   1, 101,  30,  94,  52, 122,  50,  19];

% Standardize the data
data_standardized = (data - mean(data(:))) / std(data(:));

% Create the grid for the data
[X, Y] = meshgrid(1:size(data, 2), 1:size(data, 1));

% Create a finer grid for interpolation
[X_fine, Y_fine] = meshgrid(linspace(1, size(data, 2), 100), linspace(1, size(data, 1), 100));

% Interpolate the data for smoothness
Z_fine = interp2(X, Y, data_standardized, X_fine, Y_fine, 'cubic');

% Plot the smoothed 3D surface
figure;
surf(X_fine, Y_fine, Z_fine);
shading interp; % Smooth shading

% Add labels and title
xlabel('X axis');
ylabel('Y axis');
zlabel('Importance (Standardized)');
title('Smoothed 3D Plot of Standardized Importance Matrix');
colormap('viridis'); % Color map for better visualization
colorbar; % Show color bar
