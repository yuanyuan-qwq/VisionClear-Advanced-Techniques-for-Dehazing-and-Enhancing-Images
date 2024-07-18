function [ClearOut] = RADE1(HazyInput)
    
    %Parameter
    x = 5;
    alpha = 0.8;
    belta = 0.03;
    whiteThreshold = 0.95;    % Threshold for white regions
    grayThreshold = 0.6;     % Threshold for gray regions

    % Plot histogram of inverted MSR image
    figure;
    subplot(121);
    imshow(HazyInput);
    title('Original Image');
    subplot(122);
    histogram(HazyInput);
    title('Histogram of Original Image');

    % Step 1: Luminance-inverted MSR 
    yuvHazyInput = rgb2ycbcr(HazyInput); % separates the image into its luminance (Y) and chrominance (Cb and Cr) components.
    yHazyInput = yuvHazyInput(:,:,1); % Extract the Y channel (luminance)
    invertedYClearOut = computeMSRGray(255 - yHazyInput); % Compute the inverted MSR on the luminance channel
    invertedYClearOut = uint8((1 - invertedYClearOut) * (235 - 16) + 16); % Adjust the output to desired range
    invertedYuvClearOut = yuvHazyInput; % Create a copy of the input image in YUV color space
    invertedYuvClearOut(:,:,1) = invertedYClearOut; % Replace the Y channel with the inverted MSR result
    invertedGrayClearOut = ycbcr2rgb(invertedYuvClearOut); % Convert the modified YUV image back to RGB
    MSRY = invertedGrayClearOut; % Store the result in MSRY variable

    % Plot histogram of inverted MSR image
    figure;
    subplot(121);
    imshow(invertedGrayClearOut);
    title('Inverted MSR Image');
    subplot(122);
    histogram(invertedYClearOut);
    title('Histogram of Inverted MSR Image');

    % Step 2: Color Recovery - yuan
    P0 = double(invertedGrayClearOut); % Convert the inverted RGB image to double precision
    R0 = P0(:, :, 1); 
    G0 = P0(:, :, 2); 
    B0 = P0(:, :, 3); 
    II = R0 + G0 + B0; % Sum of intensities across channels
    C(:,:,1) = x * R0 ./ II; % Compute color recovery for red channel
    C(:,:,2) = x * G0 ./ II; % Compute color recovery for green channel
    C(:,:,3) = x * B0 ./ II; % Compute color recovery for blue channel
    C = log(C + 1); % Apply logarithmic transformation
    P = P0 .^ (1 + C * belta); % Apply color recovery enhancement

    % Plot histogram of color-recovered image
    figure;
    subplot(121);
    imshow(uint8(P));
    title('Color-Recovered Image');
    subplot(122);
    histogram(uint8(P));
    title('Histogram of Color-Recovered Image');

    % Step 3: Adaptive Gamma - thong
    [m, n, channel] = size(HazyInput); % Get dimensions of the input image
    if (channel > 1)
        I0 = rgb2gray(HazyInput); % Convert input image to grayscale if it has more than one channel
    end
    I1 = im2double(I0); % Convert grayscale image to double precision
    I1 = medfilt2(I1, [9, 9]); % Apply median filtering for noise reduction

    % Threshold Segmentation
    I2 = ones(m, n); % Initialize a matrix with ones of the same size as the input image
    I2 = (I1 > grayThreshold) .* 0.5; % Assign 0.5 to pixels above the gray threshold
    I2 = I2 + (I1 > whiteThreshold) .* 0.5; % Add 0.5 to pixels above the white threshold
    Iwhite = (I1 > whiteThreshold); % Binary mask for white regions
    Igray = ((I1 >= grayThreshold) & (I1 <= whiteThreshold)); % Binary mask for gray regions

    % Estimate ratio of white and grayish regions
    countWhite = sum(sum(Iwhite)); % Count the number of white pixels
    countGray = sum(sum(Igray)); % Count the number of gray pixels
    ratio = (countWhite + countGray) / (m * n); % Calculate the ratio of white and gray regions

    % Region-ratio-based Adaptive Gamma Correction
    gamma_b = 1 + alpha * (1 - ratio); % Calculate the gamma correction factor
    P = (P ./ 255) .^ gamma_b; % Apply adaptive gamma correction

    % Plot histogram of adaptive gamma-corrected image
    figure;
    subplot(121);
    imshow(uint8(P * 255));
    title('Adaptive Gamma-Corrected Image');
    subplot(122);
    histogram(uint8(P * 255));
    title('Histogram of Adaptive Gamma-Corrected Image');

    % Step 4: CLAHE - Contrast-Limited Adaptive Histogram Equalization
    CLAHEImg = zeros(m, n); % Initialize a matrix for CLAHE result
    R = HazyInput(:,:,1); % Red channel of the input image
    G = HazyInput(:,:,2); % Green channel of the input image
    B = HazyInput(:,:,3); % Blue channel of the input image
    M = adapthisteq(R); % Apply CLAHE to the red channel
    L = adapthisteq(G); % Apply CLAHE to the green channel
    N = adapthisteq(B); % Apply CLAHE to the blue channel
    CLAHEImg = cat(3, M, L, N); % Combine the CLAHE results into a single image
    CLAHEImg = im2double(CLAHEImg); % Convert CLAHE image to double precision

    % Plot histogram of CLAHE image
    figure;
    subplot(121);
    imshow(CLAHEImg);
    title('CLAHE Image');
    subplot(122);
    histogram(CLAHEImg);
    title('Histogram of CLAHE Image');

    % Step 5: Seamless Stitching 
    % Mean-filtered Mask
    meanSize = 100; % Size of the mean filter
    meanFilter = fspecial('average', [meanSize, meanSize]); % Create a mean filter
    Igray = double(Igray); % Convert gray mask to double precision
    Igray = imfilter(Igray, meanFilter, 'replicate'); % Apply mean filtering to the gray mask
    Iwhite = double(Iwhite); % Convert white mask to double precision
    Iwhite = imfilter(Iwhite, meanFilter, 'replicate'); % Apply mean filtering to the white mask

    % White regions + Other parts
    a = im2double(HazyInput); % Convert the input image to double precision
    StitchImg = P; % Initialize the stitched image with the color recovery result
    StitchImg(:,:,1) = (1 - Iwhite(:,:)) .* P(:,:,1) + Iwhite(:,:) .* a(:,:,1); % Stitch white regions with the recovered color
    StitchImg(:,:,2) = (1 - Iwhite(:,:)) .* P(:,:,2) + Iwhite(:,:) .* a(:,:,2); % Stitch white regions with the recovered color
    StitchImg(:,:,3) = (1 - Iwhite(:,:)) .* P(:,:,3) + Iwhite(:,:) .* a(:,:,3); % Stitch white regions with the recovered color

    % Gray regions + Other parts
    P = StitchImg; % Update the stitched image with the previous result
    StitchImg(:,:,1) = (1 - Igray(:,:)) .* P(:,:,1) + Igray(:,:) .* CLAHEImg(:,:,1); % Stitch gray regions with the CLAHE result
    StitchImg(:,:,2) = (1 - Igray(:,:)) .* P(:,:,2) + Igray(:,:) .* CLAHEImg(:,:,2); % Stitch gray regions with the CLAHE result
    StitchImg(:,:,3) = (1 - Igray(:,:)) .* P(:,:,3) + Igray(:,:) .* CLAHEImg(:,:,3); % Stitch gray regions with the CLAHE result

    % Calculate the ratio for the dehaze output image
    countWhiteOut = sum(sum(Iwhite)); % Count the number of white pixels in the output
    countGrayOut = sum(sum(Igray)); % Count the number of gray pixels in the output
    ratioOut = (countWhiteOut + countGrayOut) / (m * n); % Calculate the ratio of white and gray regions in the output

    ClearOut = uint8(StitchImg * 255); % Convert the stitched image to uint8 precision

    figure;
    subplot(121);
    imshow(HazyInput);
    title(['Hazy Input: ' num2str(ratio)]);
    subplot(122);
    imshow(ClearOut);
    title(['Dehaze Output: ' num2str(ratioOut)]);
end

% Helper function for computing the MSR gray
function P = computeMSRGray(a)
    Y0 = double(a); % Convert input to double precision
    [N1, M1] = size(Y0); % Get dimensions of the input
    Ylog = log(Y0 + 1); % Compute logarithm of the input
    Yfft2 = fft2(Y0); % Compute 2D Fourier transform of the input
    sigma1 = 128; % Standard deviation for the first Gaussian filter
    F1 = fspecial('gaussian', [N1, M1], sigma1); % Create a Gaussian filter
    Efft1 = fft2(double(F1)); % Compute 2D Fourier transform of the Gaussian filter
    DY0 = Yfft2 .* Efft1; % Multiply the Fourier transforms
    DY = ifft2(DY0); % Compute inverse Fourier transform
    DYlog = log(DY + 1); % Compute logarithm of the result
    Yy1 = Ylog - DYlog; % Compute the difference between the logarithms
    sigma2 = 256; % Standard deviation for the second Gaussian filter
    F2 = fspecial('gaussian', [N1, M1], sigma2); % Create a Gaussian filter
    Efft2 = fft2(double(F2)); % Compute 2D Fourier transform of the Gaussian filter
    DY0 = Yfft2 .* Efft2; % Multiply the Fourier transforms
    DY = ifft2(DY0); % Compute inverse Fourier transform
    DYlog = log(DY + 1); % Compute logarithm of the result
    Yy2 = Ylog - DYlog; % Compute the difference between the logarithms
    sigma3 = 512; % Standard deviation for the third Gaussian filter
    F3 = fspecial('gaussian', [N1, M1], sigma3); % Create a Gaussian filter
    Efft3 = fft2(double(F3)); % Compute 2D Fourier transform of the Gaussian filter
    DR0 = Yfft2 .* Efft3; % Multiply the Fourier transforms
    DY = ifft2(DY0); % Compute inverse Fourier transform
    DYlog = log(DY + 1); % Compute logarithm of the result
    Yy3 = Ylog - DYlog; % Compute the difference between the logarithms
    Yy = (Yy1 + Yy2 + Yy3) / 3; % Average the difference images
    EXPYy = exp(Yy); % Compute exponential of the result
    MIN = min(min(EXPYy)); % Find the minimum value
    MAX = max(max(EXPYy)); % Find the maximum value
    EXPYy = (EXPYy - MIN) / (MAX - MIN); % Normalize the result
    EXPYy = adapthisteq(EXPYy); % Apply adaptive histogram equalization
    P = EXPYy; % Set the output as the result of the computation
end
