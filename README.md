This project is aims to improve outdoor image quality affected by inclement weather, particularly haze and fog, which degrade the performance of imaging systems. Traditional dehazing methods often suffer from color distortions and halo artifacts, especially in scenes with the sky. The objectives of the project are to effectively remove haze and fog from images and enhance images affected by such weather conditions.

# Methodology
The project employs a combination of image processing techniques to achieve these goals:
![image](https://github.com/user-attachments/assets/d7ae5e3e-b749-4309-865c-0f4b708ead14)

1. Luminance-inverted MSR (Multi-Scale Retinex):

- Converts the input image to the YCbCr color space to extract the luminance channel.
- The MSR algorithm is applied to the inverted luminance channel to enhance details and contrast.
- The result is adjusted and converted back to the RGB color space.

2. Color Recovery:

- Converts the inverted RGB image to double precision.
- Computes color recovery values for each channel based on intensity sums.
- Applies a logarithmic transformation and enhances color recovery.

3. Adaptive Gamma Correction:

- Converts the input image to grayscale and applies noise-reducing filtering.
- Performs threshold segmentation to estimate the ratio of white and gray regions.
- Applies adaptive gamma correction to the color-recovered image based on the ratio.

4. CLAHE (Contrast-Limited Adaptive Histogram Equalization):

- Applies CLAHE to each channel of the hazy input image to improve local contrast and enhance details.
- Combines the results into a single image converted to double precision.

5. Seamless Stitching:

- Uses mean filtering on gray and white masks to stitch the color-recovered image with the input image.
- Further stitches gray regions with the CLAHE result.
- Converts the stitched image to uint8 precision for final output.

# Results
Figure 1: Original Image with Histogram
![image](https://github.com/user-attachments/assets/af4b4bcc-eb85-41ad-a6ce-fa74aad7ea36)

The image appears hazy, obscuring details like a road sign board.


Figure 2: Inverted MSR Image with Histogram
![image](https://github.com/user-attachments/assets/602f8104-b969-4309-8aa6-af08892e78b5)

The MSR algorithm enhances details and contrast, restoring visual quality.


Figure 3: Color Recovered Image with Histogram
![image](https://github.com/user-attachments/assets/947927ec-8023-4ce8-b8dd-96b936a056c5)

Color recovery widens and heightens the histogram, improving color balance.


Figure 4: Adaptive Gamma Corrected Image with Histogram
![image](https://github.com/user-attachments/assets/005fef79-94ce-4d8c-96dc-003d7037827c)

Enhances contrast based on the white-to-gray ratio, but some text on the road sign remains hard to read.


Figure 5: CLAHE Image with Histogram
![image](https://github.com/user-attachments/assets/299926e0-35e7-4ee5-8fda-02884a62bc27)

Enhances local contrast while preventing noise amplification, improving visibility of the road sign.


Figure 6: After Seamless Stitching Image with Original Image
![image](https://github.com/user-attachments/assets/247e7d4d-7ca9-49e1-be65-758c470cbdfc)

Combines regions smoothly, reducing the ratio of white and gray areas, improving clarity and legibility of the road sign.

# Conclusion
The study successfully employs image defogging and enhancement techniques to improve the visibility of hazy or foggy images. The dehazing algorithm involves luminance inversion, color recovery, adaptive gamma correction, contrast enhancement using CLAHE, and seamless stitching. The results demonstrate that the effectiveness of dehazing is influenced by algorithm parameters, particularly the ratio of white and gray regions in the input image.
