# Bilateral filtering
---
## example usage

```
upsampling.exe image_path output_folder
```

or

```
upsampling.exe depth_image_to_upscale rgb_guide_img output_folder
```

## Task 1: grid search of bilateral filters

### results on a noisy grayscale lena image.

<img src="input/lena_noise.png" width="300"/> 

| | 2 | 4 | 8 | 12 |
|---|---|---|---|---|
| 2 | <img src="images_out/lena00.png" width="230"/> |<img src="images_out/lena01.png" width="230"/> | <img src="images_out/lena02.png" width="230"/> | <img src="images_out/lena03.png" width="230"/> | 
| 4 | <img src="images_out/lena10.png" width="230"/> | <img src="images_out/lena11.png" width="230"/> | <img src="images_out/lena12.png" width="230"/> | <img src="images_out/lena13.png" width="230"/> |
| 8 | <img src="images_out/lena20.png" width="230"/> | <img src="images_out/lena21.png" width="230"/> | <img src="images_out/lena22.png" width="230"/> | <img src="images_out/lena23.png" width="230"/> |
| 16| <img src="images_out/lena30.png" width="230"/> | <img src="images_out/lena31.png" width="230"/> | <img src="images_out/lena32.png" width="230"/> | <img src="images_out/lena33.png" width="230"/> |

Horizontally: spatial sigma / Vertically: spectral sigma


## Task 2: Upscale a depth image using iterative joint bilateral filtering

### RGB Image as a guide (1390x1110), and a really downscaled depth image

<img src="input/art_rgb_hr.png" width="350"/> <img src="input/art_disp_lr.png" width="350"/> 

### Result with both sigmas = 4

<img src="images_out/art_disp_lr_upsampled.png" width=350/>

I think that is a surprisingly nice result, even without using median bilateral filter.

### TODO:
 - 1D gaussian kernel
 - Color distance measurement in Las
