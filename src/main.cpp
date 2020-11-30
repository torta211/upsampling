#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>


std::string ExtractFileName(std::string full_path)
{
    int start = full_path.find_last_of("\\/") + 1;
    int end = full_path.find_last_of(".");
    return full_path.substr(start, end - start);
}

cv::Mat CreateGaussianKernel2D(int r, float sig)
{
    cv::Mat kernel(cv::Size(2 * r + 1, 2 * r + 1), CV_32FC1);

    float sum = 0.0;
    for (int x = -1 * r; x <= r; ++x)
    {
        for (int y = -1 * r; y <= r; ++y)
        {
            float val = std::exp(-(x * x + y * y) / (2 * sig * sig));
            kernel.at<float>(x + r, y + r) = val;
            sum += val;
        }
    }

    // normalising the Kernel
    return kernel / sum;
}

float GaussianForIntensity(uchar intensity_1, uchar intensity_2, float sigma_sq_times_two, float kernel_sum)
{
    float dist = std::abs(intensity_1 - intensity_2);
    return std::exp(-1.0 * dist / sigma_sq_times_two) / kernel_sum;
}

// It is not nice to calculate distance in RGB color space
float GaussianForRGBColor(cv::Vec3b color_1, cv::Vec3b color_2, float sigma_sq_times_two, float kernel_sum)
{
    float dist = cv::norm(color_1, color_2, cv::NORM_L2);
    return std::exp(-1.0 * dist / sigma_sq_times_two) / kernel_sum;
}

void UseBilateralFilterNaive(const cv::Mat& input, cv::Mat& output, float sig_spatial, float sig_spectral)
{
    // determine the size of the kernel. We want the weights to approximate zero at every direction, so would like to cover the range -2.5 sig, 2.5 sig
    float k = 2.5;
    int r = std::round(k * sig_spatial);
    float sig_spectral_square_times_two = sig_spectral * sig_spectral * 2.0f;
    float spectral_kernel_sum = std::sqrt(2 * M_PI) * sig_spectral;

    cv::Mat Wg = CreateGaussianKernel2D(r, sig_spatial);

    const auto width = input.cols;
    const auto height = input.rows;
    for (int row = r; row < height - r; ++row)
    {
        std::cout << "Applying filter... " << std::ceil((double)row / ((double)height - (double)r) * 100) << "%\r" << std::flush;
        for (int col = r; col < width - r; ++col)
        { 
            float sum = 0.0f;
            float sum_weights = 0.0f;
            uchar p_middle = input.at<uchar>(row, col);

            for (int d_row = -1 * r; d_row <= r; ++d_row)
            {
                for (int d_col = -1 * r; d_col <= r; ++d_col)
                {
                    uchar p_kernel = input.at<uchar>(row + d_row, col + d_col);
                    float w_spatial = Wg.at<float>(r + d_row, r + d_col);
                    float w_spectral = GaussianForIntensity(p_middle, p_kernel, sig_spectral_square_times_two, spectral_kernel_sum);
                    float w = w_spatial * w_spectral;
                    sum_weights += w;
                    sum += p_kernel * w;
                }
            }
            output.at<uchar>(row, col) = sum / sum_weights;
        }
    }
    std::cout << std::endl;
}

void UseJointBilateralFilterNaive(const cv::Mat& image, const cv::Mat& colored_guide, cv::Mat& output, float sig_spectral, int r, const cv::Mat& Wg)
{
    float sig_spectral_square_times_two = sig_spectral * sig_spectral * 2.0f;
    float spectral_kernel_sum = std::sqrt(2 * M_PI) * sig_spectral;
    
    const auto width = image.cols;
    const auto height = image.rows;

    for (int row = r; row < height - r; ++row)
    {
        std::cout << "Applying filter... " << std::ceil((double)row / ((double)height - (double)r) * 100) << "%\r" << std::flush;
        for (int col = r; col < width - r; ++col)
        {
            float sum = 0.0f;
            float sum_weights = 0.0f;
            cv::Vec3b color_middle = colored_guide.at<cv::Vec3b>(row, col);

            for (int d_row = -1 * r; d_row <= r; ++d_row)
            {
                for (int d_col = -1 * r; d_col <= r; ++d_col)
                {
                    cv::Vec3b color_kernel = colored_guide.at<cv::Vec3b>(row + d_row, col + d_col);
                    uchar img_pixel_kernel = image.at<uchar>(row + d_row, col + d_col);
                    float w_spatial = Wg.at<float>(r + d_row, r + d_col);
                    float w_spectral = GaussianForRGBColor(color_middle, color_kernel, sig_spectral_square_times_two, spectral_kernel_sum);
                    float w = w_spatial * w_spectral;
                    sum_weights += w;
                    sum += img_pixel_kernel * w;
                }
            }
            output.at<uchar>(row, col) = sum / sum_weights;
        }
    }
    std::cout << std::endl;
}

void UpsampleDepthImageWithRGBGuideNaive(const cv::Mat& depth, const cv::Mat& rgb, cv::Mat& output, float sig_spatial, float sig_spectral)
{
    // determine the size of the kernel. We want the weights to approximate zero at every direction, so would like to cover the range -2.5 sig, 2.5 sig
    float k = 2.5;
    int r = std::round(k * sig_spatial);
    cv::Mat Wg = CreateGaussianKernel2D(r, sig_spatial);

    double r_height = (double)(rgb.rows) / (double)depth.rows;
    double r_width = (double)(rgb.cols) / (double)depth.cols;
    int num_steps = std::floor(std::log2(std::min(r_width, r_height)));

    output = depth.clone();
    cv::Mat rgbResized = rgb.clone();

    for (int i = 0; i <= num_steps; ++i)
    {
        cv::resize(output, output, cv::Size(), 2, 2);
        cv::resize(rgb, rgbResized, output.size());
        // apply filter
        UseJointBilateralFilterNaive(output.clone(), rgbResized, output, sig_spectral, r, Wg);
    }
    // final step (or the only one, if the downscale is smaller than 2)
    cv::resize(output, output, rgb.size());
    UseJointBilateralFilterNaive(output.clone(), rgb, output, sig_spectral, r, Wg);
}


int main(int argc, char** argv) {
    /*
    * SUBTASK 1: Implement a bilateral filter, and run it with a 4x4 grid of parameters
    * SUBTASK 2: Create a guided filter, Convert a bilateral filter to Guided Joint bilateral filter for guided image upsampling.
    */

    if (argc != 3 && argc != 4)
    {
        std::cerr << "Usage: \n - For grid of bilateral filters:\n" << argv[0] << " img_of_choice.png output_folder\n\n"
            << " - For guided bilateral filter upsampling:\n"
            << argv[0] << " disparity_img.png corresponding_rgb_img.png output_folder\n" << std::endl;
        return 1;
    }
    if (argc == 3)
    {
        cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (!image.data)
        {
            std::cerr << "No image data for " << argv[1] << std::endl;
            return EXIT_FAILURE;
        }
        std::string file_name = ExtractFileName(argv[1]);
        std::string out_folder = argv[2];

        // construct a grid of parameters for the kernel
        float sigs_spatial[] = { 2, 4, 8, 12 };       // spatial
        float sigs_spectral[] = { 2, 4, 8, 16 };      // intensity

        for (int spatial_i = 0; spatial_i < 4; ++spatial_i)
        {
            for (int spectral_i = 0; spectral_i < 4; ++spectral_i)
            {
                std::cout << "Using parameters: spatial sigma=" << sigs_spatial[spatial_i] << ", spectral sigma=" << sigs_spectral[spectral_i] << std::endl;
                cv::Mat output = cv::Mat::zeros(image.rows, image.rows, image.type());
                UseBilateralFilterNaive(image, output, sigs_spatial[spatial_i], sigs_spectral[spectral_i]);
                std::stringstream out;
                out << out_folder << "/" << file_name << spatial_i << spectral_i << ".png";
                cv::imwrite(out.str(), output);
            }
        }
    }
    if (argc == 4)
    {
        cv::Mat imageDisp = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (!imageDisp.data)
        {
            std::cerr << "No image data for " << argv[1] << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat imageRGB = cv::imread(argv[2]);
        if (!imageRGB.data)
        {
            std::cerr << "No image data for " << argv[2] << std::endl;
            return EXIT_FAILURE;
        }

        std::string file_name = ExtractFileName(argv[1]);
        std::string out_folder = argv[3];
        
        cv::Mat output = imageDisp.clone();
        UpsampleDepthImageWithRGBGuideNaive(imageDisp, imageRGB, output, 4, 4);
        std::stringstream out;
        out << out_folder << "/" << file_name << "_upsampled.png";
        cv::imwrite(out.str(), output);
    }
    return 0;
}