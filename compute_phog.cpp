#include <iostream>
#include <fstream>
#include <filesystem>
#include"cnpy.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <boost/filesystem.hpp>
#include <argparse/argparse.hpp>

namespace fs = boost::filesystem;


void printMatDetails(cv::Mat mat, std::string desc="Matrix") {
  std::string typeString;
  int type = mat.type();

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  typeString = "8U"; break;
    case CV_8S:  typeString = "8S"; break;
    case CV_16U: typeString = "16U"; break;
    case CV_16S: typeString = "16S"; break;
    case CV_32S: typeString = "32S"; break;
    case CV_32F: typeString = "32F"; break;
    case CV_64F: typeString = "64F"; break;
    default:     typeString = "User"; break;
  }

  typeString += "C";
  typeString += (chans+'0');

  double min, max; 
  cv::Scalar mean, std;
  cv::minMaxLoc(mat, &min, &max);
  cv::meanStdDev(mat, mean, std);
  if(min < 0.6 || max > 1)
    printf("%s: %s %dx%d [%.6f, %.6f] u:%.6f o:%.6f\n", desc.c_str(), typeString.c_str(), mat.cols, mat.rows, min, max, mean[0], std[0]);

}

void writeMatToFile(cv::Mat& m, std::string filename)
{
    std::ofstream fout(filename);
    if(!fout) {
        std::cout << "File Not Opened" << std::endl;  
        return;
    }

    for(int i=0; i<m.rows; i++) {
        for(int j=0; j<m.cols; j++) {
            fout << std::setprecision(18) << std::scientific << m.at<float>(i,j) << "\n";
        }
        fout << std::endl;
    }
    fout.close();
}

void getHistogram(const cv::Mat& edges, const cv::Mat& ors, const cv::Mat& mag, int startX, int startY, int width, int height, cv::Mat& hist)
{
    // Find and increment the right bin/s
    for (int x = startX; x < startX + height; x++)
    {
        for (int y = startY; y < startY + width; y++)
        {
            if (edges.at<uchar>(x,y) > 0)
            {
                int bin = (int)std::floor(ors.at<float>(x, y));
                hist.at<float>(0, bin) = hist.at<float>(0, bin) + mag.at<float>(x, y);
            }
        }
    }
}

void computePhog(const cv::Mat& image, cv::Mat& desc)
{
    int nbins = 60; // 20 bins as default, increased to 60
	int desc_size = nbins + 4 * nbins + 16 * nbins;

    cv::Mat img = image;
    if (img.channels() > 1)
    {
        // Convert the image to grayscale
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        // printMatDetails(img, "Image BW");
    }

    // Mean and Standard Deviation
    cv::Scalar cvMean;
    cv::Scalar cvStddev;
    cv::meanStdDev(img, cvMean, cvStddev);
    double mean = cvMean(0);

    // Apply Canny Edge Detector
    cv::Mat edges;
    // Reduce noise with a kernel 3x3
    cv::blur(img, edges, cv::Size(3,3));
    // printMatDetails(edges, "Image Blur");
    // Canny detector
    cv::Canny(edges, edges, 0.66 * mean, 1.33 * mean);

    // printMatDetails(edges, "edges");


    //  Computing the gradients.
    // Generate grad_x and grad_y
    cv::Mat grad_x, grad_y;

    // Gradient X
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);

    // Gradient Y
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);

    // Total Gradient (approximate)
    cv::Mat grad_m = cv::abs(grad_x) + cv::abs(grad_y);

    // printMatDetails(grad_x, "grad_x");
    // printMatDetails(grad_y, "grad_y");
    // printMatDetails(grad_m, "grad_m");

    // Computing orientations
    cv::Mat grad_o;
    cv::phase(grad_x, grad_y, grad_o, true);

    // printMatDetails(grad_o, "grad_o");

    // Quantizing orientations into bins.
    double w = 360.0 / (double)nbins;
    grad_o = grad_o / w;

    // printMatDetails(grad_o, "grad_o");

    // Creating the descriptor.
    desc = cv::Mat::zeros(1, nbins + 4 * nbins + 16 * nbins, CV_32F);
    int width = image.cols;
    int height = image.rows;

    // Level 0
    cv::Mat chist = desc.colRange(0, nbins);
    getHistogram(edges, grad_o, grad_m, 0, 0, width, height, chist);

    // Level 1
    chist = desc.colRange(nbins, 2 * nbins);
    getHistogram(edges, grad_o, grad_m, 0, 0, width / 2, height / 2, chist);

    chist = desc.colRange(2 * nbins, 3 * nbins);
    getHistogram(edges, grad_o, grad_m, 0, width / 2, width / 2, height / 2, chist);

    chist = desc.colRange(3 * nbins, 4 * nbins);
    getHistogram(edges, grad_o, grad_m, height / 2, 0, width / 2, height / 2, chist);

    chist = desc.colRange(4 * nbins, 5 * nbins);
    getHistogram(edges, grad_o, grad_m, height / 2, width / 2, width / 2, height / 2, chist);

    // Level 2
    int wstep = width / 4;
    int hstep = height / 4;
    int binPos = 5; // Next free section in the histogram
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            chist = desc.colRange(binPos * nbins, (binPos + 1) * nbins);
            getHistogram(edges, grad_o, grad_m, i * hstep, j * wstep, wstep, hstep, chist);
            binPos++;
        }
    }

    // Normalizing the histogram.
    cv::Mat_<float> sumMat;
    cv::reduce(desc, sumMat, 1, cv::REDUCE_SUM);
    float sum = sumMat.at<float>(0, 0);
    desc = desc / sum;
}

void getFilenames(const std::string& directory, std::vector<std::string>& filenames) {
    using namespace boost::filesystem;
    filenames.clear();
    path dir(directory);

    if (!(fs::exists(directory) && fs::is_directory(directory))) {
        throw std::invalid_argument(directory + " does not exist");
        return;
    }

    for (auto const& entry: fs::recursive_directory_iterator(directory)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".ppm")
            filenames.push_back(entry.path().string());
    }
}

void computePhogImg(std::string& imgfile, std::string& outfile) {
    cv::Mat image = cv::imread(imgfile);
    printMatDetails(image, "Image");
    cv::Mat gdsc;
    computePhog(image, gdsc);
    printMatDetails(gdsc, "Gdsc");

    writeMatToFile(gdsc, outfile);
    std::cout << "Successfully saved " << outfile << std::endl;
}

void computePhogImgdir(std::string& imgdir, std::string& outfile) {
    std::vector<std::string> filenames;
    getFilenames(imgdir, filenames);
    std::cout << "Files found: " << filenames.size() << std::endl;
    cv::Mat descs = cv::Mat::zeros(filenames.size(), 1260, CV_32F);
    int processed_count = 0;

    #pragma omp parallel for
    for (unsigned image_ind = 0; image_ind < filenames.size(); image_ind++) {
        cv::Mat image = cv::imread(filenames[image_ind]);
        cv::Mat desc;
        computePhog(image, desc);
        desc.row(0).copyTo(descs.row(image_ind));
        processed_count++;
        if (processed_count % 500 == 0)
            std::cout << "Processed Count: " << processed_count << std::endl;
    }

    std::cout << "Descs computed: " << descs.size() << std::endl;
    
    long unsigned int desc_length = descs.cols;
    std::filesystem::remove(outfile);

    for (unsigned image_ind = 0; image_ind < filenames.size(); image_ind++) {
        std::vector<float> vec;
        descs.row(image_ind).copyTo(vec);
        fs::path path(filenames[image_ind]);
        std::string key = path.lexically_relative(imgdir).string();
        cnpy::npz_save(outfile, key, &vec[0], {desc_length}, "a");
    }
    
    std::cout << "Successfully saved " << outfile << std::endl;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("phog");
    program.add_argument("imgfile")
        .help("path to img (or dir with -r)");
    program.add_argument("-r", "--recursive")
        .help("process images inside directory")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-o", "--outfile")
        .help("path to output file");

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string imgfile = program.get<std::string>("imgfile");
    if (program["--recursive"] == true) {
        std::cout << "Processing directory: " << imgfile << std::endl;
        std::string outfile = "../data/images.npz";
        if (auto fn = program.present("-o"))
            outfile = *fn;
        std::cout << "Output will be stored at " << outfile << std::endl;
        computePhogImgdir(imgfile, outfile);
    } else {
        std::cout << "Processing image: " << imgfile  << std::endl;
        std::string outfile = "../data/desc_cpp.txt";
        if (auto fn = program.present("-o"))
            outfile = *fn;
        std::cout << "Output will be stored at " << outfile << std::endl;
        computePhogImg(imgfile, outfile);
    }

    return 0;
}
