/*
 Compile with:
 g++ -std=c++11 `pkg-config --cflags opencv` -o example example.cpp draw.cpp `pkg-config --libs opencv`
 */
#include "draw.h" // renderPolyImage(), score()
#include <opencv2/core/core.hpp> // Mat, Point, Scalar
#include <opencv2/highgui/highgui.hpp> // imread(), imwrite(), imshow()
#include <algorithm> //sort()
#include <random> //default_random_engine, uniform_int_distribution<>, uniform_real_distribution<>
#include <iostream> // cerr, cout
#include <sstream> // ostringstream
#include <string> // string
using namespace cv;
using namespace std;

const static int MAX_VERTICES = 10;

struct PolygImg {

    //Functor for the sort function
    bool operator() (PolygImg first, PolygImg second) {
        return (first.fitScore > second.fitScore);
    }
    Mat polyImg;
    int VertCount[100];
    Point PolyPointArr[100][MAX_VERTICES];
    const Point* polys[100];
    Scalar colours[100];
    double fitScore = 0;
};

//Randomly selects parents
PolygImg* parentAssign (vector<PolygImg> &NImgbank, double tempFit);

void crossover (PolygImg &child, PolygImg *parent1, PolygImg *parent2);

//Swap two random polygons in the ordered list
void swapPolys (PolygImg &child);

//Random new polygon
void randNewPoly(PolygImg &child, int xParam, int yParam, double alph);

void addVertex(PolygImg &child, int xPoint, int yPoint);

void removeVert(PolygImg &child);

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Not enough arguments\n";
        return 0;
    }
    int N = atoi(argv[argc - 3]);
    int K = atoi(argv[argc - 2]);
    int E = atoi(argv[argc - 1]);
    int T = (E - N) / K;

    //Uncomment for outputing images
    //int picNum = 0;

    ostringstream fileOut;

    vector<PolygImg> NImgbank; //place to store N Images
    PolygImg polyImg1;

    Mat referenceImage = imread(argv[1]); // read in the file using the OpenCV imread funciton

    default_random_engine randEngine(1); // Create a random generator, seed it with the value 1


    // Create two uniform distributions, one for randomly generating
    // x-values of vertices, one for randomly generating y-values of vertices,
    // within the bounds of the image.
    uniform_int_distribution<int> xValGen(0, referenceImage.cols-1);
    uniform_int_distribution<int> yValGen(0, referenceImage.rows-1);


    // Create distributions for generating color and opacity values
    uniform_int_distribution<int> BGRgen(0,255);
    uniform_real_distribution<double> alphaGen(0,1);

    //vertCount is the number of vertices the polygon at that index has
    for (int i = 0; i < 100; i++) polyImg1.VertCount[i] = 3;

    imshow("Reference Image", referenceImage);

    double totalFitness = 0;

    for (int x = 0; x < N; ++x) { //Randomly generating N images of 100 polygons

        for (int i = 0; i < 100; i++) { //Initialize each point of each polygon randomly
            for (int j = 0; j < polyImg1.VertCount[i]; j++) {
                polyImg1.PolyPointArr[i][j] = Point(xValGen(randEngine), yValGen(randEngine));
            }
        }
        for (int i = 0; i < 100; i++) { // Initialize each pointer to point to a polygon
            polyImg1.polys[i] = &polyImg1.PolyPointArr[i][0];
        }

        for (int i = 0; i < 100; i++) { // Randomly initialize each color value
            polyImg1.colours[i] = Scalar(BGRgen(randEngine), BGRgen(randEngine),
                               BGRgen(randEngine), alphaGen(randEngine));
        }
        polyImg1.polyImg = renderPolyImage(
                                           referenceImage.cols,
                                           referenceImage.rows,
                                           100,
                                           polyImg1.polys,
                                           polyImg1.VertCount,
                                           polyImg1.colours
                                           );

        //Total fitness will be used to randomly choose parents based on fitness
        totalFitness += polyImg1.fitScore = score(polyImg1.polyImg, referenceImage);

        NImgbank.push_back(polyImg1);

        // Display the reference image and the rendered polygon image during runtime
        imshow("PolyImage", polyImg1.polyImg);
        //waitKey(0);
    }

    sort(NImgbank.begin(), NImgbank.end(), polyImg1); //Will put least fit images in back
    double  totalChildFitness = 0;
    PolygImg *parent1, *parent2, child;
    double tempFit;
    int randVar; //random variable used to select mutation

    for (int generation = 0; generation < T; generation++) {
        for (int x = 0; x < K; ++x) {//generating K children

            uniform_real_distribution<double> fitProb(0,totalFitness); //Fitness Probability relation
            tempFit = fitProb(randEngine);
            parent1 = parentAssign(NImgbank, tempFit);
            tempFit = fitProb(randEngine);
            parent2 = parentAssign(NImgbank, tempFit);

            crossover(child, parent1, parent2);

            randVar = rand() % 4;

            if (!randVar) {
                swapPolys(child);//mutation
            }
            else if (randVar == 1) {
                randNewPoly(child, referenceImage.cols, referenceImage.rows, alphaGen(randEngine)); //mutation
            }
            else if (randVar == 2) {
                addVertex(child, xValGen(randEngine), yValGen(randEngine)); //mutation
            }
            else if (randVar == 3) {
                removeVert(child); //mutation
            }

            for (int i = 0; i < 100; i++) { // Initialize each pointer to point to a polygon
                child.polys[i] = &child.PolyPointArr[i][0];
            }

            child.polyImg = renderPolyImage(
                                            referenceImage.cols,
                                            referenceImage.rows,
                                            100,
                                            child.polys,
                                            child.VertCount,
                                            child.colours
                                            );


            //totalChildFitness will add the new fitnesses to NImgbank, before reducing the vector back to
            //size N when we will also remove fitnesses from NImgbank as we remove the K least fit images
            totalChildFitness += child.fitScore = score(child.polyImg, referenceImage);

            NImgbank.push_back(child);
        }

        totalFitness += totalChildFitness;

        //resort NImgbank to find N fittest images from N + K images
        sort(NImgbank.begin(), NImgbank.end(), child);
        for (int x = 0; x < K; x++) {
            totalFitness -= NImgbank[NImgbank.size() - 1].fitScore;
            NImgbank.pop_back();
        }
        totalChildFitness = 0;

        if (!(generation % (1000 / K))) {

            //Uncomment the lines below to save an image of your most fit image for that generation

            //fileOut << picNum++ << ".png";
            //string filename = fileOut.str();
            //imwrite(filename, NImgbank[0].polyImg);
            //fileOut.str("");
            //fileOut.clear();
            cout << NImgbank[0].fitScore << "\n";
        }
    }

    // Use the OpenCV imwrite function to write the generated image to a file
    //imwrite(filename, NImgbank[0].polyImg); //Extension determines write format
    imshow("Reference Image", referenceImage);
    imshow("PolyImage", NImgbank[0].polyImg);
    waitKey(0);
    return 0;
}

//performs a full crossover from the two parents to the child
void crossover (PolygImg &child, PolygImg *parent1, PolygImg *parent2) {
    int C = rand() % 100;

    //Child obtains 0-C vertices from parent1 and C-99 from parent2
    for (int i = 0; i < C; ++i) {
        child.VertCount[i] = parent1->VertCount[i];
    }
    for (int i = C; i < 100; ++i) {
        child.VertCount[i] = parent2->VertCount[i];
    }

    //Child obtains 0-C points from parent1 and C-99 from parent2
    for (int i = 0; i < C; i++) {
        for (int j = 0; j < parent1->VertCount[i]; j++) {
            child.PolyPointArr[i][j] = parent1->PolyPointArr[i][j];
        }
    }
    for (int i = C; i < 100; i++) {
        for (int j = 0; j < parent2->VertCount[i]; j++) {
            child.PolyPointArr[i][j] = parent2->PolyPointArr[i][j];
        }
    }

    //Switching 0-C colors from parent1 and C-99 from parent2
    for (int i = 0; i < C; i++) {
        child.colours[i] = parent1->colours[i];
    }
    for (int i = C; i < 100; i++) {
        child.colours[i] = parent2->colours[i];
    }
}

//Takes the random number tempFit in the vector of fitness and finds two parents based on fitness probability
//When the children are pushed back onto NImgbank, the totalFitness is still equal to the sum of the first N
//images in the vector, so only the first N images are capable of being selected until totalFitness is recalc'd
PolygImg* parentAssign (vector<PolygImg> &NImgbank, double tempFit) {
    double counter = 0;
    for (int y = 0; y < NImgbank.size(); ++y) { //NImgBank.size() = N
        counter += NImgbank[y].fitScore;
        //temp fit is a number between 0 and totalFitness, so counter keeps track of total fitness
        //up to index y in NImgbank, making the probability of selection proportional to fitness
        if (tempFit <= counter) {
            return &NImgbank[y];
        }
    }
    return &NImgbank[NImgbank.size() - 1];
}


//swap polygon at index i and j
void swapPolys (PolygImg &child) {

    //Mandate that the two polygons being swapped are not the same
    int i = rand() % 100;
    int j = i;
    while (i == j) {
        j = rand() % 100;
    }
    swap(child.VertCount[i], child.VertCount[j]);
    swap(child.PolyPointArr[i], child.PolyPointArr[j]);
    swap(child.colours[i], child.colours[j]);
}


void randNewPoly(PolygImg &child, int xParam, int yParam, double alph) {
    int i = rand() % 100;
    for (int j = 0; j < child.VertCount[i]; j++) {
        child.PolyPointArr[i][j] = Point(rand() % xParam, rand() % yParam);
    }
    child.colours[i] = Scalar(rand() % 256, rand() % 256, rand() % 256, alph);
}


void addVertex(PolygImg &child, int xPoint, int yPoint) {
    int i = rand() % 100;
    if (child.VertCount[i] >= MAX_VERTICES) {//Do nothing if you can't add any more vertices
        return;
    }
    child.VertCount[i]++;
    child.PolyPointArr[i][child.VertCount[i] - 1] = Point(xPoint, yPoint);
}


void removeVert(PolygImg &child) {
    int i = rand() % 100;
    if (child.VertCount[i] <= 3) {//Do nothing if you can't delete any vertices
        return;
    }
    child.VertCount[i]--;
}




