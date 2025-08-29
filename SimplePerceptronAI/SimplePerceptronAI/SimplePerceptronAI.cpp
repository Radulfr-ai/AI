#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <map>
#include <set>
#include <cctype>
#include <cmath>
#include <limits>



// -------------------- Perceptron --------------------

class Perceptron {

private:

    std::vector<double> weights;
    double bias;
    double learnRate;



    static double dotProduct(const std::vector<double>& a, const std::vector<double>& b) 
    {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;

    }



    static int step(double x) 
    {
        return x >= 0.0 ? 1 : 0;

    }



public:

    Perceptron(size_t featureCount, double rate = 0.1)
        : weights(featureCount, 0.0), bias(0.0), learnRate(rate) {
    }



    int predict(const std::vector<double>& features) const 
    {
        double s = dotProduct(weights, features) + bias;
        return step(s);
    }



    void fit(const std::vector<std::vector<double>>& X,
        const std::vector<int>& y,
        int epochs) {
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            int errors = 0;
            for (size_t i = 0; i < X.size(); ++i) {
                int guess = predict(X[i]);
                int delta = y[i] - guess;
                if (delta != 0) {
                    for (size_t j = 0; j < weights.size(); ++j) 
                    {
                        weights[j] += learnRate * delta * X[i][j];
                    }
                    bias += learnRate * delta;
                    ++errors;
                }
            }
            std::cout << "epoch " << epoch << " errors " << errors << "\n";
            if (errors == 0) 
            {
                std::cout << "converged early at epoch " << epoch << "\n";
                break;
            }
        }
    }
    void printParams() const {
        std::cout << "weights: ";
        for (double w : weights) std::cout << std::fixed << std::setprecision(3) << w << " ";
        std::cout << " bias: " << std::fixed << std::setprecision(3) << bias << "\n";
    }
};



void runPerceptronDemo() 
{
    std::vector<std::vector<double>> X = 
    {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<int> y = { 0, 1, 1, 1 };
    Perceptron p(2, 0.2);
    std::cout << "training perceptron on OR logic\n";
    p.fit(X, y, 25);
    p.printParams();
    std::cout << "\npredictions\n";
    for (size_t i = 0; i < X.size(); ++i) 
    {
        int out = p.predict(X[i]);
        std::cout << X[i][0] << " " << "OR" << " " << X[i][1] << " = " << out << " (target " << y[i] << ")\n";
    }
    std::vector<double> custom = { 1.0, 0.0 };
    int guess = p.predict(custom);
    std::cout << "\ncustom input 1 0 -> " << guess << "\n";
}
// -------------------- Naive Bayes Text --------------------
class Bayes {
private:
    std::map<std::string, int> bagPos;
    std::map<std::string, int> bagNeg;
    std::set<std::string> vocab;
    int posDocs = 0;
    int negDocs = 0;
    int totalPosWords = 0;
    int totalNegWords = 0;
   static bool isLetterOrDigit(char ch) 
   {
        return std::isalpha(static_cast<unsigned char>(ch)) ||
            std::isdigit(static_cast<unsigned char>(ch));
    }



    static char toLowerSafe(char ch) 
    {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
   static int addToBag(const std::string& text,
        std::map<std::string, int>& bag,
        std::set<std::string>& vocabRef) {
        std::string token;
        int added = 0;
        for (int i = 0; i < static_cast<int>(text.size()); ++i) 
        {
            char ch = text[i];
            if (isLetterOrDigit(ch)) 
            {
                token.push_back(toLowerSafe(ch));
            }
            else
            {
                if (!token.empty()) 
                {
                    ++bag[token];
                    vocabRef.insert(token);
                    token.clear();
                    ++added;
                }
            }
        }
        if (!token.empty()) 
        {
            ++bag[token];
            vocabRef.insert(token);
            ++added;
        }
        return added;
    }
public:
    void fit(const std::vector<std::pair<std::string, int>>& data) 
    {
        bagPos.clear();
        bagNeg.clear();
        vocab.clear();
        posDocs = 0;
        negDocs = 0;
        totalPosWords = 0;
        totalNegWords = 0;

        for (int i = 0; i < static_cast<int>(data.size()); ++i) 
        {
            const std::string& text = data[i].first;
            int label = data[i].second;
            if (label == 1) 
            {
                totalPosWords += addToBag(text, bagPos, vocab);
                ++posDocs;
            }
            else 
            {
                totalNegWords += addToBag(text, bagNeg, vocab);
                ++negDocs;
            }
        }
    }
    int predict(const std::string& text) const 
    {
        if (posDocs + negDocs == 0) 
        {
            std::cout << "model has no training data\n";
            return 0;
        }
        std::map<std::string, int> local;
        {
            std::string token;
            for (int i = 0; i < static_cast<int>(text.size()); ++i) {
                char ch = text[i];
                if (isLetterOrDigit(ch)) {
                    token.push_back(toLowerSafe(ch));
                }
                else
                {
                    if (!token.empty()) 
                    {
                        ++local[token];
                        token.clear();
                    }
                }
            }
            if (!token.empty()) 
            {
                ++local[token];
            }
        }
        if (local.empty())
        {
            std::cout << "no words found to score\n";
            return 0;
        }

        const int vocabSize = static_cast<int>(vocab.size());
        const double priorPos = std::log(static_cast<double>(posDocs) /
            static_cast<double>(posDocs + negDocs));
        const double priorNeg = std::log(static_cast<double>(negDocs) /
            static_cast<double>(posDocs + negDocs));

        double scorePos = priorPos;
        double scoreNeg = priorNeg;

        for (auto it = local.begin(); it != local.end(); ++it)
        {
            const std::string& w = it->first;
            int times = it->second;

            int cPos = 0;
            int cNeg = 0;
            auto pIt = bagPos.find(w);
            if (pIt != bagPos.end()) cPos = pIt->second;
            auto nIt = bagNeg.find(w);
            if (nIt != bagNeg.end()) cNeg = nIt->second;

            double likePos = std::log((static_cast<double>(cPos) + 1.0) /
                (static_cast<double>(totalPosWords) + static_cast<double>(vocabSize)));
            double likeNeg = std::log((static_cast<double>(cNeg) + 1.0) /
                (static_cast<double>(totalNegWords) + static_cast<double>(vocabSize)));

            for (int k = 0; k < times; ++k) 
            {
                scorePos += likePos;
                scoreNeg += likeNeg;
            }
        }
        return scorePos >= scoreNeg ? 1 : 0;
    }
};
void runTextDemo() 
{
    std::cout << "training a tiny text model (spam vs not spam)\n";
    std::vector<std::pair<std::string, int>> train =
    {
        {"free money claim now", 1},
        {"winner prize cash today", 1},
        {"lowest price deals buy", 1},
        {"urgent offer act now", 1},
        {"lottery jackpot claim ticket", 1},
        {"project meeting at noon", 0},
        {"see you at school event", 0},
        {"family dinner this weekend", 0},
        {"can we talk tomorrow", 0},
        {"notes for class presentation", 0}
    };
    Bayes model;
    model.fit(train);

    std::cout << "type a message and press enter (empty line to exit)\n\n";
    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line.size() == 0) break;
        int guess = model.predict(line);
        if (guess == 1) std::cout << "guess: spam\n\n";
        else std::cout << "guess: not spam\n\n";
    }
    std::cout << "done\n";
}



// -------------------- Main menu --------------------

int main() {
    std::cout << "choose a demo\n";
    std::cout << "1) text naive bayes\n";
    std::cout << "2) perceptron logic\n";
    std::cout << "> ";

    int pick = 0;
    if (!(std::cin >> pick)) return 0;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    if (pick == 1) 
    {
        runTextDemo();
    }
    else if (pick == 2)
    {
        runPerceptronDemo();
    }
    else 
    {
        std::cout << "no such choice\n";
    }

    return 0;