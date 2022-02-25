#include<vector>
#include<string>
#include<fstream>
#include<algorithm>
#include<sstream>


void readfile(std::vector<int>& dims, std::vector<float>& data_vec, std::string file_path)
{
    std::ifstream file(file_path);
    std::string line;
    int line_num = 0;
    dims.clear();
    data_vec.clear();
    while (std::getline(file, line)){
        std::istringstream ss(line);
        if (line_num == 0){
            for (std::string each; std::getline(ss, each, ','); dims.push_back(std::atoi(each.c_str())));
        }else{
            for (std::string each; std::getline(ss, each, ','); data_vec.push_back(std::atof(each.c_str())));
        }
        line_num ++;
    }    
}