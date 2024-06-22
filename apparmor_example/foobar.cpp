#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>

#include <fstream>
#include <iostream>

using namespace std;

int main()
{
    std::ifstream infile("/var/log/foobar.log");
    if (infile.is_open()) {
        std::cout << "Read from /var/log/foobar.log" << std::endl;
        infile.close();
    } else {
        std::cerr << "Failed to read from /var/log/foobar.log:" << strerror(errno)  << std::endl;
    }

    std::ofstream outfile("/var/log/testfoobar.log");
    if (outfile.is_open()) {
        outfile << "Test writing to /var/log/testfoobar.log" << std::endl;
        std::cout << "Wrote to /var/log/testfoobar.log" << std::endl;
        outfile.close();
    } else {
        std::cerr << "Failed to write to /var/log/testfoobar.log:" << strerror(errno) << std::endl;
    }
}