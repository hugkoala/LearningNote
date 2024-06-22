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

void logSec(string msg)
{
#ifdef LOG_SEC
    printf("== Enter:%s ==", msg.c_str());
#endif
}

void logMsg(string msg)
{
    printf("\n==== %s ====\n", msg.c_str());
}

void test_file_operations() {
    std::cout << "Testing file operations..." << std::endl;
    std::ifstream infile("/etc/foo/config.txt");
    if (infile.is_open()) {
        std::cout << "Read from /etc/foo/config.txt" << std::endl;
        infile.close();
    } else {
        std::cerr << "Failed to read from /etc/foo/config.txt:" << strerror(errno)  << std::endl;
    }

    std::ofstream outfile("/tmp/foo_test.txt");
    if (outfile.is_open()) {
        outfile << "Test writing to /tmp/foo_test.txt" << std::endl;
        std::cout << "Wrote to /tmp/foo_test.txt" << std::endl;
        outfile.close();
    } else {
        std::cerr << "Failed to write to /tmp/foo_test.txt:" << strerror(errno) << std::endl;
    }

}

void test_network_operations() {
    std::cout << "Testing network operations..." << std::endl;
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return;
    }
    
    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(80);
    server.sin_addr.s_addr = inet_addr("172.217.160.78"); // example.com

    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        std::cerr << "Failed to connect to example.com" << std::endl;
    } else {
        std::cout << "Connected to example.com" << std::endl;
        close(sock);
    }
}

void test_subprocess_execution() {
    std::cout << "Testing subprocess execution..." << std::endl;
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        execl("/usr/local/bin/foobar", "foobar", NULL);
        std::cerr << "Failed to execute /usr/local/bin/foobar" << std::endl;
        _exit(1);
    } else if (pid > 0) {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            std::cout << "/usr/local/bin/foobar exited with status " << WEXITSTATUS(status) << std::endl;
        } else {
            std::cerr << "/usr/local/bin/foobar did not exit normally" << std::endl;
        }
    } else {
        std::cerr << "Failed to fork" << std::endl;
    }
}

void blacklist()
{
    logMsg("Case: read /tmp/read.txt");
    ifstream ifs;
    ifs.open("/tmp/read.txt");
    if (ifs.is_open()) {
        string str;
        getline(ifs, str);
        printf("content:%s\n", str.c_str());
    } else {
        printf("read fail:%s\n", strerror(errno));
    }
    ifs.close();

    logMsg("Case: write /tmp/write.txt");
    ofstream ofs;
    ofs.open("/tmp/write.txt");
    if (ofs.is_open()) {
        try {
            char buf[1024] = {0};
            snprintf(buf, sizeof(buf), "/tmp/write.txt write successful");
            ofs.write(buf, strlen(buf));
            printf("write successful\n");
        } catch (exception &e) {
            printf("write fail:%s", e.what());
        }
    } else {
        printf("open fail:%s\n", strerror(errno));
    }
    ofs.close();

    logMsg("Case: read /tmp/write.txt");
    ifs.open("/tmp/write.txt");
    if (ifs.is_open()) {
        string str;
        getline(ifs, str);
        printf("content:%s\n", str.c_str());
    } else {
        printf("read fail:%s\n", strerror(errno));
    }
    ifs.close();

}

void whitelist()
{
    
}

void profileexample()
{
    test_file_operations();
    test_network_operations();
    test_subprocess_execution();
}

int main(int argc, char *argv[])
{
#ifdef BLACK_LIST
    logSec("Black List");
    blacklist();
#endif

#ifdef WHITE_LIST
    logSec("White List");
    whitelist();
#endif

#ifdef PROFILE_EXAMPLE
    logSec("Profile Example");
    profileexample();
#endif

#ifdef QDMS_EXAMPLE
    logSec("QDMS Example");
#endif
    return 0;
}