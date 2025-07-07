#ifndef SERVER_SOCKET_HPP
#define SERVER_SOCKET_HPP

#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdexcept> // For std::runtime_error
#include <opencv2/opencv.hpp>

class ServerSocket {
public:
    ServerSocket();
    ~ServerSocket();

    bool start(const std::string& ip, int port);
    int acceptClient();
    void sendResponse(int client_sock_fd, const std::string& response);
    std::vector<uchar> receiveFrame(int client_sock_fd);
    void stop();
    std::string getClientAddr() const { return client_ip_addr; }

private:
    int server_sock_fd;
    int client_sock_fd;
    std::string client_ip_addr;
    
    std::vector<uchar> recvAll(int socket_fd, size_t length);
};

#endif // SERVER_SOCKET_HPP
