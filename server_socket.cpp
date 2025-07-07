#include "server_socket.hpp"
#include <iostream>
#include <cstring> // For memset
#include <arpa/inet.h> // For htonl, ntohl, inet_ntoa (deprecated) / inet_pton (recommended)
#include <vector>
#include <opencv2/opencv.hpp>

ServerSocket::ServerSocket() : server_sock_fd(-1), client_sock_fd(-1) {}

ServerSocket::~ServerSocket() {
    stop();
}

bool ServerSocket::start(const std::string& ip, int port) {
    server_sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock_fd == -1) {
        perror("socket");
        return false;
    }

    // SO_REUSEADDR 옵션 설정 (서버 재시작 시 포트 바로 사용 가능)
    int optval = 1;
    if (setsockopt(server_sock_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
        perror("setsockopt SO_REUSEADDR");
        close(server_sock_fd);
        return false;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // IP 주소 설정 (0.0.0.0인 경우 INADDR_ANY)
    if (ip == "0.0.0.0") {
        serv_addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
            perror("inet_pton");
            close(server_sock_fd);
            return false;
        }
    }

    if (bind(server_sock_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("bind");
        close(server_sock_fd);
        return false;
    }

    if (listen(server_sock_fd, 1) < 0) { // 동시에 1개의 연결만 처리 (예시)
        perror("listen");
        close(server_sock_fd);
        return false;
    }

    std::cout << "INFO: 서버가 " << ip << ":" << port << "에서 대기 중입니다..." << std::endl;
    return true;
}

int ServerSocket::acceptClient() {
    struct sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    
    client_sock_fd = accept(server_sock_fd, (struct sockaddr *)&client_addr, &client_addr_len);
    if (client_sock_fd < 0) {
        perror("accept");
        client_sock_fd = -1; // 에러 발생 시 -1로 초기화
        return -1;
    }
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);
    client_ip_addr = client_ip;
    return client_sock_fd;
}

void ServerSocket::sendResponse(int client_sock_fd, const std::string& response) {
    uint32_t len = response.length();
    uint32_t net_len = htonl(len); // 네트워크 바이트 순서로 변환

    if (send(client_sock_fd, &net_len, 4, 0) < 0) {
        throw std::runtime_error("응답 길이 전송 실패");
    }
    if (send(client_sock_fd, response.data(), len, 0) < 0) {
        throw std::runtime_error("응답 데이터 전송 실패");
    }
}

std::vector<uchar> ServerSocket::receiveFrame(int client_sock_fd) {
    // 4바이트 길이 정보 수신
    std::vector<uchar> len_buf = recvAll(client_sock_fd, 4);
    if (len_buf.empty()) {
        throw std::runtime_error("클라이언트로부터 프레임 길이를 받지 못했습니다. 연결 종료.");
    }
    
    uint32_t frame_len;
    memcpy(&frame_len, len_buf.data(), 4);
    frame_len = ntohl(frame_len); // 호스트 바이트 순서로 변환

    // 실제 프레임 데이터 수신
    std::vector<uchar> frame_data = recvAll(client_sock_fd, frame_len);
    if (frame_data.empty() && frame_len > 0) {
        throw std::runtime_error("클라이언트로부터 프레임 데이터를 받지 못했습니다.");
    }
    return frame_data;
}

void ServerSocket::stop() {
    if (client_sock_fd != -1) {
        close(client_sock_fd);
        client_sock_fd = -1;
    }
    if (server_sock_fd != -1) {
        close(server_sock_fd);
        server_sock_fd = -1;
    }
    std::cout << "INFO: 서버 소켓이 닫혔습니다." << std::endl;
}

std::vector<uchar> ServerSocket::recvAll(int socket_fd, size_t length) {
    std::vector<uchar> data(length);
    size_t total_received = 0;
    while (total_received < length) {
        ssize_t bytes_received = recv(socket_fd, data.data() + total_received, length - total_received, 0);
        if (bytes_received <= 0) {
            // 0: Connection closed, -1: Error
            return {}; // Empty vector indicates connection closed or error
        }
        total_received += bytes_received;
    }
    return data;
}
