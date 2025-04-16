#include "NetworkDistributor.h"
#include <lz4.h>
#include <openssl/evp.h>
#include <asio.hpp>
#include <queue>
#include <shared_mutex>

using asio::ip::tcp;

// Stałe sieciowe
constexpr int NETWORK_THREADS = 4;
constexpr size_t PACKET_SIZE = 1'000'000;
constexpr auto TIMEOUT = std::chrono::seconds(30);

class NetworkDistributor::Impl {
public:
    Impl(const std::vector<std::string>& nodes, const std::string& key)
     : pool(NETWORK_THREADS), cipherKey(key) {
        
        InitCrypto();
        JoinNetwork(nodes);
        StartIOThread();
    }

    void DistributeTasks(const std::vector<KeyRange>& ranges) {
        std::unique_lock lock(taskMutex);
        for(const auto& r : ranges) {
            compressedTasks.push(CompressRange(r));
        }
        lock.unlock();
        taskCV.notify_all();
    }

    std::vector<BitcoinResult> CollectResults() {
        std::vector<BitcoinResult> output;
        std::shared_lock lock(resultMutex);
        output.reserve(results.size());
        for(const auto& r : results) output.push_back(r);
        return output;
    }

private:
    struct NetworkPacket {
        std::array<char, PACKET_SIZE> data;
        size_t size;
    };

    struct NodeConnection {
        tcp::socket socket;
        std::array<char, 128> id;
        std::atomic<bool> busy{false};
    };

    void InitCrypto() {
        ctx = EVP_CIPHER_CTX_new();
        EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL,
                         (const uint8_t*)cipherKey.data(), NULL);
    }

    void JoinNetwork(const std::vector<std::string>& nodes) {
        for(const auto& addr : nodes) {
            auto endpoint = ResolveAddress(addr);
            pool.post([this, endpoint] { ConnectToNode(endpoint); });
        }
    }

    void StartIOThread() {
        ioThread = std::thread([this] {
            asio::io_context::work work(ioContext);
            ioContext.run();
        });
    }

    NetworkPacket CompressRange(const KeyRange& range) {
        NetworkPacket packet;
        auto compressedSize = LZ4_compress_default(
            reinterpret_cast<const char*>(&range),
            packet.data.data(), sizeof(KeyRange), PACKET_SIZE
        );
        packet.size = compressedSize;
        return EncryptPacket(packet);
    }

    NetworkPacket EncryptPacket(const NetworkPacket& input) {
        NetworkPacket output;
        int len;
        EVP_EncryptUpdate(ctx, (uint8_t*)output.data.data(), &len,
                        (const uint8_t*)input.data.data(), input.size);
        EVP_EncryptFinal_ex(ctx, (uint8_t*)output.data.data() + len, &len);
        output.size = len;
        return output;
    }

    void ConnectToNode(const tcp::endpoint& ep) {
        auto socket = std::make_shared<tcp::socket>(ioContext);
        socket->async_connect(ep, [this, socket](auto ec) {
            if(!ec) HandleConnection(socket);
        });
    }

    void HandleConnection(std::shared_ptr<tcp::socket> socket) {
        auto conn = std::make_shared<NodeConnection>(std::move(*socket));
        nodes.push_back(conn);
        
        asio::async_read(*socket, asio::buffer(conn->id), [conn](auto ec, auto) {
            if(!ec) StartHeartbeat(conn);
        });
        
        AssignWork(conn);
    }

    void StartHeartbeat(std::shared_ptr<NodeConnection> conn) {
        asio::steady_timer timer(ioContext);
        auto heartbeat = [conn, &timer](auto ec) {
            if(ec || !conn->socket.is_open()) return;
            conn->socket.async_write_some(asio::buffer("PING", 4), 
                [conn](auto ec, auto) {});
            timer.expires_after(TIMEOUT);
            timer.async_wait(heartbeat);
        };
        timer.async_wait(heartbeat);
    }

    void AssignWork(std::shared_ptr<NodeConnection> conn) {
        std::unique_lock lock(taskMutex);
        taskCV.wait(lock, [this] { return !compressedTasks.empty(); });
        
        auto packet = compressedTasks.front();
        compressedTasks.pop();
        lock.unlock();

        conn->busy = true;
        asio::async_write(conn->socket, asio::buffer(packet.data, packet.size),
            [this, conn](auto ec, auto) {
                if(ec) HandleError(conn);
                conn->busy = false;
                AssignWork(conn);
            });
    }

    void HandleError(std::shared_ptr<NodeConnection> conn) {
        std::unique_lock lock(nodeMutex);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), conn), nodes.end());
        RedistributeTasks(conn);
    }

    void RedistributeTasks(std::shared_ptr<NodeConnection> failedConn) {
        // Implementacja mechanizmu ponownego przydziału zadań
    }

    // Członkowie klasy
    asio::io_context ioContext;
    asio::thread_pool pool;
    std::thread ioThread;
    std::vector<std::shared_ptr<NodeConnection>> nodes;
    std::queue<NetworkPacket> compressedTasks;
    std::vector<BitcoinResult> results;
    
    std::shared_mutex taskMutex;
    std::shared_mutex resultMutex;
    std::shared_mutex nodeMutex;
    std::condition_variable_any taskCV;
    
    EVP_CIPHER_CTX* ctx;
    std::string cipherKey;
};

// Interfejs publiczny
NetworkDistributor::NetworkDistributor(
    const std::vector<std::string>& nodes, const std::string& key)
     : impl(new Impl(nodes, key)) {}

NetworkDistributor::~NetworkDistributor() = default;

void NetworkDistributor::DistributeTasks(
    const std::vector<KeyRange>& ranges) { impl->DistributeTasks(ranges); }

std::vector<BitcoinResult> NetworkDistributor::CollectResults() {
    return impl->CollectResults();
}
