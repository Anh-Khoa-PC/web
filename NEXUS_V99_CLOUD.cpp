#include <iostream>
#include <immintrin.h>
#include <thread>
#include <atomic>
#include <vector>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>

using namespace std;

// --- CONFIG ---
#define MAX_HW 78           // Chỉ lấy từ 78 trở xuống
#define NUM_THREADS 2       // GitHub Actions chuẩn thường có 2-Core CPU

atomic<uint64_t> total_hashes{0};
atomic<int> global_best{256};
mutex file_mtx;

// SHA-256 AVX2 MACROS
#define ROR(x, n) _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - n))
#define CH(x, y, z) _mm256_xor_si256(_mm256_and_si256(x, y), _mm256_andnot_si256(x, z))
#define MAJ(x, y, z) _mm256_xor_si256(_mm256_xor_si256(_mm256_and_si256(x, y), _mm256_and_si256(x, z)), _mm256_and_si256(y, z))
#define EP0(x) _mm256_xor_si256(_mm256_xor_si256(ROR(x, 2), ROR(x, 13)), ROR(x, 22))
#define EP1(x) _mm256_xor_si256(_mm256_xor_si256(ROR(x, 6), ROR(x, 11)), ROR(x, 25))
#define SIG0(x) _mm256_xor_si256(_mm256_xor_si256(ROR(x, 7), ROR(x, 18)), _mm256_srli_epi32(x, 3))
#define SIG1(x) _mm256_xor_si256(_mm256_xor_si256(ROR(x, 17), ROR(x, 19)), _mm256_srli_epi32(x, 10))

#define ROUND(a,b,c,d,e,f,g,h,w,k) { \
    __m256i t1 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(h, EP1(e)), CH(e,f,g)), _mm256_add_epi32(_mm256_set1_epi32(k), w)); \
    __m256i t2 = _mm256_add_epi32(EP0(a), MAJ(a,b,c)); \
    d = _mm256_add_epi32(d, t1); \
    h = _mm256_add_epi32(t1, t2); \
}

const uint32_t K[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

void cloud_hunter(uint32_t n, int id) {
    const __m256i H_I[8] = { _mm256_set1_epi32(0x6a09e667), _mm256_set1_epi32(0xbb67ae85), _mm256_set1_epi32(0x3c6ef372), _mm256_set1_epi32(0xa54ff53a), _mm256_set1_epi32(0x510e527f), _mm256_set1_epi32(0x9b05688c), _mm256_set1_epi32(0x1f83d9ab), _mm256_set1_epi32(0x5be0cd19) };

    while(true) {
        __m256i W[64];
        W[0] = _mm256_add_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0,1,2,3,4,5,6,7));
        for(int i=1; i<15; i++) W[i] = _mm256_setzero_si256();
        W[15] = _mm256_set1_epi32(0x80000000);
        for(int i=16; i<64; i++) W[i] = _mm256_add_epi32(_mm256_add_epi32(SIG1(W[i-2]), W[i-7]), _mm256_add_epi32(SIG0(W[i-15]), W[i-16]));

        __m256i a=H_I[0], b=H_I[1], c=H_I[2], d=H_I[3], e=H_I[4], f=H_I[5], g=H_I[6], h=H_I[7];
        for(int i=0; i<64; i++) { ROUND(a,b,c,d,e,f,g,h, W[i], K[i]); __m256i t=h; h=g; g=f; f=e; e=d; d=c; c=b; b=a; a=t; }

        alignas(32) uint32_t r[8][8];
        _mm256_store_si256((__m256i*)r[0], _mm256_add_epi32(a, H_I[0]));
        _mm256_store_si256((__m256i*)r[1], _mm256_add_epi32(b, H_I[1]));
        _mm256_store_si256((__m256i*)r[2], _mm256_add_epi32(c, H_I[2]));
        _mm256_store_si256((__m256i*)r[3], _mm256_add_epi32(d, H_I[3]));
        _mm256_store_si256((__m256i*)r[4], _mm256_add_epi32(e, H_I[4]));
        _mm256_store_si256((__m256i*)r[5], _mm256_add_epi32(f, H_I[5]));
        _mm256_store_si256((__m256i*)r[6], _mm256_add_epi32(g, H_I[6]));
        _mm256_store_si256((__m256i*)r[7], _mm256_add_epi32(h, H_I[7]));

        for(int i=0; i<8; i++) {
            int pop = 0;
            for(int j=0; j<8; j++) pop += __builtin_popcount(r[j][i]);
            
            if(pop < global_best) global_best = pop;

            if(pop <= MAX_HW) {
                lock_guard<mutex> lk(file_mtx);
                ofstream f("GITHUB_HUNTER_LOG.txt", ios::app);
                f << "Nonce: " << (n + i) << " | HW: " << pop << endl;
                f.close();
            }
        }
        n += (NUM_THREADS * 8);
        total_hashes += 8;
    }
}

int main() {
    auto start = chrono::steady_clock::now();
    for(int i=0; i<NUM_THREADS; i++) {
        thread(cloud_hunter, (uint32_t)time(0) + (i * 0x1234567), i).detach();
    }

    while(true) {
        this_thread::sleep_for(chrono::seconds(60)); // In mỗi phút để tránh GitHub ngắt do "no output"
        auto now = chrono::steady_clock::now();
        double sec = chrono::duration<double>(now - start).count();
        double mhs = (total_hashes.load() / sec) / 1000000.0;
        
        cout << "[CLOUD] Time: " << (int)sec/3600 << "h | Speed: " << fixed << setprecision(2) << mhs 
             << " MH/s | Best HW: " << global_best.load() << endl;
    }
    return 0;
}
