#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <malloc.h>
#include <psapi.h>
#include <dxgi1_4.h>
#undef small
#undef hyper
#undef interface
#undef near
#undef far
#pragma comment(lib, "advapi32")
#pragma comment(lib, "dxgi")
#else
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#endif

namespace os {

enum class File : intptr_t {};
constexpr uint64_t VRAM_IDLE = 64ull << 20, VRAM_DESKTOP = 1ull << 30;
constexpr File NO_FILE = File(-1);

#ifdef _WIN32

inline File open_read(const char *path) {
    const HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    return h == INVALID_HANDLE_VALUE ? NO_FILE : File((intptr_t)h);
}
inline void close(File f) { if (f != NO_FILE) CloseHandle((HANDLE)(intptr_t)f); }
inline uint64_t size(File f) {
    LARGE_INTEGER n;
    return f != NO_FILE && GetFileSizeEx((HANDLE)(intptr_t)f, &n) ? (uint64_t)n.QuadPart : 0;
}
inline int64_t read_at(File f, void *dst, size_t n, uint64_t off) {
    OVERLAPPED o{};
    o.Offset = (DWORD)off;
    o.OffsetHigh = (DWORD)(off >> 32);
    DWORD got = 0;
    const DWORD want = n > ((size_t)1 << 30) ? (DWORD)1 << 30 : (DWORD)n;
    if (!ReadFile((HANDLE)(intptr_t)f, dst, want, &got, &o)) return GetLastError() == ERROR_HANDLE_EOF ? 0 : -1;
    return (int64_t)got;
}
inline void *map(File f, uint64_t) {
    const HANDLE m = CreateFileMappingA((HANDLE)(intptr_t)f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) return nullptr;
    void *p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    return p;
}
inline void unmap(void *p, size_t) { if (p) UnmapViewOfFile(p); }
inline void *reserve(size_t bytes) { return VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_READWRITE); }
inline bool commit(void *p, size_t bytes) { return VirtualAlloc(p, bytes, MEM_COMMIT, PAGE_READWRITE) != nullptr; }
inline void *anon(size_t bytes) { return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
inline void release(void *p, size_t) { if (p) VirtualFree(p, 0, MEM_RELEASE); }
inline void drop(void *p, size_t bytes) { if (bytes) VirtualFree(p, bytes, MEM_DECOMMIT); }
inline void drop_mapped(void *p, size_t bytes) { if (bytes) VirtualUnlock(p, bytes); }
inline void want(void *p, size_t bytes) {
    WIN32_MEMORY_RANGE_ENTRY r{p, bytes};
    if (bytes) PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
}
inline void random(void *, size_t) {}
inline void huge(void *, size_t, bool) {}
inline void forget(File, uint64_t, uint64_t) {}
inline void *aligned(size_t align, size_t bytes) { return _aligned_malloc(bytes, align); }
inline void aligned_free(void *p) { _aligned_free(p); }
inline uint64_t ram_total() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys : 0;
}
inline double ram_available() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    return GlobalMemoryStatusEx(&m) ? (double)m.ullAvailPhys : 0;
}
inline double ram_held() {
    PROCESS_MEMORY_COUNTERS_EX c{};
    return GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c) ? (double)c.PrivateUsage : 0;
}
inline bool vram_now(int, int, int, uint64_t *, uint64_t *) { return false; }
inline uint64_t vram_others(int, int, int) { return 0; }
inline uint64_t vram_grant(unsigned vendor, const char *gpu, int, int, int, uint64_t free_bytes, uint64_t) {
    IDXGIFactory4 *f = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), (void **)&f))) return free_bytes;
    wchar_t name[128] = {};
    MultiByteToWideChar(CP_UTF8, 0, gpu, -1, name, 127);
    uint64_t named = free_bytes, made = free_bytes;
    bool found = false;
    IDXGIAdapter1 *a = nullptr;
    for (UINT i = 0; f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 d{};
        IDXGIAdapter3 *a3 = nullptr;
        DXGI_QUERY_VIDEO_MEMORY_INFO m{};
        if (SUCCEEDED(a->GetDesc1(&d)) && d.VendorId == vendor && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            SUCCEEDED(a->QueryInterface(__uuidof(IDXGIAdapter3), (void **)&a3))) {
            if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &m))) {
                const uint64_t left = m.Budget > m.CurrentUsage ? m.Budget - m.CurrentUsage : 0;
                made = left < made ? left : made;
                if (!wcscmp(d.Description, name)) { found = true; named = left < named ? left : named; }
            }
            a3->Release();
        }
        a->Release();
    }
    f->Release();
    return found ? named : made;
}
inline int sibling(int cpu) {
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<char> buf(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &bytes)) return -1;
    for (DWORD at = 0; at < bytes;) {
        const auto *info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf.data() + at);
        const KAFFINITY mask = info->Processor.GroupMask[0].Mask;
        if (mask >> cpu & 1) { for (int c = cpu + 1; c < 64; c++) if (mask >> c & 1) return c; return -1; }
        at += info->Size;
    }
    return -1;
}
inline void pin(std::thread &t, int cpu) { SetThreadAffinityMask((HANDLE)t.native_handle(), (DWORD_PTR)1 << cpu); }
inline File from_fd(int64_t file) { return File((intptr_t)file); }
inline size_t missing(const void *, size_t bytes) { return (bytes + 4095) / 4096; }
inline std::string cpu_name() {
    char name[256] = {};
    DWORD n = sizeof name;
    RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString", RRF_RT_REG_SZ, nullptr, name, &n);
    return name;
}
inline std::vector<int> cores() {
    std::vector<int> out;
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
    std::vector<char> buf(bytes);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &bytes)) return out;
    for (DWORD at = 0; at < bytes;) {
        const auto *info = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf.data() + at);
        const KAFFINITY mask = info->Processor.GroupMask[0].Mask;
        for (int c = 0; c < 64; c++) if (mask >> c & 1) { out.push_back(c); break; }
        at += info->Size;
    }
    return out;
}

#else

inline File open_read(const char *path) { return File(::open(path, O_RDONLY)); }
inline void close(File f) { if (f != NO_FILE) ::close((int)f); }
inline uint64_t size(File f) {
    struct stat st;
    return f != NO_FILE && fstat((int)f, &st) == 0 ? (uint64_t)st.st_size : 0;
}
inline int64_t read_at(File f, void *dst, size_t n, uint64_t off) { return (int64_t)pread((int)f, dst, n, (off_t)off); }
inline void *map(File f, uint64_t bytes) {
    void *p = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, (int)f, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline void unmap(void *p, size_t bytes) { if (p) munmap(p, bytes); }
inline void *reserve(size_t bytes) {
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline bool commit(void *, size_t) { return true; }
inline void *anon(size_t bytes) {
    void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? nullptr : p;
}
inline void release(void *p, size_t bytes) { if (p) munmap(p, bytes); }
inline void drop(void *p, size_t bytes) { if (bytes) madvise(p, bytes, MADV_DONTNEED); }
inline void drop_mapped(void *p, size_t bytes) { if (bytes) madvise(p, bytes, MADV_DONTNEED); }
inline void want(void *p, size_t bytes) { if (bytes) madvise(p, bytes, MADV_WILLNEED); }
inline void random(void *p, size_t bytes) { if (bytes) madvise(p, bytes, MADV_RANDOM); }
#if defined(__linux__)
inline void huge(void *p, size_t bytes, bool on) { if (bytes) madvise(p, bytes, on ? MADV_HUGEPAGE : MADV_NOHUGEPAGE); }
inline void forget(File f, uint64_t off, uint64_t n) { posix_fadvise((int)f, (off_t)off, (off_t)n, POSIX_FADV_DONTNEED); }
inline void pin(std::thread &t, int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof set, &set);
}
inline double cgroup_available() {
    std::string path;
    char line[512];
    if (FILE *f = fopen("/proc/self/cgroup", "r")) {
        if (fgets(line, sizeof line, f) && !strncmp(line, "0::", 3)) {
            line[strcspn(line, "\n")] = 0;
            path = std::string("/sys/fs/cgroup") + (line + 3);
        }
        fclose(f);
    }
    for (std::string dir = path; dir.size() >= strlen("/sys/fs/cgroup"); dir = dir.substr(0, dir.find_last_of('/'))) {
        double limit = -1, used = -1, file = 0, shmem = 0;
        if (FILE *f = fopen((dir + "/memory.max").c_str(), "r")) {
            if (fgets(line, sizeof line, f) && strncmp(line, "max", 3)) limit = atof(line);
            fclose(f);
        }
        if (limit < 0) { if (dir == "/sys/fs/cgroup") break; continue; }
        if (FILE *f = fopen((dir + "/memory.current").c_str(), "r")) { if (fgets(line, sizeof line, f)) used = atof(line); fclose(f); }
        if (FILE *f = fopen((dir + "/memory.stat").c_str(), "r")) {
            while (fgets(line, sizeof line, f)) {
                if (!strncmp(line, "file ", 5)) file = atof(line + 5);
                else if (!strncmp(line, "shmem ", 6)) shmem = atof(line + 6);
            }
            fclose(f);
        }
        if (used >= 0) return limit - used + file - shmem;
        break;
    }
    return -1;
}
inline double ram_available() {
    const double cg = cgroup_available();
    if (cg >= 0) return cg;
    double kb = 0;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "MemAvailable:", 13)) { kb = atof(line + 13); break; }
        fclose(f);
    }
    return kb * 1024;
}
inline double ram_held() {
    double anon = 0;
    if (FILE *f = fopen("/proc/self/status", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "RssAnon:", 8)) { anon = atof(line + 8) * 1024; break; }
        fclose(f);
    }
    return anon;
}
inline int sibling(int cpu) {
    int first = -1, second = -1;
    char sep = 0;
    if (FILE *f = fopen(("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list").c_str(), "r")) {
        if (fscanf(f, "%d%c%d", &first, &sep, &second) < 3 || sep != ',') second = -1;
        fclose(f);
    }
    return second;
}
inline std::string cpu_name() {
    std::string name;
    if (FILE *f = fopen("/proc/cpuinfo", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "model name", 10) && strchr(line, ':')) { name = strchr(line, ':') + 2; break; }
        fclose(f);
    }
    return name;
}
inline std::vector<int> cores() {
    std::vector<int> out;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        FILE *f = fopen(("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list").c_str(), "r");
        if (!f) continue;
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == c) out.push_back(c);
        fclose(f);
    }
    return out;
}
#else
inline void huge(void *, size_t, bool) {}
inline void forget(File, uint64_t, uint64_t) {}
inline void pin(std::thread &, int) {}
inline double ram_available() { return 0; }
inline double ram_held() { return 0; }
inline int sibling(int) { return -1; }
inline std::string cpu_name() { return {}; }
inline std::vector<int> cores() { return {}; }
#endif
inline bool vram_now(int domain, int bus, int dev, uint64_t *total, uint64_t *used) {
#if defined(__linux__)
    char path[128];
    uint64_t *const out[2] = {total, used};
    const char *const name[2] = {"mem_info_vram_total", "mem_info_vram_used"};
    for (int i = 0; i < 2; i++) {
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%04x:%02x:%02x.0/%s", domain, bus, dev, name[i]);
        FILE *f = fopen(path, "r");
        if (!f) return false;
        unsigned long long v = 0;
        const bool ok = fscanf(f, "%llu", &v) == 1;
        fclose(f);
        if (!ok) return false;
        *out[i] = v;
    }
    return true;
#else
    (void)domain; (void)bus; (void)dev; (void)total; (void)used;
    return false;
#endif
}
inline uint64_t vram_others(int domain, int bus, int dev) {
    uint64_t sum = 0;
#if defined(__linux__)
    char card[32], path[96], link[64], text[4096];
    snprintf(card, sizeof card, "drm-pdev:\t%04x:%02x:%02x.0", domain, bus, dev);
    std::vector<long> seen;
    DIR *proc = opendir("/proc");
    if (!proc) return 0;
    while (const dirent *p = readdir(proc)) {
        char *end = nullptr;
        const long pid = strtol(p->d_name, &end, 10);
        if (*end || pid <= 0 || pid == (long)getpid()) continue;
        snprintf(path, sizeof path, "/proc/%ld/fd", pid);
        DIR *fds = opendir(path);
        if (!fds) continue;
        while (const dirent *f = readdir(fds)) {
            snprintf(path, sizeof path, "/proc/%ld/fd/%s", pid, f->d_name);
            const ssize_t n = readlink(path, link, sizeof link - 1);
            if (n < 9 || strncmp(link, "/dev/dri/", 9)) continue;
            snprintf(path, sizeof path, "/proc/%ld/fdinfo/%s", pid, f->d_name);
            FILE *info = fopen(path, "r");
            if (!info) continue;
            const size_t got = fread(text, 1, sizeof text - 1, info);
            fclose(info);
            text[got] = 0;
            const char *id = strstr(text, "drm-client-id:"), *vram = strstr(text, "drm-total-vram:");
            if (!strstr(text, card) || !id || !vram) continue;
            const long client = strtol(id + 14, nullptr, 10);
            bool counted = false;
            for (long c : seen) counted = counted || c == client;
            if (counted) continue;
            seen.push_back(client);
            char unit[8] = {};
            unsigned long long v = 0;
            sscanf(vram + 15, "%llu %7s", &v, unit);
            sum += v * (unit[0] == 'K' ? 1024ull : unit[0] == 'M' ? 1048576ull : unit[0] == 'G' ? 1073741824ull : 1ull);
        }
        closedir(fds);
    }
    closedir(proc);
#else
    (void)domain; (void)bus; (void)dev;
#endif
    return sum;
}
inline uint64_t vram_grant(unsigned, const char *, int domain, int bus, int dev, uint64_t free_bytes, uint64_t others) {
    uint64_t total = 0, used = 0;
    if (others <= VRAM_IDLE || !vram_now(domain, bus, dev, &total, &used)) return free_bytes;
    const uint64_t left = total > used ? total - used : 0, most = left < free_bytes ? left : free_bytes;
    const uint64_t room = others > VRAM_DESKTOP ? others : VRAM_DESKTOP;
    return most > room ? most - room : 0;
}
inline File from_fd(int64_t file) { return File((intptr_t)file); }
inline size_t missing(const void *p, size_t bytes) {
    const size_t pages = (bytes + 4095) / 4096;
#if defined(__linux__)
    std::vector<unsigned char> resident(pages);
#else
    std::vector<char> resident(pages);
#endif
    if (mincore((void *)p, bytes, resident.data()) != 0) return pages;
    size_t n = 0;
    for (size_t k = 0; k < pages; k++) n += !(resident[k] & 1);
    return n;
}
inline void *aligned(size_t align, size_t bytes) { return aligned_alloc(align, bytes); }
inline void aligned_free(void *p) { free(p); }
inline uint64_t ram_total() { return (uint64_t)sysconf(_SC_PHYS_PAGES) * (uint64_t)sysconf(_SC_PAGESIZE); }

#endif

inline bool absolute(const std::string &path) {
    return !path.empty() && (path[0] == '/' || path[0] == '\\' || (path.size() > 1 && path[1] == ':'));
}
inline std::string folder(const std::string &path) { return path.substr(0, path.find_last_of("/\\") + 1); }

}
