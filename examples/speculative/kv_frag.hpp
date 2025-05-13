#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sstream>   // std::ostringstream
#include <iomanip>   // std::setw
#include "llama.h"          // for llama_context
#include "llama-context.h" // for llama_get_kv_cache_unified
#include "llama-kv-cache.h" // for llama_kv_cache_unified

// ───────────────────────────────────────────────────────────────
// forward decls
struct llama_kv_cache;

struct kv_frag_stat {
    uint32_t total_cells       = 0;
    uint32_t live_cells        = 0;   // at least one seq_id
    uint32_t inner_holes       = 0;   // free cells between first & last live
};

inline kv_frag_stat kv_cache_fragmentation(const llama_context * ctx) {
    const llama_kv_cache_unified * kc = llama_get_kv_cache_unified(ctx);
    if (!kc) return kv_frag_stat();

    kv_frag_stat st;
    st.total_cells = kc->size;

    // find first & last used slot
    uint32_t first = 0, last = 0;
    bool found = false;
    
    for (uint32_t i = 0; i < kc->size; ++i) {
        if (!kc->cells[i].is_empty()) {
            if (!found) {
                first = i;
                found = true;
            }
            last = i;
        }
    }
    if (!found) return st;                 // cache totally empty

    for (uint32_t i = first; i <= last; ++i) {
        if (kc->cells[i].is_empty())
            ++st.inner_holes;
        else
            ++st.live_cells;
    }
    return st;
}

inline float frag_ratio(const kv_frag_stat & s) {
    return (s.live_cells + s.inner_holes) == 0
         ? 0.0f
         : 100.0f * s.inner_holes / float(s.live_cells + s.inner_holes);
}


//-----------------------------------------------------------------
// 내부 유틸 : first / last 사용 셀 인덱스 구하기
//-----------------------------------------------------------------
inline std::pair<uint32_t, uint32_t>
kv_cache_used_range(const llama_kv_cache_unified * kc, bool & any_used) {
    uint32_t first = 0, last = 0;
    any_used = false;

    for (uint32_t i = 0; i < kc->size; ++i) {
        if (!kc->cells[i].is_empty()) {
            if (!any_used) {
                first = i;
                any_used = true;
            }
            last = i;
        }
    }
    return { first, last };
}

//-----------------------------------------------------------------
// 1) 상세 덤프 :  index : 0/1 {seq_ids}
//    • cols : 한 행에 몇 셀씩 배치할지 (기본 64)
//-----------------------------------------------------------------
inline std::string kv_cache_dump(const llama_context * ctx, int cols = 10) {
    const llama_kv_cache_unified * kc = llama_get_kv_cache_unified(ctx);
    if (!kc) return "[unified cache not found]\n";

    bool used = false;
    auto [first, last] = kv_cache_used_range(kc, used);
    if (!used) return "[cache empty]\n";

    std::ostringstream oss;
    oss << "cells " << first << " – " << last << " ("
        << (last - first + 1) << " used slots)\n";

    for (uint32_t i = first; i <= last; ++i) {
        const auto & cell = kc->cells[i];

        // index & 0/1
        oss << std::setw(6) << i << ": " << (cell.is_empty() ? '0' : '1');

        // seq-ids
        if (!cell.is_empty()) {
            oss << " {";
            bool first_id = true;
            for (auto id : cell.seq_id) {
                if (!first_id) oss << ',';
                oss << id;
                first_id = false;
            }
            oss << '}';
        }
        // 동일 행에서 다음 칼럼으로
        if (((i - first + 1) % cols) == 0 || i == last)
            oss << '\n';
        else
            oss << "  ";  // 열 간 구분
    }
    return oss.str();
}

//-----------------------------------------------------------------
// 2) 비트맵 : first – last 범위 내 0/1 문자열만
//-----------------------------------------------------------------
inline std::string kv_cache_bitmap(const llama_context * ctx) {
    const llama_kv_cache_unified * kc = llama_get_kv_cache_unified(ctx);
    if (!kc) return "";

    bool used = false;
    auto [first, last] = kv_cache_used_range(kc, used);
    if (!used) return "";

    std::string bmp;
    bmp.reserve(last - first + 1);
    for (uint32_t i = first; i <= last; ++i) {
        bmp.push_back(kc->cells[i].is_empty() ? '0' : '1');
    }
    return bmp;
}


inline std::string kv_cache_table(const llama_context * ctx,
                                          int cols  = 5,   // 한 행에 몇 칸
                                          int colw  = 26)  // 한 칸 폭
{
    const auto * kc = llama_get_kv_cache_unified(ctx);
    if (!kc) return "[unified cache not found]\n";

    // 사용 영역(first~last) 구하기
    bool used = false;
    uint32_t first = 0, last = 0;
    for (uint32_t i = 0; i < kc->size; ++i) {
        if (!kc->cells[i].is_empty()) {
            if (!used) first = i;
            last = i;
            used = true;
        }
    }
    if (!used) return "[kv cache empty]\n";

    std::ostringstream oss;
    oss << "KV cache table:  (빈칸=x)\n\n";

    for (uint32_t i = first; i <= last; ++i) {
        const auto & cell = kc->cells[i];

        std::ostringstream one;
        one << '[' << std::setw(3) << i << "] ";

        if (cell.is_empty()) {
            one << 'x';
        } else {
            one << cell.pos << " (";
            bool first_id = true;
            for (auto sid : cell.seq_id) {
                if (!first_id) one << ',';
                one << sid;
                first_id = false;
            }
            one << ')';
        }

        // 고정 폭으로 내보내기
        oss << std::left << std::setw(colw) << one.str();

        // cols 칸마다 줄바꿈
        if (((i - first + 1) % cols) == 0 || i == last)
            oss << '\n';
    }
    return oss.str();
}


// CSV 헤더는 한 번만 찍도록 static 플래그 사용
inline void save_frag(const char * kind,          // "draft" | "target"
                      int          seq_len,       // 현재 컨텍스트 길이
                      const llama_context * ctx)  // 측정할 컨텍스트
{
    // 1) 목적 파일 경로 선택
    const char * path = std::strcmp(kind, "draft") == 0
                        ? "kv_frag_draft.csv"
                        : "kv_frag_target.csv";

    // 2) 헤더는 파일마다 첫 번만 출력
    static bool first_draft  = true;
    static bool first_target = true;
    bool & first = (std::strcmp(kind, "draft") == 0) ? first_draft : first_target;

    FILE * fp = std::fopen(path, "a");
    if (!fp) { std::perror(path); return; }

    if (first) {
        std::fprintf(fp, "seq_len,total,ratio\n");
        first = false;
    }

    // 3) 조각률 계산
    kv_frag_stat st = kv_cache_fragmentation(ctx);
    uint32_t total  = st.live_cells + st.inner_holes;

    std::fprintf(fp, "%d,%u,%.3f\n", seq_len, total, frag_ratio(st));
    std::fclose(fp);
}

// ---------------------------------------------------------------
// decoding time을 csv 파일에 저장하는 함수
// ---------------------------------------------------------------
inline void save_decode_time(int n_past_tgt,           // 타겟 모델 컨텍스트 위치 (token position)
                            float decode_time_ms,     // 디코딩에 소요된 시간 (밀리초)
                            const llama_context * ctx_tgt) // 타겟 컨텍스트 (do_defrag 상태 확인용)
{
    const char* path = "decode_time.csv";
    
    // 헤더는 파일이 존재하지 않을 때만 출력
    static bool first_write = true;
    
    // 파일이 존재하는지 확인
    bool file_exists = false;
    {
        FILE* fp = std::fopen(path, "r");
        if (fp) {
            file_exists = true;
            std::fclose(fp);
        }
    }
    
    // do_defrag 플래그 확인
    int do_defrag = 0;
    const llama_kv_cache_unified * kc = llama_get_kv_cache_unified(ctx_tgt);
    if (kc && kc->do_defrag) {
        do_defrag = 1;
    }
    
    // 파일 열기 (추가 모드)
    FILE* fp = std::fopen(path, "a");
    if (!fp) { 
        std::perror(path); 
        return; 
    }
    
    // 헤더 출력 (파일이 존재하지 않는 경우)
    if (!file_exists && first_write) {
        std::fprintf(fp, "n_past_tgt,decode_time_ms,do_defrag\n");
        first_write = false;
    }
    
    // 데이터 출력 (do_defrag 상태 포함)
    std::fprintf(fp, "%d,%.3f,%d\n", n_past_tgt, decode_time_ms, do_defrag);
    std::fclose(fp);
}


// allow_split 플래그 설정
inline void allow_split(llama_context * ctx, bool enable) {
    ctx->kv_cache_allow_split(enable);
}