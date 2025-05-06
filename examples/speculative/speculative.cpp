#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include "kv_frag.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

// 스펙큘레이티브 디코딩에서 허용되는 타겟 모델과 드래프트 모델 간의 최대 어휘 크기 차이
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
// 어휘 검사를 시작할 토큰 ID (처음 몇 개의 특수 토큰은 건너뜀)
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// 스펙큘레이티브 디코딩에서 사용하는 드래프트 시퀀스를 관리하기 위한 구조체
struct seq_draft {
    bool active   = false;  // 시퀀스가 현재 활성화 상태인지 여부
    bool drafting = false;  // 시퀀스가 현재 드래프팅 중인지 여부
    bool skip     = false;  // 시퀀스를 건너뛸지 여부 

    int i_batch_dft = 0;    // 드래프트 모델에서의 배치 인덱스
    std::vector<int> i_batch_tgt;  // 타겟 모델에서의 배치 인덱스 (여러 개 가능)

    std::vector<llama_token> tokens;  // 이 드래프트 시퀀스에 포함된 토큰들
    std::vector<std::vector<llama_token_data>> dists;  // 각 토큰 위치에서의 확률 분포

    struct common_sampler * smpl = nullptr;  // 이 시퀀스의 샘플링을 위한 객체 포인터
};

int main(int argc, char ** argv) {
    common_log_set_verbosity_thold(LOG_DEFAULT_DEBUG);  // = 1  ─▶ DBG 출력 활성화
    /* 옵션: */
    common_log_set_colors    (common_log_main(), true); // ANSI 색상 켜기
    common_log_set_prefix    (common_log_main(), true); // 레벨/시간 접두사
    common_log_set_timestamps(common_log_main(), true); // [M.S.ms.µs] 타임스탬프

    
    common_params params;

    // 온도가 0 이하인 경우에도 후보 확률을 얻기 위해 필요함
    params.sampling.n_probs = 128;

    // 명령줄 인자 파싱
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // 예측할 토큰 수 유효성 검사
    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    // 드래프트 모델이 지정되었는지 확인
    if (params.speculative.model.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // 병렬 드래프팅 시퀀스(트리 브랜치)의 최대 수
    const int n_seq_dft = params.n_parallel;

    // 드래프트 브랜치를 분할하기 위한 확률 임계값 (n_seq_dft > 1인 경우에만 사용)
    const float p_draft_split = params.speculative.p_split;

    // 난수 생성기 초기화
    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // llama.cpp 백엔드 초기화
    llama_backend_init();
    llama_numa_init(params.numa);

    // 모델과 컨텍스트 포인터 초기화
    llama_model * model_tgt = NULL;  // 타겟 모델 (주 모델)
    llama_model * model_dft = NULL;  // 드래프트 모델 (예측용 모델)

    llama_context * ctx_tgt = NULL;  // 타겟 모델의 컨텍스트
    llama_context * ctx_dft = NULL;  // 드래프트 모델의 컨텍스트

    // 타겟 모델 로드
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // 드래프트 모델 로드를 위한 파라미터 설정
    params.devices = params.speculative.devices;
    params.model = params.speculative.model;
    params.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get();
    ctx_dft   = llama_init_dft.context.get();

    // 각 모델의 어휘 가져오기
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    // 어휘 유형 확인 및 로깅
    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    // 타겟 모델과 드래프트 모델의 어휘 유형이 일치하는지 확인
    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    // 특수 토큰(BOS, EOS)이 양쪽 모델에서 일치하는지 확인
    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    // 어휘 크기와 토큰 내용 검증
    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        // 어휘 크기 차이가 허용 범위 내인지 확인
        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        // 개별 토큰 내용이 일치하는지 확인
        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // 프롬프트 토큰화
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    // 컨텍스트 크기 제한 설정
    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    // 입력 토큰 수가 제한을 초과하는지 확인
    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    // 토큰화된 프롬프트 출력
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    // 인코딩 시작 시간 기록
    const auto t_enc_start = ggml_time_us();

    // 양쪽 모델에서 프롬프트 평가
    // 타겟 모델은 마지막 토큰을 별도로 처리 (최적화를 위해)
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));
    // 드래프트 모델은 전체 프롬프트를 한 번에 처리
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));

    // 인코딩 종료 시간 기록
    const auto t_enc_end = ggml_time_us();

    // 각 반복마다 드래프트할 토큰 수
    int n_draft = params.speculative.n_max;

    // 예측 관련 카운터 초기화
    int n_predict = 0;  // 예측해야 할 총 토큰 수
    int n_drafted = 0;  // 지금까지 드래프트된 토큰 수
    int n_accept  = 0;  // 수락된 드래프트 토큰 수

    // 각 모델의 컨텍스트 위치 추적
    int n_past_tgt = inp.size();  // 타겟 모델의 현재 컨텍스트 길이
    int n_past_dft = inp.size();  // 드래프트 모델의 현재 컨텍스트 길이

    // 생성 종료 여부 결정에 사용
    bool has_eos = false;

    // 타겟 모델 샘플링 컨텍스트 (llama_context의 샘플링 인스턴스 재사용)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // 드래프트 시퀀스 데이터 초기화
    std::vector<seq_draft> drafts(n_seq_dft);

    // 각 드래프트 시퀀스에 대한 샘플러 할당
    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    // 배치 처리를 위한 구조체 초기화
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    // 디코딩 시작 시간 기록
    const auto t_dec_start = ggml_time_us();

    // 프롬프트의 마지막 토큰에서 샘플링을 시작하기 위한 초기화
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    
    // 메인 생성 루프
    while (true) {
        // 활성 시퀀스 추적을 위한 집합
        std::set<int> active_seqs = {};

        // 현재 드래프트 시퀀스 출력 (디버깅용)
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;  // 드래프트 토큰 인덱스
        int s_keep = 0;  // 유지할 시퀀스 인덱스

        llama_token token_id;  // 생성된 토큰 ID
        std::string token_str;  // 생성된 토큰 문자열 표현

        // 드래프트 토큰이 더 이상 없거나 토큰 검증에 실패할 때까지 반복
        // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n");
        // LOG_DBG("   sampling 단계 \n");
        // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n\n");
        while (true) {
            // 타겟 모델의 토큰이 드래프트 시퀀스와 일치하는지 확인
            // 확률적 샘플링의 경우, 드래프트 토큰과 타겟 토큰의 매칭 시도
            {
                bool accept = false;  // 드래프트 토큰 수락 여부
                if (params.sampling.temp > 0) {
                    // 확률적 검증 방식 (온도가 0보다 클 때)
                    // 타겟 모델에서 현재 위치의 토큰 분포 샘플링
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                    // 타겟 모델의 토큰 확률 분포 가져오기
                    auto & dist_tgt = *common_sampler_get_candidates(smpl);

                    float p_tgt = 0.0f;  // 타겟 모델에서의 토큰 확률
                    float p_dft = 0.0f;  // 드래프트 모델에서의 토큰 확률
                    
                    // 활성 시퀀스가 있는 동안 반복
                    while (active_seqs.size() > 0) {
                        // 활성 시퀀스에서 무작위로 검증할 시퀀스 선택
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        
                        // 현재 시퀀스의 토큰 인덱스가 범위를 벗어나면 비활성화
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        
                        // 이미 토큰이 수락되었다면 같은 토큰을 가진 시퀀스만 유지
                        if (accept) {
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }
                        
                        //LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        
                        // 0~1 사이의 무작위 값 생성 (확률적 수락을 위함)
                        float r = u_dist(rng);
                        
                        // 드래프트 모델의 토큰 확률 분포 설정
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        // 타겟 모델과 드래프트 모델에서 현재 토큰의 확률 찾기
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        
                        //LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        
                        // 스펙큘레이티브 검증: r ≤ p_tgt/p_dft 일 때 수락
                        // 이는 드래프트 모델에서 확률이 높지만 타겟 모델에서는 낮은 토큰을 필터링함
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;  // 이 시퀀스를 유지
                            accept = true;  // 토큰 수락 표시
                            token_id = drafts[s].tokens[i_dft];  // 수락된 토큰 ID
                            token_str = common_token_to_piece(ctx_tgt, token_id);  // 토큰 문자열 얻기
                            common_sampler_accept(smpl, token_id, true);  // 샘플러에 토큰 수락 알림

                            // LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            // 토큰 거부 - 현재 시퀀스 비활성화
                            //LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;

                            // 잔여 확률 계산 (드래프트 모델의 예측을 제외한 확률 분포)
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // ID별로 분포 정렬
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            // 타겟 확률에서 드래프트 확률을 뺀 잔여 확률 계산
                            // 이는 드래프트 모델의 예측을 제외한 타겟 모델만의 확률 분포를 얻기 위함
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            // 확률 재정규화
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // 확률 내림차순으로 정렬
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        // 현재 검증된 시퀀스 제거
                        active_seqs.erase(s);
                        
                        // 같은 토큰을 가진 다른 시퀀스들의 활성 상태 동기화
                        for(int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // 같은 토큰을 가진 시퀀스의 활성 상태 동기화
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }
                    
                    // 모든 드래프트 토큰이 거부된 경우
                    if (!accept) {
                        // 잔여 확률 분포에서 새 토큰 샘플링
                        //LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        // 확률에 따른 이산 분포 생성 및 샘플링
                        std::discrete_distribution<> dist(probs.begin(), probs.end());
                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;  // 새 토큰 ID
                        common_sampler_accept(smpl, token_id, true);  // 샘플러에 토큰 수락
                        token_str = common_token_to_piece(ctx_tgt, token_id);  // 토큰 문자열 얻기
                    }
                } else {
                    // 그리디 검증 방식 (온도가 0일 때)
                    // 타겟 모델에서 가장 확률이 높은 토큰 선택

                    // LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);
                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    // 모든 활성 시퀀스 확인하여 매칭되는 드래프트 토큰 찾기
                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        // 타겟 토큰이 드래프트 토큰과 일치하면 수락
                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            // LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;  // 이 시퀀스 유지
                            accept = true;  // 토큰 수락
                        } else {
                            // 불일치하는 시퀀스 비활성화
                            drafts[s].active = false;
                        }
                    }
                }
                
                // 토큰이 생성 종료 토큰인지 확인
                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;  // 예측된 토큰 수 증가

                if (accept) {
                    // 드래프트 토큰이 수락된 경우
                    ++n_accept;  // 수락된 토큰 수 증가
                    ++n_past_tgt;  // 타겟 모델 컨텍스트 위치 증가
                    ++n_past_dft;  // 드래프트 모델 컨텍스트 위치 증가
                    ++i_dft;  // 다음 드래프트 토큰으로 이동
                    
                    // 토큰 출력 (색상 지정 여부에 따라)
                    if (params.use_color) {
                        // 시퀀스 번호에 따라 토큰 색상 지정
                        //LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        //LOG("%s", token_str.c_str());
                    }
                    continue;  // 다음 드래프트 토큰 검증 계속
                } else {
                    // 드래프트 토큰이 거부된 경우
                    //LOG("%s", token_str.c_str());
                    break;  // 드래프트 검증 루프 종료
                }
            }
        }

        {
            //LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {   
                // LOG_DBG("\n///////////////////////////////////////////////////////////////////////////// \n");
                // LOG_DBG("KV cache 정리 단계 \n");
                // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n\n");
                
                // LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);
                
                //내가추가--------------------------------------
                // LOG_DBG("\n--- 정리하기 전 draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache

                // KV 캐시를 정리하고 유지할 시퀀스만 보존
                llama_kv_self_seq_keep(ctx_dft, s_keep);
                // LOG_DBG("\n--- 선택한 seq만 남김 : draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache
                llama_kv_self_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_self_seq_keep(ctx_dft, 0);
                // LOG_DBG("\n--- seq를 0으로 변경 : draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache
                
                // LOG_DBG("\n--- 정리하기 전 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
                llama_kv_self_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                // LOG_DBG("\n--- reject seq 제거 : target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
                llama_kv_self_seq_keep(ctx_tgt, s_keep);
                // LOG_DBG("\n--- keep seq 남김 : target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
                llama_kv_self_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_self_seq_keep(ctx_tgt, 0);
                // LOG_DBG("\n--- seq를 0으로 변경 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
            }

            //내가추가--------------------------------------
            {
                auto st_d = kv_cache_fragmentation(ctx_dft);
                auto st_t = kv_cache_fragmentation(ctx_tgt);

                // LOG_DBG("KV-FRAG draft : live %u, hole %u, %.2f%%\n",
                //         st_d.live_cells,  st_d.inner_holes, frag_ratio(st_d));
                // LOG_DBG("KV-FRAG target: live %u, hole %u, %.2f%%\n",
                //         st_t.live_cells,  st_t.inner_holes, frag_ratio(st_t));
                // save_frag("draft", n_past_dft, ctx_dft);
                // save_frag("target", n_past_tgt, ctx_tgt);
            }   
            //----------------------------------------------------


            // 모든 드래프트 시퀀스 초기화
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // 타겟 모델에서 생성된 토큰을 첫 번째 드래프트 시퀀스에 추가
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            // 드래프트 모델 배치 준비
            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);


            //내가추가--------------------------------------
            // LOG_DBG("\n///////////////////////////////////////////////////////////////////////////// \n");
            // LOG_DBG("draft KV cache에 bonus token 추가 \n");
            // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n\n");
            // LOG_DBG("정리하기 전 draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache
            // 드래프트 모델의 KV 캐시에서 불필요한 부분 제거
            llama_kv_self_seq_rm(ctx_dft, 0, n_past_dft, -1);
            // LOG_DBG("통과한 것만 남김 draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache
            // 드래프트 모델로 새 토큰 디코딩
            llama_decode(ctx_dft, batch_dft);
            // LOG_DBG("target token 추가 draft bitmap ---\n%s\n", kv_cache_table(ctx_dft).c_str()); // draft kv cache
            // 드래프트 모델의 컨텍스트 위치 증가
            ++n_past_dft;
            
        }

        // 생성 종료 조건 확인: 예측 토큰 수 또는 EOS 토큰
        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // 첫 번째 드래프트 시퀀스의 샘플러 재설정
        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        // 타겟 모델의 샘플러를 복제하여 사용
        drafts[0].smpl = common_sampler_clone(smpl);

        // 현재 활성 시퀀스 수와 컨텍스트 위치 초기화
        int n_seq_cur  = 1;  // 현재 활성 시퀀스 수
        int n_past_cur = n_past_dft;  // 현재 컨텍스트 위치

        // 모든 드래프트 시퀀스 초기화
        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        // 첫 번째 시퀀스만 활성화 및 드래프팅 상태로 설정
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        // 타겟 모델 배치 초기화 및 첫 번째 토큰 추가
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        //내가추가--------------------------------------
        // LOG_DBG("\n///////////////////////////////////////////////////////////////////////////// \n");
        // LOG_DBG("drafting 단계 \n");
        // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n\n");
        

        // 트리 기반 샘플링을 사용하여 드래프트 모델에서 n_draft 개의 토큰 샘플링
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            // 모든 시퀀스의 skip 플래그 초기화
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            // 각 활성 드래프팅 시퀀스에 대해 처리
            for (int s = 0; s < n_seq_dft; ++s) {
                // 드래프팅 중이 아니거나 건너뛰기로 표시된 시퀀스는 무시
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                // 드래프트 모델에서 현재 시퀀스의 다음 토큰 샘플링
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                // 후보 토큰들의 확률 분포 가져오기
                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl);

                // 디버깅 목적으로 상위 후보 토큰들 출력
                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    //LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    //        k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // 현재 시퀀스 ID를 저장하는 벡터 (브랜치 분할에 사용)
                std::vector<int> sa(1, s);

                // 확률이 충분히 높은 경우 브랜치 분할 시도
                // 상위 2~8번째 토큰 후보(f=1~7)에 대해 브랜치 생성 검토
                for (int f = 1; f < 8; ++f) {
                    // 현재 활성 시퀀스 수가 최대치보다 작고, 후보 토큰의 확률이 임계값보다 높은 경우
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        //LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        // 새 시퀀스를 위한 KV 캐시 cell의 seq id 변경 (브랜치 생성)
                        llama_kv_self_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                        llama_kv_self_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                        // 이 브랜치의 이전 토큰들을 새 브랜치에도 추가
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // 드래프트 상태 복사하여 새 시퀀스 설정
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;  // 현재 반복에서는 처리 건너뜀

                        // 토큰, 확률 분포, 배치 인덱스 등 복사
                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        // 샘플러 초기화 및 복제
                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        // 새 시퀀스 ID를 벡터에 추가
                        sa.push_back(n_seq_cur);

                        // 활성 시퀀스 수 증가
                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // 각 시퀀스(원본 + 분할된 것들)에 드래프트 토큰 추가
                for (int is = 0; is < (int) sa.size(); ++is) {
                    // is 인덱스에 따라 다른 토큰 선택 (트리 샘플링)
                    const llama_token id = cur_p->data[is].id;
                    const int s = sa[is];

                    // 샘플러에 토큰 수락 알림
                    common_sampler_accept(drafts[s].smpl, id, true);

                    // 시퀀스에 토큰 및 확률 분포 추가
                    drafts[s].tokens.push_back(id);
                    // 전체 확률 분포 저장 (나중에 검증에 사용)
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // 타겟 배치에 고유 드래프트 토큰 추가
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);
                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // 드래프트 모델 배치에 토큰 추가 (일괄 디코딩용)
                    drafts[s].i_batch_dft = batch_dft.n_tokens;
                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    // 최대 드래프트 토큰 수에 도달하면 드래프팅 중단
                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // 드래프팅 중인 시퀀스가 없으면 종료
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // 드래프트 모델에서 드래프트된 토큰 평가
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;  // 컨텍스트 위치 증가
            ++n_drafted;   // 드래프트된 토큰 수 증가

            // 최대 드래프트 토큰 수에 도달하면 종료
            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
            
            // LOG_DBG("draft step : %d, draft bitmap ---\n%s\n", i, kv_cache_table(ctx_dft).c_str()); // draft kv cache
        }

        // 타겟 모델에서 드래프트된 토큰 평가
        // LOG_DBG("\n///////////////////////////////////////////////////////////////////////////// \n");
        // LOG_DBG("target 모델에서 드래프트된 토큰 평가 \n");
        // LOG_DBG("///////////////////////////////////////////////////////////////////////////// \n\n");
        {
            // LOG_DBG("\n--- 정리하기 전 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
            // KV 캐시 준비: 첫 번째 시퀀스만 유지하고 나머지 시퀀스에 복사
            llama_kv_self_seq_keep(ctx_tgt, 0);
            // LOG_DBG("\n--- 쓸모없는거 제거 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_self_seq_cp(ctx_tgt, 0, s, -1, -1);
            }
            // LOG_DBG("\n--- draft seq 개수 만큼 추가 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
            
            // 타겟 모델에서 배치 디코딩
            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());

            const auto t_dec_start = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            const auto t_dec_end = ggml_time_us();
            const float decode_time_ms = (t_dec_end - t_dec_start) / 1000.0f;
            // LOG_DBG("target decode time: %f ms\n", decode_time_ms);
            // 디코딩 시간 저장
            // save_decode_time(n_past_tgt, decode_time_ms, ctx_tgt);
            // LOG_DBG("\n--- 드래프트된 토큰 평가 완료 target bitmap ---\n%s\n", kv_cache_table(ctx_tgt).c_str()); // target kv cache
            ++n_past_tgt;  // 타겟 모델 컨텍스트 위치 증가
            
        }

        
        // 첫 번째 토큰은 스펙큘레이션 루프 전에 타겟 모델에서 이미 제안되었으므로 여기서 제거
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                
                continue;
            }
            
            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
            
        }
    }

    // 디코딩 종료 시간 기록
    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
