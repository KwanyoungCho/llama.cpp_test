// simple.cpp - llama.cpp 라이브러리를 사용한 기본 텍스트 생성 예제
// 이 예제는 모델 로드, 토큰화, 추론 과정의 기본적인 흐름을 보여줍니다.

// llama.cpp 라이브러리의 메인 헤더 파일 포함
// 모델 로드, 토큰화, 추론 등에 필요한 모든 API 포함
#include "llama.h"
#include <cstdio>  // 입출력 함수 (printf, fprintf 등)
#include <cstring> // 문자열 처리 함수 (strcmp 등)
#include <string>  // C++ 표준 문자열 클래스
#include <vector>  // C++ 표준 동적 배열 컨테이너

// 프로그램 사용법 출력 함수
// argc: 명령행 인수 개수, argv: 명령행 인수 배열
static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    // 기본 사용법 출력 - 모델 경로, 생성할 토큰 수, GPU 레이어 수, 프롬프트 지정 방법
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

// 메인 함수 - 프로그램 진입점
int main(int argc, char ** argv) {
    // 모델 파일(.gguf) 경로 저장 변수
    std::string model_path;
    // 텍스트 생성을 위한 초기 프롬프트 (기본값: "Hello my name is")
    std::string prompt = "Hello my name is";
    // GPU로 오프로드할 레이어 수 (기본값: 99, 대부분의 모델에서 모든 레이어를 의미)
    int ngl = 99;
    // 생성할 토큰 수 (기본값: 32)
    int n_predict = 32;

    // 명령줄 인수 파싱 블록
    {
        int i = 1; // 첫 번째 인수부터 시작 (0번째는 프로그램 이름)
        for (; i < argc; i++) {
            // -m 옵션: 모델 파일 경로 지정
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    // 다음 인수를 모델 경로로 설정
                    model_path = argv[++i];
                } else {
                    // -m 다음에 인수가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } 
            // -n 옵션: 생성할 토큰 수 지정
            else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        // 문자열을 정수로 변환 (예외 처리 포함)
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        // 변환 실패 시 사용법 출력 후 종료
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    // -n 다음에 인수가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } 
            // -ngl 옵션: GPU로 오프로드할 레이어 수 지정
            else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        // 문자열을 정수로 변환 (예외 처리 포함)
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        // 변환 실패 시 사용법 출력 후 종료
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    // -ngl 다음에 인수가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // 다른 옵션이 아니면 프롬프트로 간주하고 루프 종료
                break;
            }
        }
        
        // 모델 경로가 지정되지 않았으면 사용법 출력 후 종료
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        
        // 남은 인수가 있으면 프롬프트로 설정
        if (i < argc) {
            // 첫 번째 부분은 그대로 설정
            prompt = argv[i++];
            // 나머지 인수들은 공백을 추가하여 연결
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // 모든 가능한 백엔드(CPU, GPU 등) 동적 로드
    // 다양한 하드웨어 가속 옵션을 사용 가능하게 함
    ggml_backend_load_all();

    // 모델 초기화 및 로드 시작

    // 모델 파라미터 구조체를 기본값으로 초기화
    llama_model_params model_params = llama_model_default_params();
    // GPU로 오프로드할 레이어 수 설정
    model_params.n_gpu_layers = ngl;

    // 모델 파일을 메모리에 로드
    // GGUF 포맷으로 저장된 모델 파일을 읽어 가중치, 구조, 하이퍼파라미터 등을 로드
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    // 모델의 어휘 정보 가져오기 (토큰화, 변환에 필요)
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 모델 로드 실패 시 오류 메시지 출력 후 종료
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // 프롬프트 토큰화 시작

    // 프롬프트에 포함된 토큰 수 계산
    // 첫 번째 호출에서는 토큰을 저장하지 않고 개수만 계산
    // 음수 반환 시 절대값이 토큰 수를 의미
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 토큰을 저장할 벡터 할당 (계산된 토큰 수만큼)
    std::vector<llama_token> prompt_tokens(n_prompt);
    // 실제 토큰화 수행 (토큰 ID를 벡터에 저장)
    // 마지막 두 개의 true는 BOS(Begin Of Sequence) 토큰 추가 및 특수 토큰 처리 옵션
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // 컨텍스트 초기화 시작

    // 컨텍스트 파라미터 구조체를 기본값으로 초기화
    llama_context_params ctx_params = llama_context_default_params();
    // 컨텍스트 크기 설정 (프롬프트 토큰 수 + 생성할 토큰 수 - 1)
    // 이 크기는 KV 캐시의 크기를 결정하며, 모델이 "기억"할 수 있는 최대 토큰 수
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // 단일 llama_decode 호출에서 처리할 수 있는 최대 토큰 수
    // 배치 크기를 프롬프트 크기로 설정하여 프롬프트를 한 번에 처리
    ctx_params.n_batch = n_prompt;
    // 성능 측정 카운터 활성화
    ctx_params.no_perf = false;

    // 모델에서 컨텍스트 생성
    // 컨텍스트는 모델 상태와 KV 캐시를 포함하며, 실제 추론이 이루어지는 공간
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 컨텍스트 생성 실패 시 오류 메시지 출력 후 종료
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // 샘플러 초기화 시작
    // 샘플러는 모델 출력을 기반으로 다음 토큰을 선택하는 알고리즘

    // 샘플러 체인 파라미터 구조체를 기본값으로 초기화
    auto sparams = llama_sampler_chain_default_params();
    // 성능 측정 카운터 활성화
    sparams.no_perf = false;
    // 샘플러 체인 생성 (여러 샘플링 방법을 연결할 수 있는 구조)
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // 탐욕적(greedy) 샘플러 추가 - 항상 가장 높은 확률의 토큰 선택
    // 다른 옵션으로는 top-k, top-p, 온도 조절 등이 있음
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // 프롬프트를 토큰 단위로 출력 (사용자에게 보여주기 위함)
    for (auto id : prompt_tokens) {
        char buf[128]; // 토큰을 텍스트로 변환할 버퍼
        // 토큰 ID를 사람이 읽을 수 있는 텍스트 조각으로 변환
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        // 변환된 텍스트 조각 출력
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // 프롬프트를 위한 배치 준비
    // 배치는 한 번에 모델에 입력할 토큰 집합을 의미
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 메인 생성 루프 시작

    // 성능 측정을 위한 시작 시간 기록 (마이크로초 단위)
    const auto t_main_start = ggml_time_us();
    // 실제로 디코딩(생성)한 토큰 수
    int n_decode = 0;
    // 생성된 새 토큰 ID를 저장할 변수
    llama_token new_token_id;

    // 프롬프트 토큰과 생성할 토큰을 합친 총 토큰 수만큼 반복
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // 현재 배치를 트랜스포머 모델로 평가 (추론 실행)
        // 이 함수는 KV 캐시를 업데이트하고 로짓(각 토큰의 확률 점수)을 계산
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // 처리 완료된 토큰 수만큼 위치 카운터 증가
        n_pos += batch.n_tokens;

        // 다음 토큰 샘플링
        {
            // 샘플러를 사용하여 다음 토큰 선택
            // -1은 배치에서 마지막 토큰의 로짓을 사용하라는 의미
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // 생성 종료(EOG) 토큰인지 확인
            // EOG 토큰이면 생성 루프 종료
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // 생성된 토큰을 텍스트로 변환
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            // 변환된 텍스트 출력
            std::string s(buf, n);
            printf("%s", s.c_str());
            // 출력 버퍼를 즉시 플러시하여 실시간으로 텍스트가 표시되도록 함
            fflush(stdout);

            // 샘플링된 토큰으로 다음 배치 준비 (한 번에 하나의 토큰 처리)
            batch = llama_batch_get_one(&new_token_id, 1);

            // 디코딩한 토큰 수 증가
            n_decode += 1;
        }
    }

    // 줄바꿈으로 생성 완료 표시
    printf("\n");

    // 성능 측정을 위한 종료 시간 기록
    const auto t_main_end = ggml_time_us();

    // 성능 통계 출력 (디코딩한 토큰 수, 소요 시간, 초당 토큰 수)
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    // 샘플러와 컨텍스트의 상세 성능 정보 출력
    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl); // 샘플링 성능 통계
    llama_perf_context_print(ctx);  // 컨텍스트(추론) 성능 통계
    fprintf(stderr, "\n");

    // 리소스 해제 - 역순으로 해제하는 것이 일반적
    llama_sampler_free(smpl); // 샘플러 메모리 해제
    llama_free(ctx);          // 컨텍스트 메모리 해제
    llama_model_free(model);  // 모델 메모리 해제

    return 0; // 정상 종료
}