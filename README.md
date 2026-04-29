# Eroscript Maker V2

동영상을 분석해 `.funscript` 초안을 자동 생성하고, 사용자가 결과를 보면서 보정할 수 있도록 돕는 데스크톱 도구입니다.

이 프로젝트의 현재 목표는 **완전 자동으로 사람이 만든 수준의 최종본을 항상 생성하는 것**이 아니라, AI 추적과 후처리 UI를 이용해 **수정 가능한 고품질 초안**을 빠르게 만드는 것입니다.

## 주요 기능

- 동영상 큐 처리: 여러 영상을 `PROCESSING QUEUE`에 드래그앤드롭하거나 클릭해서 추가할 수 있습니다.
- 초기 추적점 확인: 처리 시작 전에 `Initial Target Review` 창에서 추적할 지점을 선택하고 검증합니다.
- AI 추적 파이프라인: SAM3, DINOv3, optical flow 기반 신호를 조합해 움직임을 분석합니다.
- 실시간 프리뷰: 처리 중 영상 프리뷰, 현재 스트로크 위치, 리소스 사용량을 확인할 수 있습니다.
- 실패 대응: 추적 신뢰도가 떨어지는 구간에서는 재지정 또는 이전 흐름 유지 같은 복구 흐름을 사용합니다.
- FINAL FUNSCRIPT 미리보기: 생성된 액션 그래프를 확인하고 저장 전 후처리를 적용할 수 있습니다.
- 후처리 도구: `Snap Peaks`, `Snap All`, `Normalize`, `Custom Range`, `React` 등으로 결과를 다듬습니다.
- 저장 제어: 후처리 변경은 바로 파일에 저장되지 않으며, `Save`를 눌러야 `.funscript`에 반영됩니다.

## 프로젝트 구조

```text
.
├── main.py                         # 앱 진입점
├── run.bat                         # Windows 실행 스크립트
├── requirements.txt                # Python 의존성
├── core_ai/                        # SAM3, DINOv3, 비디오 처리 파이프라인
├── tracking/                       # 추적, 신뢰도, 복구, funscript 생성 로직
├── ui/                             # CustomTkinter GUI
├── monitoring/                     # CPU/GPU/RAM 리소스 모니터
├── tools/                          # 결과 비교 및 분석 도구
├── Models/                         # AI 모델 파일 위치
└── Video/                          # 입력 영상과 출력 funscript 예시
```

## 요구 사항

- Windows 환경 권장
- Python 3.11 이상 권장
- NVIDIA GPU 권장
- 충분한 VRAM 권장
- 모델 파일:
  - `Models/sam3.pt`
  - `Models/dinov3-vitl16-pretrain-lvd1689m.safetensors`

처음 실행 시 `run.bat`이 `.venv`를 만들고 `requirements.txt`를 설치합니다.

## 실행 방법

가장 쉬운 방법은 프로젝트 폴더에서 `run.bat`을 실행하는 것입니다.

```bat
run.bat
```

`run.bat`이 하는 일:

1. `.venv` 가상환경이 없으면 생성합니다.
2. 가상환경을 활성화합니다.
3. `requirements.txt` 의존성을 설치합니다.
4. `python main.py`로 프로그램을 실행합니다.

수동 실행이 필요하면 다음처럼 실행할 수 있습니다.

```bat
.venv\Scripts\activate
python main.py
```

## 기본 사용법

1. 프로그램을 실행합니다.
2. 왼쪽 `PROCESSING QUEUE` 영역에 영상을 드래그앤드롭합니다.
3. 또는 큐 영역을 클릭해 파일 탐색기에서 영상을 선택합니다.
4. `RUN ANALYSIS`를 누릅니다.
5. `Initial Target Review` 창에서 추적할 위치를 선택합니다.
6. 선택한 지점 검증이 끝나면 처리가 시작됩니다.
7. 처리 중 프리뷰, 스트로크, AI 로딩 상태, 리소스 사용량을 확인합니다.
8. 완료 후 `FINAL FUNSCRIPT` 그래프에서 결과를 확인합니다.
9. 필요한 후처리를 적용합니다.
10. 결과가 마음에 들면 `Save`를 눌러 `.funscript` 파일에 저장합니다.

출력 파일은 기본적으로 입력 영상과 같은 폴더에 생성됩니다.

예:

```text
Video/sample.mp4
Video/sample.funscript
```

## Initial Target Review

처리 시작 전 표시되는 창입니다.

- 자동 추천 포인트를 확인할 수 있습니다.
- 사용자가 직접 추적할 지점을 클릭할 수 있습니다.
- 선택한 지점이 안정적으로 추적 가능한지 검증합니다.
- 검증 중에는 진행률이 표시됩니다.

좋은 추적점을 고르는 기준:

- 화면에서 오래 보이는 지점
- 배경이나 다른 인물과 색/형태가 덜 겹치는 지점
- 실제 스트로크 움직임과 연관성이 높은 지점
- 가려짐이 너무 잦지 않은 지점

## FINAL FUNSCRIPT 후처리

생성된 결과는 `FINAL FUNSCRIPT` 그래프에 표시됩니다.

그래프의 왼쪽 숫자는 스트로크 높이입니다.

- `100`: 최고점
- `50`: 중간
- `0`: 최저점

후처리로 변경된 포인트는 빨간 링으로 표시됩니다. 이 변경은 미리보기 상태이며, `Save`를 눌러야 실제 파일에 저장됩니다.

### Snap Peaks

국소 최고점과 최저점을 0 또는 100에 가깝게 정리합니다.

스트로크의 핵심 상하점만 깔끔하게 맞추고 싶을 때 사용합니다.

### Snap All

중간값까지 포함해 전체 포인트를 기준선 위/아래로 나누어 0 또는 100으로 정리합니다.

예:

```text
0 > 80 > 10 > 90 > 10 > 80
→ 0 > 100 > 0 > 100 > 0 > 100
```

### Normalize 0-100

현재 결과의 최저값을 0, 최고값을 100으로 다시 매핑합니다.

전체 스트로크 범위가 너무 좁게 나온 경우 유용합니다.

### Custom Range

사용자가 지정한 범위로 전체 값을 재매핑합니다.

예:

```text
5,95
```

최저점을 5, 최고점을 95로 맞춥니다.

### React

하단 스트로크에 작은 반동을 추가합니다.

예:

```text
100 > 0 > 100
→ 100 > 0 > 10 > 0 > 100
```

React 옵션:

- `strength`: 반동의 높이입니다.
- `1x / 2x / 3x`: 반동 횟수입니다.
- `Tight / Normal / Loose`: 포인트 사이의 안전 간격입니다.
- `Micro / Quick / Short / Normal`: 반동이 차지하는 시간 폭입니다.

추천 설정:

- 아주 짧은 반동: `Micro + 1x`
- 작은 반동: `Quick + 1x`
- 자연스러운 반동: `Short + 1x`

React는 상단 100 지점이 아니라 하단 0 근처에 도달한 경우에만 적용됩니다.

## Save와 Discard

후처리 버튼을 누르면 그래프에는 바로 반영되지만 파일에는 저장되지 않습니다.

- `Save`: 현재 미리보기 결과를 `.funscript`에 저장합니다.
- `Discard`: 저장하지 않은 후처리 변경을 되돌립니다.

이미 저장된 파일을 되돌리려면 원본 `.funscript` 백업이 필요합니다.

## Resource Monitor

하단 리소스 모니터에서 CPU, RAM, GPU 사용량을 확인할 수 있습니다.

처리 속도가 느려지거나 GPU 메모리가 부족한 경우 이 영역을 확인하면 병목을 파악하는 데 도움이 됩니다.

## 결과 비교 도구

사람이 만든 기준 스크립트가 `Video/HumanMade/`에 있으면 다음 도구로 자동 생성 결과와 비교할 수 있습니다.

```bat
python tools\evaluate_funscript_similarity.py --video-dir Video --human-dir Video\HumanMade --name "영상 파일명"
```

예:

```bat
python tools\evaluate_funscript_similarity.py --video-dir Video --human-dir Video\HumanMade --name "Anna Anon - 300k"
```

## 현재 한계

- 추적 대상이 가려지거나 화면 밖으로 나가면 결과가 흔들릴 수 있습니다.
- 비슷한 색/형태의 대상이 가까이 있으면 마스크나 추적점이 다른 곳으로 튈 수 있습니다.
- 생성된 `.funscript`는 초안으로 보는 것이 좋으며, 최종 품질은 사용자의 검토와 후처리에 영향을 받습니다.
- SAM3, DINOv3 같은 무거운 모델을 사용하므로 환경에 따라 로딩과 처리 시간이 길 수 있습니다.
- 영상별 카메라 움직임, 장면 전환, 신체 가림 정도에 따라 결과 품질 차이가 큽니다.

## 권장 작업 흐름

1. 안정적인 추적점을 선택합니다.
2. 자동 생성 결과를 확인합니다.
3. 스트로크 방향과 타이밍이 맞는지 봅니다.
4. 높낮이가 약하면 `Normalize` 또는 `Custom Range`를 적용합니다.
5. 극점이 애매하면 `Snap Peaks` 또는 `Snap All`을 적용합니다.
6. 하단 반동이 필요하면 `React`를 약하게 적용합니다.
7. 그래프와 영상 리듬이 맞으면 `Save`합니다.
8. 필요하면 OpenFunscripter 같은 외부 편집기에서 최종 수정합니다.

## 주의 사항

- 사용 권한이 있는 영상에만 사용하세요.
- 후처리 결과는 저장 전까지 미리보기 상태입니다.
- 중요한 결과는 `.funscript` 파일을 백업한 뒤 추가 실험하는 것을 권장합니다.

