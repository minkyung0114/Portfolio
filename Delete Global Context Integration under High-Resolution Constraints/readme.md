
## Executive Summary
고해상도 입력 환경에서 YOLO 기반 객체 검출 모델의 학습 효율 및 전역 문맥 인식 한계를 개선 하기 위해 
고해상도를 이미지를 Patch단위로 학습하면서  전역문맥(Global Context)의 손실을 방지하기 위하여패치가 속한 전체 이미지의 전역구조 정보를 함께 반영하는  Feature-wise Linear Modulation(FILM) 방식으로 통합하였다.
고해상도 특성상 직접적인 글로벌 모델링이 어려운 제약 하에서,
전역 정보를 효율적으로 반영하는 구조를 설계하고 성능 개선을 검증하였다.


## Problem & Constraints
### Problem
고해상도 이미지에서 객체 간 전역적 관계 정보 부족
YOLO 구조 특성상 로컬 receptive field 중심의 feature 학습
단순 해상도 축소 시 작은 객체 및 디테일 손실 발생

### Constraints
입력 해상도 유지 필수 (Downsampling 제한)
실시간/준실시간 추론 요구
YOLO backbone/neck 구조의 대규모 변경은 불가

## Key Design Decisions
1. Global Context를 Attention 대신 FILM으로 통합
2. YOLO 구조를 유지한 조건부 Feature Modulation

## Architecture Overview

- Input: High-resolution image 5120x5120
- Detection Branch:
  - YOLOv5 Backbone
  - FPN / PAN Neck (P3, P4, P5)
- Global Context Branch:
  - ConvNeXt 기반 전역 feature 추출
- Integration Module:
  - FILM-based Feature-wise Modulation
  - Projection Adapter (Additive Injection)
 
- 고해상도 입력 이미지를 다운샘플링한 후,
ConvNeXt backbone을 통해 전역 구조 정보를 포함하는
global feature g를 추출한다.
해당 branch는 Detection Branch와 독립적으로 구성되며,
local feature에 직접적인 attention 연산을 적용하지 않고
전역 문맥 정보를 전달하는 역할을 수행한다.

### Integration into YOLO Neck
추출된 global feature는 YOLOv5 Neck의 P4, P5 feature stage에 주입되며,
다음 두 가지 방식으로 통합된다.

1. FILM-based Feature-wise Modulation  
   - Global feature를 MLP를 통해 (γ, β) 파라미터로 변환
   - 각 feature map에 대해 FiLM(x) = γ ⊙ x + β 형태로 적용
   - 전역 문맥에 따라 local feature 표현을 동적으로 조절

2. Projection Adapter (Additive Injection)  
   - Global feature를 linear/MLP projection을 통해
     YOLO feature 차원으로 사상
   - y = x + g' 형태로 단순 결합
   - 구조 변경과 연산 오버헤드를 최소화한 전역 정보 주입

본 구조는 YOLOv5의 기존 backbone 및 neck 구조를 유지하면서,
고해상도 환경에서 전역 문맥 정보를 효율적으로 반영하도록 설계되었다.
