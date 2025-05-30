# Convolutional AutoEncoder

## 1. 사용할 자료
- CIFAR-10
- torchvision.datasets.CIFAR10
- training 50000, test 10000
- 32 × 32, 컬러 이미지

## 2. 할 작업
### 사전 자료 준비
- 이미지를 정규화하여 [0,1] 또는 [-1, 1] 사이의 값으로 변환
- batch size는 128

## 3. 모형
#### Encoder
- 3개의 합성곱 층, 통과후 채널 갯수가 3 → 32 → 64 → 128로 변화
- 3 × 3 커널을 사용
- 활성화 함수는 ReLU를 사용
- 2 × 2 max pooling을 사용
- 패딩의 크기를 1로 지정하여 이미지의 크기 변화는 pooling에서만 발생
  - 3개의 합성곱 층을 통과하면서 매번 pooling을 주어 최종 크기가 결정

#### Decoder
- Encoder의 대칭이 되는 모형으로 설정 (ConvTranspose2d 사용)
- 생성되는 이미지(최종 출력값)가 3 채널을 갖도록
- 출력되는 값이 입력값과 같은 범위의 값이 되도록 활성화함수 설정
  - [0,1]의 자료를 입력으로 했다면, Sigmoid 함수를 사용
  - [-1, 1]의 자료를 입력했다면, hyperbolic tangent 함수를 사용

#### convAE
- Encoder와 Decoder를 사용해서 정의
- forward 함수의 역할
  - 입력값 → Encoder → 잠재변수 → Decoder → 이미지 생성
  - 잠재변수는 128 × 4 × 4 크기의 벡터

## 4. 모형 학습
- 손실함수: MSE 사용
- 입력 이미지와 생성된 이미지의 차이를 손실로 계산
- 최적화(optimizer): Adam 사용, 학습률은 1e-3, 감소율(weight_decay)은 1e-5로 설정
- 세대(epoch)는 25로 지정하고, 매 5epoch 마다 손실 출력
- 학습 완료 후 학습한 모형 저장:

## 5. 모형 평가
- MSE와 Accuracy 출력
- 10개의 임의의 입력과 생성된 이미지 비교
