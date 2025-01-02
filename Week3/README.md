# 어떤 task를 선택하셨나요 ?
MNLI task를 선택하여 두 문장의 전제와 가설의 관계를 학습하는 task를 선택하였습니다.

# 모델은 어떻게 설계하셨나요 ? 설계한 모델의 입력과 출력 형태가 어떻게 되나요 ?
모델의 입력은 pre-trained된 distilbert encoder를 그대로 사용하였습니다.
출력 형태는 encoder의 마지막 hidden 계층을 가져와 dropout -> Linear로 출력되도록 하였습니다.

# 어떤 pred-trained 모델을 활용하였나요 ?
distilbert 모델을 사용하였습니다.

# 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?
pre-traiend된 모델은 loss curve는 1에 수렴할 정도로 loss가 잘 감소하였습니다.
non-traiend된 모델은 loss curve가 잘 감소하지 않는 문제가 있었습니다.

차이점은 loss의 감소가 pre-trained된 모델이 훨씬 더 잘 감소하였습니다.