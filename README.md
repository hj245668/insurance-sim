![tests](https://github.com/hj245668/insurance-sim/actions/workflows/tests.yml/badge.svg)
# insurance-sim

보험수리 수학(이자론, 생존모델, 보험료, 책임준비금, 손해보험)을 파이썬으로 모델링하고, 
**예제**를 테스트벤치로 검증하는 프로젝트입니다.

> 가상 생명표와 단순화된 가정을 쓴 학습용 코드입니다.
> 실제 보험료 산출이나 상품 판매에 쓰지 않습니다.

## 구조

```
src/insurance_sim/
  interest.py     현가, 기말/기초 연금
  life.py         생명표, 정기·양로보험 APV, 등가원리 보험료, 준비금
  nonlife.py      빈도·심도, 복합포아송, 정규근사, Lundberg 파산확률
  simulation.py   총손해·파산확률 몬테카를로
tests/
  test_interest_life.py   1~4장 예제 + 성질 검증
  test_nonlife_sim.py     5장 예제 + 몬테카를로 대 이론값
```

## 실행

```bash
pip install -r requirements.txt
pytest -v
```

## 테스트가 확인하는 것

- **예제 10개**: 현가, 연금, 생존확률, 정기보험료, 연납보험료, 준비금, 복합포아송, 파산확률
- **수학적 성질**: 등가원리(P·ä = APV), 양로보험 항등식 A = 1 − d·ä, 준비금 V(0) = 0, 사망률이 높을수록 보험료 증가
- **몬테카를로 대조**: 총손해의 평균·분산·99% 분위수, 파산확률을 이론값(Lundberg)과 비교

## 다음 단계

- 실제 경험생명표로 `LifeTable` 교체
- GLM(포아송·감마)과 해석 가능한 모델로 개인별 위험률 추정
- Streamlit 대시보드로 모듈 조립형 보험료 시각화
